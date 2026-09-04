"""
crack.py — scale-invariant hybrid CAE solver v2.0 for Brazilian disc splitting
=====================================================
Architecture: symbolic physics (Term A) + NN scale correction (Term B) + Germano self-supervision
Theory: dD = (l/L₀)^d(x) × ΔD_base, with d(x) predicted by the NN from local invariants
Numerics: residual stiffness + nonlocal eps_eq + adaptive damping + exponential clipping

Phase 1 (warmup): pure Mazars damage, no NN
Phase 2 (coupled): NN predicts d(x), Germano self-supervision, scale-corrected damage evolution

Usage: python crack.py
Output: snapshots/step_XXX.png + snapshots/coupled/step_XXX.png
"""

import os
import sys

# Tolerate non-UTF-8 consoles (e.g. Windows GBK): Unicode output must not crash
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import torch
import torch.optim as optim

os.makedirs("snapshots", exist_ok=True)
os.makedirs("snapshots/coupled", exist_ok=True)


# ---------------------------------------------------------------------------
# Single-implementation convergence (P0-1): numerical kernels come from src/,
# this file keeps only the driver logic.
# In-package import (python -m src.solvers.crack / crack-cae after install)
# ---------------------------------------------------------------------------
from src.damage_models import compute_damage_base as _compute_damage_base  # noqa: E402
from src.fem_utils import (  # noqa: E402
    plane_strain_C,
    rect_b_matrix_center as b_matrix_center,
    rect_q4_stiffness_template as q4_stiffness_template,
)
from src.networks import (  # noqa: E402
    PhysicsScaleNetSolid,
    compute_features as _compute_features,
    compute_germano_signal as _compute_germano_signal,
    compute_loss as _compute_loss,
)


# ===========================================================================
# 1. Physics engine base (numerical kernels from src.fem_utils)
# ===========================================================================


# ===========================================================================
# 2. PhysicsScaleNet — scale exponent predictor (Term B, from src.networks)
# ===========================================================================


# ===========================================================================
# 3. CrackSolver — scale-invariant hybrid solver
# ===========================================================================
class CrackSolver:
    def __init__(self, *,
                 Nx, Ny, L_domain, R, E, nu, sigma_t, K_Ic,
                 loading_half_width, n_warmup, n_coupled, disp_step,
                 flat_height=0.0, beta_crack=0.0, a_crack=0.0,
                 hidden_dim=32, lr=2e-3,
                 lam_g=0.3, lam_e=0.5, lam_f=0.3, lam_d=0.2, lam_s=0.1,
                 l_c=0.5, l_d=1.0):
        # ── Mesh ──
        self.Nx, self.Ny = Nx, Ny
        self.L = L_domain; self.R = R
        self.dx = L_domain/(Nx-1); self.dy = L_domain/(Ny-1)
        self.Xc = self.Yc = L_domain/2

        # ── Material ──
        self.E, self.nu = E, nu
        self.sigma_t = sigma_t; self.K_Ic = K_Ic
        self.a_crack = a_crack; self.beta_crack = beta_crack
        self.C = plane_strain_C(E, nu)
        self.k0_unit = q4_stiffness_template(self.dx, self.dy, plane_strain_C(1.0, nu))
        self.B_cen = b_matrix_center(self.dx, self.dy)
        self.eps0 = sigma_t/E
        Gf = K_Ic**2*(1-nu**2)/E
        self.beta_soft = sigma_t/max(Gf/self.dx - sigma_t**2/(2*E), 1e-12)
        print(f"  Mazars: ε₀={self.eps0:.2e}, β={self.beta_soft:.1f}, Gf={Gf:.4e} MPa·mm")

        # ── Numerical parameters ──
        self.residual_stiffness = 1e-6
        self.damping_warmup = 0.3    # uniform damping during warmup (well below 0.8, prevents cascading)
        self.damping_base = 0.5      # far-field damping in the coupled phase
        self.damping_fast = 0.7      # damping in high-drive regions in the coupled phase
        self.exp_clip = 50.0
        self.nonlocal_radius = 1
        self.eps_eq_cap = self.eps0*200

        # ── Scale-correction parameters ──
        self.scale_ratio = 0.3      # l/L₀ (subgrid/characteristic-length ratio)

        # ════════════════════════════════════════════════════════
        # Circular geometric domain masking
        # ════════════════════════════════════════════════════════
        Ne_x, Ne_y = Nx-1, Ny-1
        self.active = []; self.elem_ji = {}
        self.is_active_elem = np.zeros((Ne_y, Ne_x), dtype=bool)
        for j in range(Ne_y):
            for i in range(Ne_x):
                xc = (i+0.5)*self.dx; yc = (j+0.5)*self.dy
                in_circle = (xc-self.Xc)**2 + (yc-self.Yc)**2 <= R**2
                in_flat = (abs(yc-self.Yc) <= R-flat_height) if flat_height>0 else True
                if in_circle and in_flat:
                    e = j*Ne_x + i
                    self.active.append(e); self.elem_ji[e] = (j, i)
                    self.is_active_elem[j, i] = True
        self.n_active = len(self.active)
        print(f"  几何: {self.n_active}/{Ne_x*Ne_y} 活性单元 (R={R} mm)")

        # ── DOF mapping & COO assembly ──
        self.elem_dof_array = np.zeros((self.n_active, 8), dtype=int)
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            n0=j*Nx+i; n1=j*Nx+i+1; n2=(j+1)*Nx+i+1; n3=(j+1)*Nx+i
            self.elem_dof_array[idx] = [2*n0,2*n0+1,2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1]
        N_dof = 2*Nx*Ny; self.N_dof = N_dof
        self.is_active_dof = np.zeros(N_dof, dtype=bool)
        self.is_active_dof[self.elem_dof_array.flatten()] = True
        k0f = self.k0_unit.flatten()
        self._coo_rows = np.zeros(self.n_active*64, dtype=int)
        self._coo_cols = np.zeros(self.n_active*64, dtype=int)
        self._k0_tile = np.tile(k0f, self.n_active)
        for idx in range(self.n_active):
            dofs = self.elem_dof_array[idx]
            self._coo_rows[idx*64:(idx+1)*64] = np.repeat(dofs, 8)
            self._coo_cols[idx*64:(idx+1)*64] = np.tile(dofs, 8)
        inactive = np.where(~self.is_active_dof)[0]
        self._diag_rows = self._diag_cols = inactive
        self._diag_vals = np.ones(len(inactive))

        # ── Boundary conditions ──
        self.top_nodes = []; self.bottom_nodes = []
        self.top_center_node = self.bottom_center_node = None
        min_top = min_bot = 1e9; hw = loading_half_width
        for i in range(Nx):
            xn = i*self.dx
            if abs(xn-self.Xc) > hw: continue
            j_top, j_bot = -1, Ny
            for j in range(Ny):
                if self.is_active_dof[2*(j*Nx+i)]:
                    if j<j_bot: j_bot=j
                    if j>j_top: j_top=j
            if j_top>0:
                node=j_top*Nx+i; self.top_nodes.append(node)
                if abs(xn-self.Xc)<min_top: min_top=abs(xn-self.Xc); self.top_center_node=node
            if j_bot<Ny:
                node=j_bot*Nx+i; self.bottom_nodes.append(node)
                if abs(xn-self.Xc)<min_bot: min_bot=abs(xn-self.Xc); self.bottom_center_node=node
        print(f"  BC: top={len(self.top_nodes)}, bot={len(self.bottom_nodes)}")

        # ── Pre-crack (with smooth transition band) ──
        self.D = np.zeros(self.n_active)
        if a_crack > 0:
            rad = np.radians(beta_crack); cos_b, sin_b = np.cos(rad), np.sin(rad)
            transition_hw = self.dx*2
            for idx, e in enumerate(self.active):
                j,i = self.elem_ji[e]
                xc, yc = (i+0.5)*self.dx, (j+0.5)*self.dy
                dx_c, dy_c = xc-self.Xc, yc-self.Yc
                xp = dx_c*cos_b - dy_c*sin_b
                yp = dx_c*sin_b + dy_c*cos_b
                if abs(xp) < self.dx*0.6 and abs(yp) < a_crack:
                    self.D[idx] = 0.999
                elif abs(yp) < a_crack and abs(xp) < transition_hw:
                    dist = (abs(xp)-self.dx*0.6)/(transition_hw-self.dx*0.6)
                    self.D[idx] = 0.999*np.exp(-3*dist)

        # ── Displacement & mechanical fields ──
        self.U = np.zeros(N_dof)
        self.strains = np.zeros((self.n_active, 3))
        self.stresses = np.zeros((self.n_active, 3))

        # ── Stepping ──
        self.n_warmup = n_warmup; self.n_coupled = n_coupled
        self.disp_step = disp_step
        self.total_steps = n_warmup + n_coupled

        # ── NN & scale field ──
        self.nn = PhysicsScaleNetSolid(input_dim=5, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.nn_active = False
        self.d_field = np.full(self.n_active, -0.5)  # initial elastic anchor
        self.lam_g = lam_g; self.lam_e = lam_e; self.lam_f = lam_f
        self.lam_d = lam_d; self.lam_s = lam_s
        self.l_c = l_c; self.l_d = l_d

        # ── History ──
        self.history = {"load_disp": [], "max_damage": [],
                        "loss_total": [], "mean_d": []}
        self._step_counter = 0

    # =====================================================================
    # I. Elastic solve (Term A)
    # =====================================================================
    def solve_elasticity(self, disp_val):
        D_clipped = np.clip(self.D, 0, 1-self.residual_stiffness)
        scale = np.repeat((1-D_clipped)*self.E + self.residual_stiffness*self.E, 64)
        elem_vals = scale*self._k0_tile
        all_r = np.concatenate([self._coo_rows, self._diag_rows])
        all_c = np.concatenate([self._coo_cols, self._diag_cols])
        all_v = np.concatenate([elem_vals, self._diag_vals])
        K_pen = 1e10*self.E; F = np.zeros(self.N_dof)
        bc_r, bc_c, bc_v = [], [], []
        for n in self.bottom_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
        for n in self.top_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
            F[2*n+1] += K_pen*(-disp_val)
        if self.bottom_center_node is not None:
            bc_r.append(2*self.bottom_center_node); bc_c.append(2*self.bottom_center_node); bc_v.append(K_pen)
        if self.top_center_node is not None:
            bc_r.append(2*self.top_center_node); bc_c.append(2*self.top_center_node); bc_v.append(K_pen)
        all_r=np.concatenate([all_r, np.array(bc_r,int)]); all_c=np.concatenate([all_c, np.array(bc_c,int)])
        all_v=np.concatenate([all_v, np.array(bc_v)])
        K=csr_matrix((all_v,(all_r,all_c)), shape=(self.N_dof,self.N_dof))
        self.U = spsolve(K, F)
        total_fy = 0.0
        for n in self.top_nodes: total_fy += K_pen*(-disp_val-self.U[2*n+1])
        return abs(total_fy)

    # =====================================================================
    # II. Strains and stresses
    # =====================================================================
    def compute_strains_stresses(self):
        u_elem = self.U[self.elem_dof_array]
        self.strains = u_elem @ self.B_cen.T
        self.stresses = (self.strains @ self.C.T)*(1-self.D)[:,None]

    # =====================================================================
    # III. Mazars damage (nonlocal + adaptive damping + relaxed eps cap) → src.damage_models
    # =====================================================================
    def compute_damage_base(self, phase="warmup"):
        delta_D, eps_eq = _compute_damage_base(
            self.strains, self.D, self.eps0, self.beta_soft, self.eps_eq_cap,
            self.exp_clip, self.damping_warmup, self.damping_base, self.damping_fast,
            phase, eps_eq_grid=np.zeros((self.Ny - 1, self.Nx - 1)),
            nonlocal_radius=self.nonlocal_radius,
            active_elem_indices=self.active, elem_ji=self.elem_ji,
            Ny=self.Ny - 1, Nx=self.Nx - 1,
        )
        return delta_D, eps_eq

    # =====================================================================
    # IV. Feature extraction — 5D mechanical invariants → NN input (→ src.networks)
    # =====================================================================
    def compute_features(self):
        return _compute_features(
            self.D, self.strains, self.stresses, self.nu, self.eps0, self.l_c,
            self.Ny - 1, self.Nx - 1, self.active, self.elem_ji, self.dx, self.dy,
        )

    # =====================================================================
    # V. Germano self-supervision signal (→ src.networks)
    # =====================================================================
    def compute_germano_signal(self, delta_D_base):
        return _compute_germano_signal(
            self.strains, self.stresses, self.D, delta_D_base,
            self.Ny - 1, self.Nx - 1, self.active, self.elem_ji,
        )

    # =====================================================================
    # VI. Hybrid loss function (5 terms, → src.networks)
    # =====================================================================
    def compute_loss(self, d_pred, phi_grid_flat, phi_test_flat):
        return _compute_loss(
            d_pred, self.D, phi_grid_flat, phi_test_flat, 3.0,
            self.lam_g, self.lam_e, self.lam_f, self.lam_d, self.lam_s,
            self.l_d, self.dx, self.dy, self.Ny - 1, self.Nx - 1, self.active,
        )

    # =====================================================================
    # VII. Scale-corrected damage update
    # =====================================================================
    def update_damage(self, delta_D_base, d_field_np, use_scaling=True):
        if use_scaling:
            scale = np.clip(self.scale_ratio**d_field_np, 0.1, 10.0)
            dD = scale*delta_D_base
        else:
            dD = delta_D_base
        self.D = np.clip(self.D+dD, 0.0, 0.99999)

    # =====================================================================
    # VIII. Phase 1 — pure-FEM warmup step
    # =====================================================================
    def step_fem_only(self, disp_val):
        F = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base(phase="warmup")
        self.update_damage(delta_D, self.d_field, use_scaling=False)
        return F, eps_eq

    # =====================================================================
    # IX. Phase 2 — coupled step (NN + Germano + scale correction)
    # =====================================================================
    def step_coupled(self, disp_val):
        self._step_counter += 1
        F = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base(phase="coupled")

        feats = self.compute_features()
        d_pred = self.nn(feats)

        H_arr, w_arr, phi, phi_test = self.compute_germano_signal(delta_D)
        phi_test_flat = np.zeros(self.n_active)
        for idx, e in enumerate(self.active):
            j,i = self.elem_ji[e]; phi_test_flat[idx] = phi_test[j,i]

        loss_t, loss_g, loss_e, loss_f, loss_s = self.compute_loss(d_pred, phi, phi_test_flat)
        self.optimizer.zero_grad()
        if torch.isfinite(loss_t):
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
            self.optimizer.step()

        with torch.no_grad():
            d_eval = self.nn(feats).numpy().flatten()
        self.d_field = d_eval
        self.update_damage(delta_D, d_eval, use_scaling=True)
        return F, eps_eq, loss_t

    # =====================================================================
    # X. Step dispatch
    # =====================================================================
    def step(self, disp_val, step_idx):
        if step_idx < self.n_warmup:
            F, eps_eq = self.step_fem_only(disp_val)
            return (F, eps_eq, None), True
        else:
            if not self.nn_active:
                self.nn_active = True
                print(f"\n  >>> 预热完成 (step {step_idx}), 激活 NN + Germano 标度修正\n")
            F, eps_eq, loss_t = self.step_coupled(disp_val)
            return (F, eps_eq, loss_t), False

    # =====================================================================
    # XI. Visualization
    # =====================================================================
    def _get_precrack_line(self, ax):
        """Compute and draw the pre-crack line segment (if present)."""
        if self.a_crack <= 0:
            return None, None

        rad = np.radians(self.beta_crack)
        cos_b, sin_b = np.cos(rad), np.sin(rad)

        # Crack length: 2a (extends a from the center to each side)
        crack_len = 2 * self.a_crack

        # Compute segment start/end points (centered on the domain center)
        # Note: Xc, Yc are the geometric center coordinates of the domain
        # Crack orientation: dx = sin(beta), dy = cos(beta)
        half_dx = (crack_len / 2) * sin_b
        half_dy = (crack_len / 2) * cos_b

        x_start = self.Xc - half_dx
        y_start = self.Yc - half_dy
        x_end = self.Xc + half_dx
        y_end = self.Yc + half_dy

        # Draw a black line on every subplot
        for i_ax, a in enumerate(ax):
            a.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=2.5, zorder=10)

            # Annotate the beta_crack value on the Damage D panel
            if i_ax == 0:
                # Midpoint of the segment (center of the domain)
                x_mid = self.Xc
                y_mid = self.Yc

                # Text position (offset perpendicular to the crack)
                offset = self.R * 0.08
                # Direction perpendicular to the crack (rotated by 90 deg)
                x_text = x_mid + offset * cos_b
                y_text = y_mid - offset * sin_b

                a.text(x_text, y_text, f'β={self.beta_crack:.0f}°',
                       fontsize=11, fontweight='bold',
                       color='black', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.95),
                       zorder=11)

        return x_start, x_end, y_start, y_end

    def visualize(self, step_idx, disp_val, is_warmup):
        Ne_x, Ne_y = self.Nx-1, self.Ny-1
        D_g = np.full((Ne_y, Ne_x), np.nan)
        S_g = np.full((Ne_y, Ne_x), np.nan)
        d_g = np.full((Ne_y, Ne_x), np.nan) if not is_warmup else None
        for idx, e in enumerate(self.active):
            j,i = self.elem_ji[e]
            D_g[j,i] = self.D[idx]
            sxx, syy, sxy = self.stresses[idx]
            szz = self.nu*(sxx+syy)
            S_g[j,i] = np.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2)+3*sxy**2)
            if not is_warmup: d_g[j,i] = self.d_field[idx]

        if is_warmup:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        phase = "Warmup" if is_warmup else "Coupled"
        fig.suptitle(f"Brazilian Disc [{phase}] Step {step_idx:03d}  "
                     f"(d={disp_val*1e3:.1f} um) | max(D)={np.nanmax(D_g):.4f}",
                     fontsize=13, fontweight="bold")

        im0 = axes[0].imshow(D_g, origin='lower', cmap='inferno', vmin=0, vmax=1,
                           extent=[0, self.L, 0, self.L])
        axes[0].set_title("Damage D"); axes[0].set_aspect("equal")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        vmax_s = max(np.nanmax(S_g), 1e-3)
        im1 = axes[1].imshow(S_g, origin='lower', cmap='viridis', vmin=0, vmax=vmax_s,
                           extent=[0, self.L, 0, self.L])
        axes[1].set_title("Von Mises Stress (MPa)"); axes[1].set_aspect("equal")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        if not is_warmup:
            im2 = axes[2].imshow(d_g, origin='lower', cmap='coolwarm', vmin=-3, vmax=-0.3,
                               extent=[0, self.L, 0, self.L])
            axes[2].set_title("Scale exponent d(x)"); axes[2].set_aspect("equal")
            plt.colorbar(im2, ax=axes[2], shrink=0.8)

        # Draw the pre-crack line segment and annotation
        self._get_precrack_line(axes)

        out_dir = "snapshots" if is_warmup else "snapshots/coupled"
        plt.tight_layout(); plt.savefig(f"{out_dir}/step_{step_idx:03d}.png", dpi=120); plt.close()

    def plot_load_displacement(self):
        ld = self.history["load_disp"]
        if len(ld) < 2: return
        disp_vals = [x[0]*1e3 for x in ld]; force_vals = [x[1]/1e3 for x in ld]
        warmup_n = min(self.n_warmup, len(disp_vals))
        plt.figure(figsize=(8, 5))
        plt.plot(disp_vals[:warmup_n], force_vals[:warmup_n], "gray", lw=1, alpha=0.5, label="Warmup")
        plt.plot(disp_vals[warmup_n:], force_vals[warmup_n:], "b-o", ms=2, lw=1.5, label="Coupled (NN+Germano)")
        plt.xlabel("Displacement (μm)"); plt.ylabel("Load (kN)")
        plt.title("Brazilian Disc — Load–Displacement (Scale-Invariant)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("snapshots/load_displacement.png", dpi=120); plt.close()


# ===========================================================================
# 4. Main program
# ===========================================================================
if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # Parameters — Brazilian disc splitting test, ISRM standard
    # ═══════════════════════════════════════════════════════════════════
    # Mesh: Nx=80, Ny=80, L_domain=60 mm, R=25 mm (φ50), flat=0.4 mm
    # Material (sandstone): E=30 GPa, ν=0.25, σ_t=6 MPa, K_Ic=1.0 MPa·√m
    # Crack: 2a=10 mm (a/R=0.2), β=45° (mixed mode I-II)
    # Loading: disp_step=3 μm, 250 steps total → 0.75 mm
    # Switch: NN activated after 50 warmup steps (near damage onset)
    # NN: hidden=32, lr=2e-3, Lamé loss weights
    # ═══════════════════════════════════════════════════════════════════

    # Fixed random seeds (reproducible NN online training)
    torch.manual_seed(2026)
    np.random.seed(2026)

    solver = CrackSolver(
        Nx=80, Ny=80, L_domain=60.0, R=25.0, flat_height=0.4,
        E=30000.0, nu=0.25, sigma_t=6.0, K_Ic=31.62,
        loading_half_width=4.4, disp_step=3.0e-3,
        beta_crack=90.0, a_crack=5.0,
        n_warmup=50, n_coupled=500,     # switch to coupled near damage onset (d~150 μm)
        hidden_dim=32, lr=2e-3,
        lam_g=0.3, lam_e=0.5, lam_f=0.3, lam_d=0.2, lam_s=0.1,
        l_c=0.5, l_d=1.0)

    total = solver.total_steps
    print(f"\n{'='*60}")
    print("  巴西圆盘劈裂 — 尺度不变混合 CAE 求解器 v2.0")
    print("  架构: 符号物理 (Term A) + NN标度修正 (Term B) + Germano自监督")
    print(f"  数值: k_res={solver.residual_stiffness:.0e}  "
          f"warmup_damp={solver.damping_warmup}  "
          f"coupled_damp={solver.damping_base}/{solver.damping_fast}  "
          f"nonlocal_r={solver.nonlocal_radius}")
    print(f"  标度: ratio={solver.scale_ratio} (l/L0),  d(x) in (-inf, -0.5]")
    print(f"{'='*60}")
    print(f"  网格: {solver.Nx}×{solver.Ny}, 活性单元: {solver.n_active}")
    print(f"  裂缝: 2a={2*solver.a_crack:.0f} mm, β={solver.beta_crack}°")
    print(f"  步数: {total} (预热 {solver.n_warmup} + 耦合 {solver.n_coupled})")
    print(f"  最大位移: {total*solver.disp_step:.3f} mm")
    print(f"{'='*60}\n")

    for t in range(total):
        disp = (t+1)*solver.disp_step
        (F, eps_eq, loss_t), is_warmup = solver.step(disp, t)
        solver.history["load_disp"].append((disp, F))
        solver.history["max_damage"].append(np.max(solver.D))
        if not is_warmup and loss_t is not None:
            solver.history["loss_total"].append(
                loss_t.item() if torch.isfinite(loss_t) else float('inf'))
            solver.history["mean_d"].append(np.mean(solver.d_field))

        if t % 10 == 0 or t == total-1:
            d_um = disp*1e3; f_kN = F/1e3
            maxD = solver.D.max(); n_cracked = np.sum(solver.D > 0.99)
            tag = "Warmup" if is_warmup else "Coupled"
            extra = ""
            if not is_warmup and len(solver.history["loss_total"]) > 0:
                lt = solver.history["loss_total"][-1]
                md = solver.history["mean_d"][-1]
                extra = f" | Loss={lt:.2e} | <d>={md:.4f}"
            print(f"  [{tag}] Step {t+1:3d}/{total} | d={d_um:6.1f} um | "
                  f"F={f_kN:7.2f} kN | max(D)={maxD:.4f} | "
                  f"cracked={n_cracked}{extra}")
            solver.visualize(t+1, disp, is_warmup)

    solver.plot_load_displacement()
    print("\n  仿真完成。预热图 → snapshots/step_*.png")
    print("  耦合图 → snapshots/coupled/step_*.png")
    print("  荷载-位移 → snapshots/load_displacement.png")

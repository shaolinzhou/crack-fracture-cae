from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None
    PolyCollection = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None

from dat_parser import DatModel


# 单一实现收敛 (P0-1): 数值内核由共享库 src/ 提供
# 已安装为包时无需改 sys.path; 否则回退到仓库根
try:
    from src import config as _src_probe  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SolverConfig  # noqa: E402
from src.damage_models import (  # noqa: E402
    compute_damage_parameters,
    mazars_damage_target,
)
from src.fem_utils import (  # noqa: E402
    mazars_equivalent_strain,
    plane_strain_C,
    q4_unit_stiffness,
)

if HAS_TORCH:
    from src.networks import PhysicsScaleNetSolid  # noqa: E402
else:

    class PhysicsScaleNetSolid:  # noqa: D101
        """Fallback for the no-PyTorch path (matches previous behaviour)."""

        def __init__(self, *args, **kwargs):  # noqa: D107
            raise RuntimeError("PyTorch is not available")


class DatCrackSolver:
    def __init__(self, model: DatModel, output_dir: str | Path, config: SolverConfig):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.coupled_dir = self.snapshot_dir / "coupled"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.coupled_dir.mkdir(parents=True, exist_ok=True)
        gid_name = config.gid_name or model.path.stem
        self.gid_msh_path = self.output_dir / f"{gid_name}.msh"
        self.gid_res_path = self.output_dir / f"{gid_name}.res"

        self.nodes = model.nodes
        self.elements = model.elements
        self.n_nodes = model.n_nodes
        self.n_active = model.n_elements
        self.N_dof = 2 * self.n_nodes
        self.centers = self.nodes[self.elements].mean(axis=1)
        self.node_tree = cKDTree(self.nodes)
        self.elem_tree = cKDTree(self.centers)

        first_mat = next(iter(model.materials.values()))
        self.E = first_mat.E
        self.nu = first_mat.nu
        self.sigma_t = first_mat.sigma_t
        self.K_Ic = first_mat.K_Ic
        self.C_by_mat = {
            mat_id: plane_strain_C(mat.E, mat.nu) for mat_id, mat in model.materials.items()
        }
        self.materials = model.materials

        self.k0_unit, self.B_cen, self.elem_area = self._precompute_elements()
        self.char_len = float(np.sqrt(np.mean(self.elem_area)))
        self.eps0, self.beta_soft = compute_damage_parameters(
            self.E, self.nu, self.sigma_t, self.K_Ic, self.char_len
        )
        self.eps_eq_cap = self.eps0 * config.eps_eq_cap_factor

        self.elem_dof_array = self._build_element_dofs()
        self._coo_rows, self._coo_cols, self._k0_tile = self._build_coo_cache()

        self.fixed_dofs, self.fixed_values, self.top_disp_dofs, self.top_disp_values = (
            self._build_boundary_conditions()
        )

        self.U = np.zeros(self.N_dof, dtype=float)
        self.D = np.zeros(self.n_active, dtype=float)
        self.strains = np.zeros((self.n_active, 3), dtype=float)
        self.stresses = np.zeros((self.n_active, 3), dtype=float)
        self.d_field = np.full(self.n_active, -0.5, dtype=float)
        self._initialize_wall_damage()

        if HAS_TORCH:
            self.nn = PhysicsScaleNetSolid(input_dim=5, hidden_dim=config.hidden_dim)
            self.optimizer = optim.Adam(self.nn.parameters(), lr=config.lr)
        else:
            self.nn = None
            self.optimizer = None
        self.nn_active = False

        self.history = {"load_disp": [], "max_damage": [], "loss_total": [], "mean_d": []}
        print(
            f"  Mesh: nodes={self.n_nodes}, Q4 elements={self.n_active}, "
            f"char_len={self.char_len:.4g}"
        )
        print(
            f"  Material: E={self.E:g}, nu={self.nu:g}, sigma_t={self.sigma_t:g}, "
            f"K_Ic={self.K_Ic:g}, eps0={self.eps0:.3e}, beta={self.beta_soft:.3g}"
        )
        print(
            f"  BC: fixed dofs={len(self.fixed_dofs)}, prescribed top uy nodes={len(self.top_disp_dofs)}"
        )
        print(f"  Wall nodes={len(model.wall_nodes)}, initial cracked elements={int(np.sum(self.D > 0.9))}")
        if not HAS_TORCH:
            print("  PyTorch not found: coupled phase uses analytical scale field instead of NN training.")
        if not HAS_MATPLOTLIB:
            print("  Matplotlib not found: image snapshots are skipped; CSV history is still written.")
        self.write_gid_mesh()
        self.init_gid_results()

    def _precompute_elements(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_list = []
        b_list = []
        a_list = []
        for eidx, conn in enumerate(self.elements):
            mat_id = int(self.model.element_material_ids[eidx])
            mat = self.materials[mat_id]
            coords = self.nodes[conn]
            k0, Bc, area = q4_unit_stiffness(coords, mat.nu)
            k_list.append(k0)
            b_list.append(Bc)
            a_list.append(area)
        return np.array(k_list), np.array(b_list), np.array(a_list)

    def _build_element_dofs(self) -> np.ndarray:
        dofs = np.zeros((self.n_active, 8), dtype=int)
        for eidx, conn in enumerate(self.elements):
            row = []
            for nidx in conn:
                row.extend([2 * int(nidx), 2 * int(nidx) + 1])
            dofs[eidx] = row
        return dofs

    def _build_coo_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.zeros(self.n_active * 64, dtype=int)
        cols = np.zeros(self.n_active * 64, dtype=int)
        k_tile = np.zeros(self.n_active * 64, dtype=float)
        for eidx, dofs in enumerate(self.elem_dof_array):
            start = eidx * 64
            rows[start : start + 64] = np.repeat(dofs, 8)
            cols[start : start + 64] = np.tile(dofs, 8)
            k_tile[start : start + 64] = self.k0_unit[eidx].ravel()
        return rows, cols, k_tile

    def _node_index(self, node_id: int) -> int:
        return self.model.node_id_to_index[int(node_id)]

    def _build_boundary_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fixed: dict[int, float] = {}
        for node_id, dofs in self.model.constraints.items():
            nidx = self._node_index(node_id)
            if dofs.get("ux", 1.0) == 0.0:
                fixed[2 * nidx] = 0.0
            if dofs.get("uy", 1.0) == 0.0:
                fixed[2 * nidx + 1] = 0.0

        if self.config.auto_anchor_x and not any(dof % 2 == 0 for dof in fixed):
            candidate_ids = list(self.model.constraints) or list(self.model.prescribed_uy)
            if not candidate_ids:
                candidate_ids = [int(self.model.node_ids[np.argmin(self.nodes[:, 1])])]
            candidate_indices = np.array([self._node_index(nid) for nid in candidate_ids], dtype=int)
            coords = self.nodes[candidate_indices]
            ymin = np.min(coords[:, 1])
            bottom = candidate_indices[np.isclose(coords[:, 1], ymin)]
            anchor = int(bottom[np.argmin(np.abs(self.nodes[bottom, 0]))])
            fixed[2 * anchor] = 0.0
            print(
                f"  Auto anchor: UX fixed at node id {int(self.model.node_ids[anchor])} "
                "to remove rigid-body x motion."
            )

        top_dofs = []
        top_vals = []
        for node_id, val in self.model.prescribed_uy.items():
            nidx = self._node_index(node_id)
            top_dofs.append(2 * nidx + 1)
            top_vals.append(float(val))

        return (
            np.array(list(fixed.keys()), dtype=int),
            np.array(list(fixed.values()), dtype=float),
            np.array(top_dofs, dtype=int),
            np.array(top_vals, dtype=float),
        )

    def _initialize_wall_damage(self) -> None:
        wall_ids = [nid for nid in self.model.wall_nodes if nid in self.model.node_id_to_index]
        if not wall_ids:
            return
        wall_idx = [self._node_index(nid) for nid in wall_ids]
        wall_set = set(wall_idx)

        for eidx, conn in enumerate(self.elements):
            if any(int(nidx) in wall_set for nidx in conn):
                self.D[eidx] = 0.999

        if len(wall_idx) >= 2:
            wall_points = self.nodes[wall_idx]
            dists = np.full(self.n_active, np.inf, dtype=float)
            for a, b in zip(wall_points[:-1], wall_points[1:]):
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom <= 1e-30:
                    continue
                t = np.clip(((self.centers - a) @ ab) / denom, 0.0, 1.0)
                proj = a + t[:, None] * ab
                dists = np.minimum(dists, np.linalg.norm(self.centers - proj, axis=1))
            band = max(2.0 * self.char_len, 1e-12)
            transition = 0.999 * np.exp(-3.0 * dists / band)
            self.D = np.maximum(self.D, np.where(dists < band, transition, 0.0))
        self.D = np.clip(self.D, 0.0, 0.999)

    def _nonlocal_average(self, values: np.ndarray, radius: float | None = None) -> np.ndarray:
        radius = radius or (2.5 * self.char_len)
        neighbors = self.elem_tree.query_ball_point(self.centers, radius)
        out = np.empty_like(values)
        for i, ids in enumerate(neighbors):
            out[i] = float(np.mean(values[ids])) if ids else values[i]
        return out

    def solve_elasticity(self, load_factor: float) -> float:
        D_clipped = np.clip(self.D, 0.0, 1.0 - self.config.residual_stiffness)
        mat_E = np.array([self.materials[int(mid)].E for mid in self.model.element_material_ids])
        scale = np.repeat((1.0 - D_clipped + self.config.residual_stiffness) * mat_E, 64)
        elem_vals = scale * self._k0_tile

        K_pen = 1e10 * self.E
        F = np.zeros(self.N_dof, dtype=float)
        bc_dofs = list(self.fixed_dofs) + list(self.top_disp_dofs)
        bc_vals = list(self.fixed_values) + list(load_factor * self.top_disp_values)

        bc_r = np.array(bc_dofs, dtype=int)
        bc_c = np.array(bc_dofs, dtype=int)
        bc_v = np.full(len(bc_dofs), K_pen, dtype=float)
        for dof, val in zip(bc_dofs, bc_vals):
            F[int(dof)] += K_pen * float(val)

        rows = np.concatenate([self._coo_rows, bc_r])
        cols = np.concatenate([self._coo_cols, bc_c])
        vals = np.concatenate([elem_vals, bc_v])
        K = csr_matrix((vals, (rows, cols)), shape=(self.N_dof, self.N_dof))
        self.U = spsolve(K, F)

        total_reaction = 0.0
        target_vals = load_factor * self.top_disp_values
        for dof, val in zip(self.top_disp_dofs, target_vals):
            total_reaction += K_pen * (val - self.U[int(dof)])
        return abs(float(total_reaction))

    def compute_strains_stresses(self) -> None:
        u_elem = self.U[self.elem_dof_array]
        self.strains = np.einsum("ei,eji->ej", u_elem, self.B_cen)
        for eidx, mat_id in enumerate(self.model.element_material_ids):
            C = self.C_by_mat[int(mat_id)]
            self.stresses[eidx] = (C @ self.strains[eidx]) * (1.0 - self.D[eidx])

    def compute_damage_base(self, phase: str) -> tuple[np.ndarray, np.ndarray]:
        exy = self.strains[:, 2] * 0.5
        eps_eq = mazars_equivalent_strain(self.strains[:, 0], self.strains[:, 1], exy)
        eps_eq = self._nonlocal_average(eps_eq)

        D_target = mazars_damage_target(
            eps_eq, self.eps0, self.beta_soft, self.eps_eq_cap,
            exp_clip=self.config.exp_clip,
        )
        driving = np.maximum(D_target - self.D, 0.0)
        if phase == "warmup":
            damping = np.full_like(driving, self.config.damping_warmup)
        else:
            damping = np.where(driving > 0.1, self.config.damping_fast, self.config.damping_base)
        return driving * damping, eps_eq

    def compute_features(self):
        sxx, syy, sxy = self.stresses[:, 0], self.stresses[:, 1], self.stresses[:, 2]
        szz = self.nu * (sxx + syy)
        sm = (sxx + syy + szz) / 3.0
        Sxx, Syy, Szz = sxx - sm, syy - sm, szz - sm
        J2 = 0.5 * (Sxx**2 + Syy**2 + Szz**2) + sxy**2
        seq = np.sqrt(3.0 * J2 + 1e-30)
        eta = sm / (seq + 1e-12)
        J3 = Sxx * Syy * Szz - Szz * sxy**2
        cos_arg = np.clip(27.0 * J3 / (2.0 * seq**3 + 1e-30), -1.0, 1.0)
        theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos_arg)

        exy = self.strains[:, 2] * 0.5
        eps_eq = mazars_equivalent_strain(self.strains[:, 0], self.strains[:, 1], exy)

        neighbor_ids = self.elem_tree.query(self.centers, k=min(7, self.n_active))[1]
        gD = np.zeros(self.n_active, dtype=float)
        for i, ids in enumerate(np.atleast_2d(neighbor_ids)):
            dist = np.linalg.norm(self.centers[ids] - self.centers[i], axis=1)
            diff = np.abs(self.D[ids] - self.D[i])
            mask = dist > 1e-12
            gD[i] = np.max(diff[mask] / dist[mask]) if np.any(mask) else 0.0

        F_np = np.stack(
            [
                self.D,
                np.tanh(eta),
                np.tanh(theta_bar),
                np.tanh(eps_eq / self.eps0 - 1.0),
                np.tanh(self.config.l_c * gD),
            ],
            axis=1,
        )
        if HAS_TORCH:
            return torch.tensor(F_np, dtype=torch.float32)
        return F_np

    def compute_germano_signal(self, delta_D_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exx, eyy, exy = self.strains[:, 0], self.strains[:, 1], self.strains[:, 2] * 0.5
        W = 0.5 * (self.stresses[:, 0] * exx + self.stresses[:, 1] * eyy + 2.0 * self.stresses[:, 2] * exy)
        Y = W / ((1.0 - self.D) ** 2 + 1e-30)
        phi = Y * delta_D_base
        phi_test = self._nonlocal_average(phi, radius=3.0 * self.char_len)
        return phi, phi_test

    def compute_loss(
        self, d_pred, phi: np.ndarray, phi_test: np.ndarray
    ):
        if not HAS_TORCH:
            raise RuntimeError("compute_loss requires PyTorch")
        lambda_L = 3.0
        phi_g_t = torch.tensor(phi, dtype=torch.float32).unsqueeze(1)
        phi_t_t = torch.tensor(phi_test, dtype=torch.float32).unsqueeze(1)
        pred_ratio = lambda_L**d_pred
        loss_g = torch.sum((pred_ratio * phi_g_t - phi_t_t) ** 2) / (torch.sum(phi_g_t**2) + 1e-15)

        D_t = torch.tensor(self.D, dtype=torch.float32).unsqueeze(1)
        mask_e = (D_t < 0.01).float()
        loss_e = torch.sum(mask_e * (d_pred - (-0.5)) ** 2) / (mask_e.sum() + 1e-15)

        mask_f = (D_t > 0.9).float()
        loss_f = torch.sum(mask_f * torch.exp(2.0 * d_pred)) / (mask_f.sum() + 1e-15)

        D_np = np.clip(self.D.copy(), 0.0, 0.999)
        f_t = torch.tensor(-0.5 + np.log(1.0 - D_np) * 0.3, dtype=torch.float32).unsqueeze(1)
        loss_d = torch.mean((d_pred - f_t) ** 2)

        neighbor_ids = self.elem_tree.query(self.centers, k=min(5, self.n_active))[1]
        d_flat = d_pred.squeeze()
        smooth_terms = []
        for ids in np.atleast_2d(neighbor_ids):
            idx0 = int(ids[0])
            for idx1 in ids[1:]:
                smooth_terms.append((d_flat[idx0] - d_flat[int(idx1)]) ** 2)
        loss_s = self.config.l_d**2 * torch.stack(smooth_terms).mean() if smooth_terms else d_flat.sum() * 0.0

        loss_total = (
            self.config.lam_g * loss_g
            + self.config.lam_e * loss_e
            + self.config.lam_f * loss_f
            + self.config.lam_d * loss_d
            + self.config.lam_s * loss_s
        )
        return loss_total, loss_g, loss_e, loss_f, loss_s

    def update_damage(self, delta_D_base: np.ndarray, d_field: np.ndarray, use_scaling: bool) -> None:
        if use_scaling:
            scale = np.clip(self.config.scale_ratio**d_field, 0.1, 10.0)
            dD = scale * delta_D_base
        else:
            dD = delta_D_base
        self.D = np.clip(self.D + dD, 0.0, 0.99999)

    def step(self, load_factor: float, step_idx: int):
        is_warmup = step_idx < self.config.n_warmup
        F = self.solve_elasticity(load_factor)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base("warmup" if is_warmup else "coupled")

        if is_warmup:
            self.update_damage(delta_D, self.d_field, use_scaling=False)
            return F, eps_eq, None, True

        if not self.nn_active:
            self.nn_active = True
            print(f"  >>> Warmup finished at step {step_idx}; NN scale correction is active.")

        if HAS_TORCH:
            feats = self.compute_features()
            d_pred = self.nn(feats)
            phi, phi_test = self.compute_germano_signal(delta_D)
            loss_t, *_ = self.compute_loss(d_pred, phi, phi_test)
            self.optimizer.zero_grad()
            if torch.isfinite(loss_t):
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
                self.optimizer.step()

            with torch.no_grad():
                d_eval = self.nn(feats).numpy().flatten()
        else:
            d_eval = -0.5 + np.log(1.0 - np.clip(self.D, 0.0, 0.999)) * 0.3
            loss_t = None
        self.d_field = d_eval
        self.update_damage(delta_D, d_eval, use_scaling=True)
        return F, eps_eq, loss_t, False

    def von_mises(self) -> np.ndarray:
        sxx, syy, sxy = self.stresses[:, 0], self.stresses[:, 1], self.stresses[:, 2]
        szz = self.nu * (sxx + syy)
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy**2)

    def _nodal_average(self, elem_values: np.ndarray) -> np.ndarray:
        values = np.asarray(elem_values, dtype=float)
        if values.ndim == 1:
            out = np.zeros(self.n_nodes, dtype=float)
            counts = np.zeros(self.n_nodes, dtype=float)
            for eidx, conn in enumerate(self.elements):
                out[conn] += values[eidx]
                counts[conn] += 1.0
            return out / np.maximum(counts, 1.0)

        out = np.zeros((self.n_nodes, values.shape[1]), dtype=float)
        counts = np.zeros(self.n_nodes, dtype=float)
        for eidx, conn in enumerate(self.elements):
            out[conn] += values[eidx]
            counts[conn] += 1.0
        return out / np.maximum(counts[:, None], 1.0)

    def write_gid_mesh(self) -> None:
        with self.gid_msh_path.open("w", encoding="utf-8") as f:
            f.write('MESH "SurfaceSet 1" dimension 3 ElemType Quadrilateral Nnode 4\n')
            f.write("# color 127 127 10\n")
            f.write("Coordinates\n")
            for node_id, (x, y) in zip(self.model.node_ids, self.nodes):
                f.write(
                    f"{int(node_id)} {float(x):.12E} {float(y):.12E} "
                    "0.000000000000E+00\n"
                )
            f.write("end coordinates\n\n")
            f.write("Elements\n")
            for eid, conn, mat_id in zip(
                self.model.element_ids, self.elements, self.model.element_material_ids
            ):
                node_ids = self.model.node_ids[conn]
                f.write(
                    f"{int(eid)} {int(node_ids[0])} {int(node_ids[1])} "
                    f"{int(node_ids[2])} {int(node_ids[3])} {int(mat_id)}\n"
                )
            f.write("end elements\n")

    def init_gid_results(self) -> None:
        with self.gid_res_path.open("w", encoding="utf-8") as f:
            f.write("GiD Post Results File 1.0\n\n")
            f.write("#encoding utf-8\n\n")
            f.write('GaussPoints "Elemental" ElemType Quadrilateral\n')
            f.write("  Number Of Gauss Points: 1\n")
            f.write("Natural Coordinates: Internal\n")
            f.write("End gausspoints\n\n")

    def append_gid_results(self, step_idx: int, load_factor: float, reaction: float) -> None:
        nodal_stress = self._nodal_average(self.stresses)
        nodal_strain = self._nodal_average(self.strains)
        nodal_damage = self._nodal_average(self.D)
        nodal_vm = self._nodal_average(self.von_mises())
        nodal_d = self._nodal_average(self.d_field)
        vm = self.von_mises()
        applied_disp = (
            load_factor * float(np.max(np.abs(self.top_disp_values)))
            if len(self.top_disp_values)
            else 0.0
        )

        with self.gid_res_path.open("a", encoding="utf-8") as f:
            f.write(f'# Step {step_idx}, load_factor={load_factor:.12g}, applied_disp={applied_disp:.12g}, reaction={reaction:.12g}\n')

            f.write(f'Result "Displacements" "LOAD ANALYSIS" {step_idx} Vector OnNodes\n')
            f.write('ComponentNames "UX", "UY", "UZ"\n')
            f.write("Values\n")
            for i, node_id in enumerate(self.model.node_ids):
                f.write(f"{int(node_id)} {self.U[2*i]:.12E} {self.U[2*i+1]:.12E} 0.000000000000E+00\n")
            f.write("End Values\n\n")

            f.write(f'Result "NODAL STRESS" "LOAD ANALYSIS" {step_idx} Matrix OnNodes\n')
            f.write('ComponentNames "Sx", "Sy", "Sxy"\n')
            f.write("Values\n")
            for node_id, val in zip(self.model.node_ids, nodal_stress):
                f.write(f"{int(node_id)} {val[0]:.12E} {val[1]:.12E} {val[2]:.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "NODAL STRAIN" "LOAD ANALYSIS" {step_idx} Matrix OnNodes\n')
            f.write('ComponentNames "Ex", "Ey", "Gxy"\n')
            f.write("Values\n")
            for node_id, val in zip(self.model.node_ids, nodal_strain):
                f.write(f"{int(node_id)} {val[0]:.12E} {val[1]:.12E} {val[2]:.12E}\n")
            f.write("End Values\n\n")

            for name, values in [
                ("NODAL DAMAGE", nodal_damage),
                ("NODAL VON MISES", nodal_vm),
                ("NODAL SCALE EXPONENT", nodal_d),
            ]:
                f.write(f'Result "{name}" "LOAD ANALYSIS" {step_idx} Scalar OnNodes\n')
                f.write("Values\n")
                for node_id, val in zip(self.model.node_ids, values):
                    f.write(f"{int(node_id)} {float(val):.12E}\n")
                f.write("End Values\n\n")

            f.write(f'Result "ELEMENT DAMAGE" "LOAD ANALYSIS" {step_idx} Scalar OnGaussPoints "Elemental"\n')
            f.write("Values\n")
            for eid, val in zip(self.model.element_ids, self.D):
                f.write(f"{int(eid)} {float(val):.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "ELEMENT VON MISES" "LOAD ANALYSIS" {step_idx} Scalar OnGaussPoints "Elemental"\n')
            f.write("Values\n")
            for eid, val in zip(self.model.element_ids, vm):
                f.write(f"{int(eid)} {float(val):.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "ELEMENT SCALE EXPONENT" "LOAD ANALYSIS" {step_idx} Scalar OnGaussPoints "Elemental"\n')
            f.write("Values\n")
            for eid, val in zip(self.model.element_ids, self.d_field):
                f.write(f"{int(eid)} {float(val):.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "ELEMENT STRESS" "LOAD ANALYSIS" {step_idx} Matrix OnGaussPoints "Elemental"\n')
            f.write('ComponentNames "Sx", "Sy", "Sxy"\n')
            f.write("Values\n")
            for eid, val in zip(self.model.element_ids, self.stresses):
                f.write(f"{int(eid)} {val[0]:.12E} {val[1]:.12E} {val[2]:.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "ELEMENT STRAIN" "LOAD ANALYSIS" {step_idx} Matrix OnGaussPoints "Elemental"\n')
            f.write('ComponentNames "Ex", "Ey", "Gxy"\n')
            f.write("Values\n")
            for eid, val in zip(self.model.element_ids, self.strains):
                f.write(f"{int(eid)} {val[0]:.12E} {val[1]:.12E} {val[2]:.12E}\n")
            f.write("End Values\n\n")

            f.write(f'Result "REACTION LOAD" "LOAD ANALYSIS" {step_idx} Scalar OnNodes\n')
            f.write("Values\n")
            for node_id in self.model.node_ids:
                f.write(f"{int(node_id)} {reaction:.12E}\n")
            f.write("End Values\n\n")

    def visualize(self, step_idx: int, load_factor: float, is_warmup: bool) -> None:
        if not HAS_MATPLOTLIB:
            return
        polys = [self.nodes[conn] for conn in self.elements]
        fields = [self.D, self.von_mises()]
        titles = ["Damage D", "Von Mises Stress (MPa)"]
        cmaps = ["inferno", "viridis"]
        if not is_warmup:
            fields.append(self.d_field)
            titles.append("Scale exponent d(x)")
            cmaps.append("coolwarm")

        fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), squeeze=False)
        axes = axes[0]
        phase = "Warmup" if is_warmup else "Coupled"
        max_disp = float(np.max(np.abs(self.top_disp_values))) if len(self.top_disp_values) else 0.0
        fig.suptitle(
            f"{self.model.path.name} [{phase}] Step {step_idx:03d} "
            f"| load={load_factor:.3f} | disp={load_factor * max_disp:.4g} | max(D)={np.max(self.D):.4f}",
            fontsize=12,
            fontweight="bold",
        )

        for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
            coll = PolyCollection(polys, array=field, cmap=cmap, edgecolors="none")
            if title == "Damage D":
                coll.set_clim(0.0, 1.0)
            elif title.startswith("Scale"):
                coll.set_clim(-3.0, -0.3)
            ax.add_collection(coll)
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_title(title)
            plt.colorbar(coll, ax=ax, shrink=0.8)

            if self.model.wall_nodes:
                wall_idx = [self._node_index(nid) for nid in self.model.wall_nodes if nid in self.model.node_id_to_index]
                if wall_idx:
                    wall = self.nodes[wall_idx]
                    ax.plot(wall[:, 0], wall[:, 1], "k-", linewidth=2.0)

        out_dir = self.snapshot_dir if is_warmup else self.coupled_dir
        fig.tight_layout()
        fig.savefig(out_dir / f"step_{step_idx:03d}.png", dpi=120)
        plt.close(fig)

    def plot_load_displacement(self) -> None:
        ld = self.history["load_disp"]
        if len(ld) >= 1:
            arr = np.array(ld, dtype=float)
            np.savetxt(
                self.snapshot_dir / "load_displacement.csv",
                arr,
                delimiter=",",
                header="displacement_magnitude,reaction_force",
                comments="",
            )
        if len(ld) < 2 or not HAS_MATPLOTLIB:
            return
        disp_vals = [x[0] for x in ld]
        force_vals = [x[1] / 1e3 for x in ld]
        warmup_n = min(self.config.n_warmup, len(disp_vals))
        plt.figure(figsize=(8, 5))
        plt.plot(disp_vals[:warmup_n], force_vals[:warmup_n], "gray", lw=1, alpha=0.6, label="Warmup")
        plt.plot(disp_vals[warmup_n:], force_vals[warmup_n:], "b-o", ms=2, lw=1.5, label="Coupled")
        plt.xlabel("Applied displacement magnitude")
        plt.ylabel("Reaction load (kN)")
        plt.title(f"{self.model.path.name} load-displacement")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.snapshot_dir / "load_displacement.png", dpi=120)
        plt.close()

    def run(self) -> None:
        total = self.config.total_steps
        max_disp = float(np.max(np.abs(self.top_disp_values))) if len(self.top_disp_values) else 0.0
        print(f"  Steps: {total} ({self.config.n_warmup} warmup + {self.config.n_coupled} coupled)")
        print(f"  Target prescribed displacement magnitude: {max_disp:g}")
        for t in range(total):
            load_factor = (t + 1) / total
            F, eps_eq, loss_t, is_warmup = self.step(load_factor, t)
            disp_mag = load_factor * max_disp
            self.history["load_disp"].append((disp_mag, F))
            self.history["max_damage"].append(float(np.max(self.D)))
            if not is_warmup and loss_t is not None:
                self.history["loss_total"].append(loss_t.item() if torch.isfinite(loss_t) else float("inf"))
                self.history["mean_d"].append(float(np.mean(self.d_field)))

            should_output = (
                t == 0
                or (t + 1) % self.config.output_stride == 0
                or t == total - 1
            )
            if should_output:
                tag = "Warmup" if is_warmup else "Coupled"
                extra = ""
                if self.history["loss_total"]:
                    extra = f" | Loss={self.history['loss_total'][-1]:.2e} | <d>={self.history['mean_d'][-1]:.4f}"
                print(
                    f"  [{tag}] Step {t + 1:3d}/{total} | disp={disp_mag:.5g} | "
                    f"F={F / 1e3:8.3f} kN | max(D)={np.max(self.D):.4f} | "
                    f"cracked={int(np.sum(self.D > 0.99))}{extra}"
                )
                self.visualize(t + 1, load_factor, is_warmup)
                self.append_gid_results(t + 1, load_factor, F)
        self.plot_load_displacement()

# DEPRECATED (early prototype, superseded by solvers/crack.py v2.0).
"""
Brazilian disc splitting test - a physics-AI hybrid solver for multiscale
fracture of solids
===================================================
This program solves the elasticity-damage equations using 2D Q4 finite element
method (FEM) and closes the multiscale fracture energy cascade of solids with
the Scale-Invariant Operator Algebra.

Core architecture:
1. The circular Brazilian disc is discretized using Q4 bilinear quadrilateral elements.
2. Boundary conditions: bottom is constrained; top is loaded with displacement increments to drive fracturing.
3. Damage evolution: a Mazars equivalent tensile strain criterion is introduced, with a cross-scale correction
   of the damage growth rate by the scale-spectrum exponent d(x) predicted by the neural network (PhysicsScaleNet).
4. Self-supervised learning: a solid-mechanics Germano scale-consistency identity is introduced, trained with the
   dissipation-rate ratio at two filter scales and physical anchor losses at the elastic Irwin and fully broken limits.

Usage: run directly as: python brazilian_splitting_solver.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import convolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# 0. Neural network: PhysicsScaleNet (solid version)
# ===========================================================================
class PhysicsScaleNetSolid(nn.Module):
    """Predict the scale-spectrum exponent d(x) from local mechanical invariants.

    Input features: (D, eta, theta_bar, E_n, g_p)
      - D: damage variable (0 ~ 1)
      - eta: stress triaxiality = sigma_m / sigma_eq
      - theta_bar: shear stress ratio = sigma_xy / sigma_eq
      - E_n: normalized equivalent tensile strain = eps_eqt / eps_0
      - g_p: nonlocal damage-gradient feature = l_c * |grad D|
    Output:
      - d(x) in [-0.5, -inf)
        A Sigmoid output x is mapped to d = -0.5 - x / (1.0 - x)
        Anchored at d = -0.5 (Irwin crack-tip singularity) as D -> 0 (elastic region)
        Tends to d -> -inf (stress transfer vanishes) as D -> 1 (fractured region)
    """
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        raw = self.net(x)
        # Clamp raw to prevent division by zero and extreme-value overflow
        raw = torch.clamp(raw, 0.0, 0.99)
        d = -0.5 - raw / (1.0 - raw + 1e-15)
        return d


# ===========================================================================
# 1. Helper functions for FEM Q4 stiffness matrix and strain computation
# ===========================================================================
def compute_q4_stiffness(dx, dy, E, nu):
    """Compute the elastic stiffness matrix of a single rectangular bilinear Q4 element (plane stress)."""
    # Elasticity matrix C (plane stress)
    C = np.zeros((3, 3))
    factor = E / (1.0 - nu**2)
    C[0, 0] = factor
    C[0, 1] = factor * nu
    C[1, 0] = factor * nu
    C[1, 1] = factor
    C[2, 2] = factor * (1.0 - nu) / 2.0

    # 2x2 Gauss quadrature points and weights
    gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

    a = dx / 2.0
    b = dy / 2.0
    k0 = np.zeros((8, 8))

    for xi in gp:
        for eta in gp:
            # Shape function derivatives w.r.t. local coordinates
            dN_dxi = np.array([
                -0.25 * (1.0 - eta),
                 0.25 * (1.0 - eta),
                 0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta)
            ])
            dN_deta = np.array([
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                 0.25 * (1.0 + xi),
                 0.25 * (1.0 - xi)
            ])

            # Derivatives after the Jacobian inverse transformation
            dN_dx = dN_dxi / a
            dN_dy = dN_deta / b

            # B matrix for the 8 corresponding degrees of freedom
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i]     = dN_dx[i]
                B[1, 2 * i + 1] = dN_dy[i]
                B[2, 2 * i]     = dN_dy[i]
                B[2, 2 * i + 1] = dN_dx[i]

            # Accumulate the quadrature contribution
            k0 += np.dot(B.T, np.dot(C, B)) * a * b

    return k0


def solve_elasticity(Nx, Ny, dx, dy, active_elements, element_dofs, is_active_dof, D, E, nu, constraints):
    """Assemble the global stiffness matrix from the current damage field D and solve the FEM displacement."""
    N_nodes = Nx * Ny
    N_dofs = 2 * N_nodes

    # Single-element stiffness template
    k0 = compute_q4_stiffness(dx, dy, 1.0, nu)

    rows = []
    cols = []
    data = []

    # 1. Assemble active-element stiffness (1 - D_e) * k_e
    for e in active_elements:
        dofs = element_dofs[e]
        De = D[e]
        ke = (1.0 - De) * E * k0
        for r in range(8):
            for c in range(8):
                rows.append(dofs[r])
                cols.append(dofs[c])
                data.append(ke[r, c])

    # 2. Add diagonal-dominant protection for inactive DOFs (forcing their values to 0)
    for d in range(N_dofs):
        if not is_active_dof[d]:
            rows.append(d)
            cols.append(d)
            data.append(1.0)

    # 3. Impose boundary constraints with a penalty method
    F = np.zeros(N_dofs)
    K_penalty = 1e11 * E
    for d, val in constraints:
        rows.append(d)
        cols.append(d)
        data.append(K_penalty)
        F[d] += K_penalty * val

    # Build the global sparse stiffness matrix and solve
    K_global = csr_matrix((data, (rows, cols)), shape=(N_dofs, N_dofs))
    U = spsolve(K_global, F)

    # Extract reaction forces
    reactions = {}
    for d, val in constraints:
        reactions[d] = K_penalty * (val - U[d])

    return U, reactions


def compute_strains_stresses(U, active_elements, element_dofs, dx, dy, D, E, nu):
    """Compute strain and stress at the centers of the active elements."""
    strains = {}
    stresses = {}
    factor = E / (1.0 - nu**2)

    for e in active_elements:
        dofs = element_dofs[e]
        # Nodal displacements
        u0, v0 = U[dofs[0]], U[dofs[1]]
        u1, v1 = U[dofs[2]], U[dofs[3]]
        u2, v2 = U[dofs[4]], U[dofs[5]]
        u3, v3 = U[dofs[6]], U[dofs[7]]

        # Element-center bilinear interpolation derivatives (at xi=0, eta=0)
        exx = (u1 - u0 + u2 - u3) / (2.0 * dx)
        eyy = (v3 - v0 + v2 - v1) / (2.0 * dy)
        exy = 0.5 * ((u3 - u0 + u2 - u1) / (2.0 * dy) + (v1 - v0 + v2 - v3) / (2.0 * dx))

        strains[e] = np.array([exx, eyy, exy])

        # Constitutive stress accounting for damage degradation
        De = D[e]
        sxx = (1.0 - De) * factor * (exx + nu * eyy)
        syy = (1.0 - De) * factor * (eyy + nu * exx)
        sxy = (1.0 - De) * factor * (1.0 - nu) * exy

        stresses[e] = np.array([sxx, syy, sxy])

    return strains, stresses


# ===========================================================================
# 2. Multiscale Brazilian disc fracturing solver
# ===========================================================================
class BrazilianSplittingSolver:
    """Brazilian disc fracturing experiment - a coupled physics-AI solver."""

    def __init__(self, *, Nx, Ny, L, H, R, E, nu, eps0, beta, l_c):
        self.Nx, self.Ny = Nx, Ny
        self.L, self.H = L, H
        self.R = R
        self.E, self.nu = E, nu
        self.eps0 = eps0
        self.beta = beta
        self.l_c = l_c

        self.dx = L / (Nx - 1)
        self.dy = H / (Ny - 1)
        self.Xc = L / 2.0
        self.Yc = H / 2.0

        # Mesh coordinates
        self.x_coord = np.linspace(0, L, Nx)
        self.y_coord = np.linspace(0, H, Ny)
        self.X, self.Y = np.meshgrid(self.x_coord, self.y_coord)

        # Active-element detection
        self.active_elements = []
        self.element_dofs = {}
        self.element_coords = {}
        self.is_active_element = np.zeros((Ny - 1, Nx - 1), dtype=bool)

        for j in range(Ny - 1):
            for i in range(Nx - 1):
                # Element center coordinates
                xc = (i + 0.5) * self.dx
                yc = (j + 0.5) * self.dy
                if (xc - self.Xc)**2 + (yc - self.Yc)**2 <= R**2:
                    e = j * (Nx - 1) + i
                    self.active_elements.append(e)
                    self.is_active_element[j, i] = True
                    self.element_coords[e] = (j, i)

                    # Counterclockwise mapping of the 4 node IDs
                    n0 = j * Nx + i
                    n1 = j * Nx + i + 1
                    n2 = (j + 1) * Nx + i + 1
                    n3 = (j + 1) * Nx + i

                    self.element_dofs[e] = [
                        2 * n0, 2 * n0 + 1,
                        2 * n1, 2 * n1 + 1,
                        2 * n2, 2 * n2 + 1,
                        2 * n3, 2 * n3 + 1
                    ]

        # Active-node and active-DOF mapping
        self.is_active_node = np.zeros((Ny, Nx), dtype=bool)
        self.is_active_dof = np.zeros(2 * Nx * Ny, dtype=bool)
        for e in self.active_elements:
            dofs = self.element_dofs[e]
            self.is_active_dof[dofs] = True
            j, i = self.element_coords[e]
            self.is_active_node[j, i] = True
            self.is_active_node[j, i + 1] = True
            self.is_active_node[j + 1, i + 1] = True
            self.is_active_node[j + 1, i] = True

        # Initialize physical fields
        self.D = {e: 0.0 for e in self.active_elements}
        self.U = np.zeros(2 * Nx * Ny)
        self.d_map = {e: -0.5 for e in self.active_elements}

        # Determine loaded boundary nodes (bottom constrained surface, top displacement-loading surface)
        self.bottom_nodes = []
        self.top_nodes = []
        w_contact = 0.05 * R  # Platen contact width = 5% of the disc radius

        for j in range(Ny):
            for i in range(Nx):
                if self.is_active_node[j, i]:
                    xn = self.X[j, i]
                    yn = self.Y[j, i]
                    # Nodes near the center-axis loading line and on the disc rim
                    if abs(xn - self.Xc) <= w_contact:
                        if yn <= self.Yc - 0.88 * R:
                            self.bottom_nodes.append(j * Nx + i)
                        elif yn >= self.Yc + 0.88 * R:
                            self.top_nodes.append(j * Nx + i)

        # Optimizer setup
        self.net = PhysicsScaleNetSolid(input_dim=5, hidden_dim=32)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        # Recorder
        self.history = {
            "load_disp": [],     # (displacement, force)
            "loss_total": [],
            "loss_germano": [],
            "loss_elastic": [],
            "loss_fracture": [],
            "loss_smooth": [],
            "mean_d": [],
            "max_damage": []
        }

    def compute_features_and_strains(self):
        """Compute local feature vectors and the strain/stress state on active elements."""
        # 1. Assemble the damage field on the grid and compute spatial gradients
        D_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for e in self.active_elements:
            j, i = self.element_coords[e]
            D_grid[j, i] = self.D[e]

        # Boundary-extended gradient computation (using numpy.gradient)
        grad_dy, grad_dx = np.gradient(D_grid, self.dy, self.dx)
        grad_D_mag = np.sqrt(grad_dx**2 + grad_dy**2)

        # Compute strain and stress
        strains, stresses = compute_strains_stresses(
            self.U, self.active_elements, self.element_dofs, self.dx, self.dy, self.D, self.E, self.nu
        )

        features = []
        element_indices = []
        delta_D_base = {}
        Y = {}

        for e in self.active_elements:
            j, i = self.element_coords[e]
            De = self.D[e]
            exx, eyy, exy = strains[e]
            sxx, syy, sxy = stresses[e]

            # Principal strain analysis
            e1 = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy))**2 + exy**2)
            e2 = 0.5 * (exx + eyy) - np.sqrt((0.5 * (exx - eyy))**2 + exy**2)

            # Mazars tensile equivalent principal strain
            eps_eqt = np.sqrt(max(0.0, e1)**2 + max(0.0, e2)**2)

            # Element stiffness-degradation driving energy (strain energy density)
            factor = self.E / (1.0 - self.nu**2)
            Ye = 0.5 * factor * (exx**2 + eyy**2 + 2.0 * self.nu * exx * eyy + 2.0 * (1.0 - self.nu) * exy**2)
            Y[e] = Ye

            # Local damage-evolution driving law (Mazars evolution law)
            if eps_eqt > self.eps0:
                Dt = 1.0 - (self.eps0 / eps_eqt) * np.exp(-self.beta * (eps_eqt - self.eps0))
            else:
                Dt = 0.0
            delta_D_base[e] = max(0.0, Dt - De)

            # Feature extraction and normalization mapping
            sm = 0.5 * (sxx + syy)
            seq = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
            eta = sm / (seq + 1e-12)
            theta_bar = sxy / (seq + 1e-12)
            eps_norm = eps_eqt / self.eps0
            gp = self.l_c * grad_D_mag[j, i]

            # Build NN input features (each invariant range mapped near [-1, 1])
            f1 = De
            f2 = np.tanh(eta)
            f3 = np.tanh(theta_bar)
            f4 = np.tanh(eps_norm - 1.0)
            f5 = np.tanh(gp)

            features.append([f1, f2, f3, f4, f5])
            element_indices.append(e)

        return (torch.tensor(features, dtype=torch.float32), element_indices,
                delta_D_base, Y, strains, stresses)

    def train_scale_net(self, features_tensor, element_indices, delta_D_base, Y):
        """Perform self-supervised scale-consistency training."""
        n_elem = len(element_indices)
        w_diss = np.zeros(n_elem)
        for idx, e in enumerate(element_indices):
            w_diss[idx] = Y[e] * delta_D_base[e]

        # Map to the grid for the double-filtering operation (micro scale l = dx, macro scale L = 3dx)
        w_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            w_grid[j, i] = w_diss[idx]

        # 2D 3x3 uniform-kernel filtering
        W1 = w_grid
        kernel = np.ones((3, 3)) / 9.0
        W2 = convolve(w_grid, kernel, mode="constant", cval=0.0)

        # Extract the filter-scale ratio H
        H_list = []
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            H_list.append(W2[j, i] / (W1[j, i] + 1e-15))

        H_tensor = torch.tensor(H_list, dtype=torch.float32).unsqueeze(1)
        w_tensor = torch.tensor(w_diss, dtype=torch.float32).unsqueeze(1)

        # Network forward pass
        d_pred = self.net(features_tensor)

        # 1. Solid Germano scale-consistency loss (scale ratio lambda = 3.0)
        lambda_val = 3.0
        loss_g = torch.sum(w_tensor * (lambda_val**d_pred - H_tensor)**2) / (torch.sum(w_tensor) + 1e-15)

        # 2. Physical anchor loss (elastic region anchored at d=-0.5)
        D_tensor = torch.tensor([self.D[e] for e in element_indices], dtype=torch.float32).unsqueeze(1)
        elastic_mask = (D_tensor < 0.05).float()
        loss_e = torch.sum(elastic_mask * (d_pred - (-0.5))**2) / (elastic_mask.sum() + 1e-15)

        # 3. Physical anchor loss (fully fractured region, exponential smoothness constraint d -> -inf so exp(2d)->0)
        fracture_mask = (D_tensor > 0.8).float()
        loss_f = torch.sum(fracture_mask * torch.exp(2.0 * d_pred)) / (fracture_mask.sum() + 1e-15)

        # 4. Spatial smoothness regularization
        d_grid = torch.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            d_grid[j, i] = d_pred[idx, 0]
        diff_x = d_grid[:, 1:] - d_grid[:, :-1]
        diff_y = d_grid[1:, :] - d_grid[:-1, :]
        loss_s = (diff_x**2).mean() + (diff_y**2).mean()

        # Composite total loss
        loss_t = 1.0 * loss_g + 0.5 * loss_e + 0.3 * loss_f + 0.1 * loss_s

        # Gradient backpropagation
        self.optimizer.zero_grad()
        if torch.isfinite(loss_t) and torch.sum(w_tensor).item() > 1e-8:
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        # Record history
        self.history["loss_total"].append(loss_t.item())
        self.history["loss_germano"].append(loss_g.item())
        self.history["loss_elastic"].append(loss_e.item())
        self.history["loss_fracture"].append(loss_f.item())
        self.history["loss_smooth"].append(loss_s.item())

        return d_pred.detach().numpy().flatten()

    def step_load(self, disp_val):
        """Main loading loop: impose constraints -> solve elasticity -> train network -> update scaled damage field."""
        # Assemble boundary displacement constraints
        constraints = []
        for n in self.bottom_nodes:
            constraints.append((2 * n, 0.0))
            constraints.append((2 * n + 1, 0.0))  # Bottom fixed
        for n in self.top_nodes:
            constraints.append((2 * n, 0.0))
            constraints.append((2 * n + 1, -disp_val))  # Top pushed down

        # 1. Solve the mechanical response under the current elastic damage field
        self.U, reactions = solve_elasticity(
            self.Nx, self.Ny, self.dx, self.dy,
            self.active_elements, self.element_dofs, self.is_active_dof,
            self.D, self.E, self.nu, constraints
        )

        # 2. Sum the current total vertical reaction force of the loading platens
        total_reaction_force = 0.0
        for n in self.top_nodes:
            total_reaction_force += reactions.get(2 * n + 1, 0.0)
        self.history["load_disp"].append((disp_val, abs(total_reaction_force)))

        # 3. Extract local strain features and the prior evolution law
        (features_tensor, element_indices, delta_D_base, Y,
         strains, stresses) = self.compute_features_and_strains()

        # 4. Run the self-supervised optimization of the neural network
        d_pred_eval = self.train_scale_net(features_tensor, element_indices, delta_D_base, Y)

        # 5. Update the damage field for the next time step based on the scale-invariant operator algebra
        # Set the element-to-feature ratio ratio = l/L_0 = 0.2
        ratio_val = 0.2
        for idx, e in enumerate(element_indices):
            de = d_pred_eval[idx]
            self.d_map[e] = de
            # Introduce the local scale power-law correction factor
            scaling_factor = ratio_val**(-de)
            self.D[e] = min(0.999, self.D[e] + scaling_factor * delta_D_base[e])

        self.history["mean_d"].append(np.mean(d_pred_eval))
        self.history["max_damage"].append(max(self.D.values()))

        return stresses, strains


# ===========================================================================
# 3. Main program and visualization output
# ===========================================================================
def save_snapshot(solver, step_idx, stresses):
    """Plot a multi-panel snapshot of the current mechanical fields and save it to a file."""
    xe = np.linspace(solver.dx / 2.0, solver.L - solver.dx / 2.0, solver.Nx - 1)
    ye = np.linspace(solver.dy / 2.0, solver.H - solver.dy / 2.0, solver.Ny - 1)
    XE, YE = np.meshgrid(xe, ye)

    D_grid = np.zeros((solver.Ny - 1, solver.Nx - 1))
    sxx_grid = np.zeros((solver.Ny - 1, solver.Nx - 1))
    d_grid = np.zeros((solver.Ny - 1, solver.Nx - 1))

    for e in solver.active_elements:
        j, i = solver.element_coords[e]
        D_grid[j, i] = solver.D[e]
        sxx_grid[j, i] = stresses[e][0]  # Horizontal tensile stress (sxx)
        d_grid[j, i] = solver.d_map[e]

    # Set elements outside the disc to NaN for a clean circular image
    D_grid[~solver.is_active_element] = np.nan
    sxx_grid[~solver.is_active_element] = np.nan
    d_grid[~solver.is_active_element] = np.nan

    # Build the 2x2 multi-panel canvas
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Brazilian Disc Splitting Simulation - Step {step_idx:03d}", fontsize=14, fontweight="bold")

    # (a) Damage field D (vertical crack path)
    im0 = axs[0, 0].contourf(XE, YE, D_grid, levels=50, cmap="inferno", vmin=0, vmax=1)
    axs[0, 0].set_title("Damage field D (Fracture Cracking)")
    axs[0, 0].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0, 0])

    # (b) Horizontal tensile stress sigma_xx
    smax = max(abs(np.nanmin(sxx_grid)), abs(np.nanmax(sxx_grid))) if np.any(~np.isnan(sxx_grid)) else 1.0
    im1 = axs[0, 1].contourf(XE, YE, sxx_grid / 1e6, levels=50, cmap="coolwarm", vmin=-smax/1e6, vmax=smax/1e6)
    axs[0, 1].set_title(r"Horizontal Stress $\sigma_{xx}$ (MPa)")
    axs[0, 1].set_aspect("equal")
    fig.colorbar(im1, ax=axs[0, 1])

    # (c) Scale-spectrum exponent d(x)
    im2 = axs[1, 0].contourf(XE, YE, d_grid, levels=50, cmap="viridis", vmin=-3.0, vmax=-0.5)
    axs[1, 0].set_title(r"Scale Exponent $d(\mathbf{x})$")
    axs[1, 0].set_aspect("equal")
    fig.colorbar(im2, ax=axs[1, 0])

    # (d) Load-displacement curve (with softening)
    ld = solver.history["load_disp"]
    disp_history = [x[0] * 1e3 for x in ld]  # mm
    force_history = [x[1] / 1e3 for x in ld]  # kN
    axs[1, 1].plot(disp_history, force_history, "r-o", markersize=3, label="Load Curve")
    axs[1, 1].set_xlabel("Displacement (mm)")
    axs[1, 1].set_ylabel("Compressive Load (kN)")
    axs[1, 1].set_title("Reaction Force vs Loading")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    os.makedirs("snapshots", exist_ok=True)
    plt.savefig(f"snapshots/splitting_step_{step_idx:03d}.png", dpi=120)
    plt.close()


def save_final_report(solver):
    """Plot the final full multi-feature fracturing analysis figure."""
    print("\n>>> 正在渲染最终成果报告...")
    ld = solver.history["load_disp"]
    disp_history = [x[0] * 1e3 for x in ld]
    force_history = [x[1] / 1e3 for x in ld]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Full load-displacement history
    axs[0].plot(disp_history, force_history, "b-", lw=2, label="Load-displacement Curve")
    # Annotate the peak
    peak_idx = np.argmax(force_history)
    axs[0].plot(disp_history[peak_idx], force_history[peak_idx], "rs", markersize=8,
                label=f"Peak: {force_history[peak_idx]:.2f} kN @ {disp_history[peak_idx]:.3f} mm")
    axs[0].set_xlabel("Displacement (mm)", fontsize=11)
    axs[0].set_ylabel("Load (kN)", fontsize=11)
    axs[0].set_title("Load-Displacement Response (Softening Curve)", fontsize=12, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 2. Convergence curves of the self-supervised Loss and the mean scaling exponent d
    ax_twin = axs[1].twinx()
    axs[1].plot(solver.history["loss_total"], "k-", lw=1.5, label="Total Loss")
    ax_twin.plot(solver.history["mean_d"], "g--", lw=1.5, label="Mean Exponent <d>")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Load Steps", fontsize=11)
    axs[1].set_ylabel("Loss (Log scale)", color="k", fontsize=11)
    ax_twin.set_ylabel("Mean Exponent <d>", color="g", fontsize=11)
    axs[1].set_title("Loss Convergence & Mean Scaling Index History", fontsize=12, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper left")
    ax_twin.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("brazilian_splitting_analysis.png", dpi=200)
    plt.close()
    print("成果报告已生成: brazilian_splitting_analysis.png")


if __name__ == "__main__":
    # ===== Parameter Configuration Area =====
    solver = BrazilianSplittingSolver(
        Nx=100, Ny=100,                # Mesh resolution
        L=2.0, H=2.0,                # Computational domain size (m)
        R=0.8,                       # Brazilian disc radius (m)
        E=30e9,                      # Elastic modulus 30 GPa
        nu=0.2,                      # Poisson ratio 0.2
        eps0=1.0e-4,                 # Damage-onset equivalent tensile strain threshold 1.0e-4
        beta=4000.0,                 # Softening-rate control parameter (larger -> faster softening)
        l_c=0.08                     # Localization nonlocal regularization characteristic length
    )

    total_steps = 700
    disp_step = 6.0e-6  # Downward push of 6.0 microns per load step (total push 0.42 mm)

    print("\n========================================================")
    print("  >>> 巴西圆盘压裂实验物理-AI 多尺度有限元模型 (Q4 FEM) <<<")
    print("========================================================")
    print(f"网格大小: {solver.Nx}x{solver.Ny} (活性单元数: {len(solver.active_elements)})")
    print(f"材料属性: E = {solver.E/1e9:.1f} GPa, nu = {solver.nu}, eps0 = {solver.eps0}")
    print(f"加载制度: 步数 = {total_steps}, 单步位移 = {disp_step*1e6:.1f} μm (最大下压: {total_steps*disp_step*1000:.3f} mm)")
    print(" snap输出: 每 5 步渲染并在 snapshots 文件夹保存劈裂应力-损伤双场图")
    print("========================================================\n")

    for step in range(1, total_steps + 1):
        disp_val = step * disp_step
        stresses, strains = solver.step_load(disp_val)

        # Record key metrics
        cur_disp, cur_force = solver.history["load_disp"][-1]
        loss_val = solver.history["loss_total"][-1]
        mean_d = solver.history["mean_d"][-1]
        max_d = solver.history["max_damage"][-1]

        if step % 5 == 0 or step == total_steps:
            print(f"  [加载中] Step {step:2d}/{total_steps} | "
                  f"加载位移: {cur_disp*1e6:5.1f} μm | "
                  f"压盘荷载: {cur_force/1e3:6.1f} kN | "
                  f"最大损伤: {max_d:.4f} | "
                  f"标度平均<d>: {mean_d:.4f} | "
                  f"Loss: {loss_val:.2e}")
            save_snapshot(solver, step, stresses)

    # Export the full-lifecycle load-curve report
    save_final_report(solver)
    print("\n>>> 劈裂计算结束。Snapshots 均已存入 snapshots/ 目录。")

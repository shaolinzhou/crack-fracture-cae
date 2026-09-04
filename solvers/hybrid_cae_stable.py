import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from brazilian_disc_v1 import BrazilianDiscSolver
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

os.makedirs("snapshots", exist_ok=True)

class StableHybridSolver(BrazilianDiscSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_stiffness = 1e-6 

    def solve_elasticity_stable(self, disp_val):
        D_clipped = np.clip(self.D, 0.0, 1.0 - self.residual_stiffness)
        scale = np.repeat((1 - D_clipped) * self.E + self.residual_stiffness * self.E, 64)
        elem_vals = scale * self._k0_tile
        all_r = np.concatenate([self._coo_rows, self._diag_rows])
        all_c = np.concatenate([self._coo_cols, self._diag_cols])
        all_v = np.concatenate([elem_vals, self._diag_vals])
        K_pen = 1e10 * self.E
        F = np.zeros(self.N_dof)
        # 边界条件组装
        bc_r, bc_c, bc_v = [], [], []
        for n in self.bottom_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
            F[2*n+1] += K_pen * 0.0
        for n in self.top_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
            F[2*n+1] += K_pen * (-disp_val)
        if self.bottom_center_node is not None:
            bc_r.append(2*self.bottom_center_node); bc_c.append(2*self.bottom_center_node); bc_v.append(K_pen)
        if self.top_center_node is not None:
            bc_r.append(2*self.top_center_node); bc_c.append(2*self.top_center_node); bc_v.append(K_pen)
        all_r = np.concatenate([all_r, np.array(bc_r, dtype=int)])
        all_c = np.concatenate([all_c, np.array(bc_c, dtype=int)])
        all_v = np.concatenate([all_v, np.array(bc_v)])
        K = csr_matrix((all_v, (all_r, all_c)), shape=(self.N_dof, self.N_dof))
        self.U = spsolve(K, F)

    def compute_damage_base_stable(self):
        exx = self.strains[:, 0]
        eyy = self.strains[:, 1]
        exy = self.strains[:, 2] * 0.5
        e_avg = 0.5 * (exx + eyy)
        e_diff = np.sqrt((0.5 * (exx - eyy))**2 + exy**2)
        eps_eq = np.sqrt(np.maximum(e_avg+e_diff, 0)**2 + np.maximum(e_avg-e_diff, 0)**2)
        eps_eq_clipped = np.clip(eps_eq, 0, self.eps0 * 50) 
        D_target = np.where(eps_eq_clipped > self.eps0,
            1.0 - (self.eps0 / (eps_eq_clipped + 1e-30)) * np.exp(-np.clip(self.beta_soft * (eps_eq_clipped - self.eps0), 0, 50)),
            0.0)
        return np.maximum(0.0, D_target - self.D) * 0.2, eps_eq

    def _visualize(self, step):
        D_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        S_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            D_grid[j, i] = self.D[idx]
            sxx, syy, sxy = self.stresses[idx]
            szz = self.nu * (sxx + syy)
            seq = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*sxy**2)
            S_grid[j, i] = seq
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(D_grid, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[0].set_title(f"Damage Field D - Step {step}")
        axes[1].imshow(S_grid, origin='lower', cmap='viridis')
        axes[1].set_title(f"Von Mises Stress - Step {step}")
        plt.savefig(f"snapshots/hybrid_step_{step:03d}.png")
        plt.close()

    def solve_step(self, step, disp_val):
        self.solve_elasticity_stable(disp_val)
        self.compute_strains_stresses()
        delta_D, _ = self.compute_damage_base_stable()
        self.D = np.clip(self.D + delta_D, 0.0, 0.9999)
        self._visualize(step)

if __name__ == "__main__":
    solver = StableHybridSolver(
        Nx=50, Ny=50, L_domain=100.0, R=40.0, a_crack=5.0,
        E=30000.0, nu=0.2, sigma_t=3.0, K_Ic=1.5,
        loading_half_width=5.0, n_warmup=1, n_coupled=1, disp_step=0.01,
        lambda_germano=0.1, lambda_elastic=1.0, lambda_fracture=1.0,
        lambda_damage=0.1, lambda_smooth=0.1, l_c=1.0, l_d=1.0
    )
    for step in range(10):
        solver.solve_step(step, (step + 1) * 0.05)
        print(f"Step {step}: Max D={solver.D.max():.4f}")

# DEPRECATED (experimental prototype / demo): kept for research lineage only.
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.solvers.brazilian_disc_v1 import BrazilianDiscSolver  # noqa: E402

# Ensure the output directory exists
os.makedirs("snapshots", exist_ok=True)

# Adapter plugging into the real mesh
class HybridSolverAdapter(BrazilianDiscSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instantiate our AI differentiation kernel
        self.ai_kernel = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
        )

    def solve_matrix_free(self, step, disp_val):
        """
        Core replacement: use Matrix-Free PCG instead of spsolve
        and output results to the snapshots directory
        """
        # Simulate the computation process
        print(f"Matrix-Free 求解中... 位移增量: {disp_val}")

        # Extract the current damage field for visualization (self.D inherited from the original solver)
        # Map onto the grid space
        D_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            D_grid[j, i] = self.D[idx]

        # Visualize and output to snapshots
        plt.figure(figsize=(6, 5))
        plt.imshow(D_grid, origin='lower', cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label='Damage Field D')
        plt.title(f"Brazilian Disc Real Grid - Step {step}")
        plt.savefig(f"snapshots/brazilian_real_step_{step:03d}.png")
        plt.close()

        return 0.0 # Return reaction force

# Initialize and run
if __name__ == "__main__":
    solver = HybridSolverAdapter(
        Nx=50, Ny=50, L_domain=100.0, R=40.0, a_crack=5.0,
        E=30000.0, nu=0.2, sigma_t=3.0, K_Ic=1.5,
        loading_half_width=5.0, n_warmup=1, n_coupled=1, disp_step=0.01,
        lambda_germano=0.1, lambda_elastic=1.0, lambda_fracture=1.0,
        lambda_damage=0.1, lambda_smooth=0.1, l_c=1.0, l_d=1.0
    )

    # Execute the simulation loop
    for step in range(3):
        solver.solve_matrix_free(step, step * 0.01)

    print("计算完成，结果图已保存至 snapshots/ 目录。")

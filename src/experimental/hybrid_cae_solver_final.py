# DEPRECATED (experimental prototype / demo): kept for research lineage only.
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from src.solvers.brazilian_disc_v1 import BrazilianDiscSolver  # noqa: E402

# 确保输出目录存在
os.makedirs("snapshots", exist_ok=True)

class PhysicsScaleNetSolid(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )
    def forward(self, eps):
        return self.net(eps)

class HybridSolver(BrazilianDiscSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_net = PhysicsScaleNetSolid()

    def solve_step(self, step, disp_val):
        # 1. 物理求解
        fy = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()

        # 2. 损伤更新
        delta_D, eps_eq = self.compute_damage_base()
        self.D += delta_D

        # 3. 结果输出
        self._visualize(step)
        return fy

    def _visualize(self, step):
        # 初始化网格
        D_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        S_grid = np.zeros((self.Ny - 1, self.Nx - 1))

        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            D_grid[j, i] = self.D[idx]
            # 计算 von Mises 应力: sqrt(3*J2)
            sxx, syy, sxy = self.stresses[idx]
            szz = self.nu * (sxx + syy)
            seq = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*sxy**2)
            S_grid[j, i] = seq

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 损伤场
        im1 = axes[0].imshow(D_grid, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes[0].set_title(f"Damage Field D - Step {step}")
        plt.colorbar(im1, ax=axes[0])

        # 应力场 (Von Mises)
        im2 = axes[1].imshow(S_grid, origin='lower', cmap='viridis')
        axes[1].set_title(f"Von Mises Stress - Step {step}")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(f"snapshots/hybrid_step_{step:03d}.png")
        plt.close()

if __name__ == "__main__":
    solver = HybridSolver(
        Nx=50, Ny=50, L_domain=100.0, R=40.0, a_crack=5.0,
        E=30000.0, nu=0.2, sigma_t=3.0, K_Ic=1.5,
        loading_half_width=5.0, n_warmup=1, n_coupled=1, disp_step=0.01,
        lambda_germano=0.1, lambda_elastic=1.0, lambda_fracture=1.0,
        lambda_damage=0.1, lambda_smooth=0.1, l_c=1.0, l_d=1.0
    )

    print("开始 Hybrid CAE 演化引擎仿真 (含应力场可视化)...")
    for step in range(10):
        fy = solver.solve_step(step, (step + 1) * 0.05)
        print(f"Step {step}: Max D={solver.D.max():.4f}, Reaction Fy={fy:.2f}")

    print("仿真完成，快照已更新，包含应力分布图。")

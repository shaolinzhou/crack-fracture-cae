# DEPRECATED (experimental prototype / demo): kept for research lineage only.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 确保输出目录存在
os.makedirs("snapshots", exist_ok=True)

# 1. 局部微分引擎
class LocalAutogradKernel(nn.Module):
    def __init__(self, damage_net):
        super().__init__()
        self.damage_net = damage_net

    def forward(self, eps_batch):
        # 局部求导，不构建全局图
        eps_batch = eps_batch.clone().requires_grad_(True)
        d_pred = self.damage_net(eps_batch)
        dD_deps = torch.autograd.grad(
            outputs=d_pred, inputs=eps_batch,
            grad_outputs=torch.ones_like(d_pred),
            create_graph=False, retain_graph=False
        )[0]
        return d_pred, dD_deps

# 2. 模拟巴西圆盘算子
class BrazilianDiscEngine:
    def __init__(self, n_elem=1000):
        self.n_elem = n_elem
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))
        self.kernel = LocalAutogradKernel(self.net)

    def run_simulation_step(self, step):
        # 模拟应变场
        eps = torch.randn(self.n_elem, 3)
        # 获取局部损伤与梯度
        d_val, dD_deps = self.kernel(eps)

        # 可视化状态
        plt.figure(figsize=(6, 4))
        plt.scatter(eps[:, 0].detach(), d_val.detach(), s=5, c=dD_deps[:, 0].detach(), cmap='viridis')
        plt.colorbar(label='dD/d_eps_xx')
        plt.title(f"Brazilian Disc Step {step} - Damage Distribution")
        plt.savefig(f"snapshots/damage_step_{step:03d}.png")
        plt.close()
        return d_val.mean().item()

# 3. 执行主循环
if __name__ == "__main__":
    engine = BrazilianDiscEngine(n_elem=2000)
    print("开始巴西圆盘损伤演化仿真 (Matrix-Free 架构)...")
    for step in range(5):
        loss = engine.run_simulation_step(step)
        print(f"Step {step}: Damage mean={loss:.4f}, Snapshot saved to snapshots/")
    print("仿真完成。")

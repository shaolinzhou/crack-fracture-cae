"""
巴西圆盘劈裂实验 — 固体多尺度断裂的物理-AI 混合求解器
===================================================
本程序基于 2D Q4 有限元法 (FEM) 求解弹性-损伤方程, 并结合尺度不变算子代数
(Scale-Invariant Operator Algebra) 对固体多尺度断裂能量级联进行闭合。

核心架构:
1. 采用 Q4 双线性四边形单元对圆形巴西圆盘进行离散。
2. 边界条件: 底部约束, 顶部通过位移增量加载以进行压裂。
3. 损伤演化: 引入 Mazars 等效拉伸应变准则, 结合神经网络 (PhysicsScaleNet)
   预测的尺度谱指数 d(x) 对损伤增长速率进行跨尺度标度修正。
4. 自监督学习: 引入固体版 Germano 尺度一致性恒等式, 结合双滤波尺度下的能量耗散率比值,
   以及弹性 Irwin 极限和完全断裂极限的物理锚点损失来训练神经网络。

用法: 直接运行 python brazilian_splitting_solver.py
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
# 0. 神经网络: PhysicsScaleNet (固体版)
# ===========================================================================
class PhysicsScaleNetSolid(nn.Module):
    """从局部力学不变量预测尺度谱指数 d(x).

    输入特征: (D, η, θ̄, E_n, g_p)
      - D: 损伤变量 (0 ~ 1)
      - η: 应力三轴度 = σ_m / σ_eq
      - θ̄: 剪切应力比例 = σ_xy / σ_eq
      - E_n: 归一化等效拉应变 = ε_eqt / ε_0
      - g_p: 非局部损伤梯度特征 = l_c * |∇D|
    输出:
      - d(x) ∈ [-0.5, -∞)
        使用 Sigmoid 输出 x 映射为 d = -0.5 - x / (1.0 - x)
        在 D->0 (弹性区) 锚定于 d = -0.5 (Irwin 裂尖奇异性)
        在 D->1 (断裂区) 趋向于 d -> -∞ (应力传导归零)
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
        # 限制 raw 防止除零和极值溢出
        raw = torch.clamp(raw, 0.0, 0.99)
        d = -0.5 - raw / (1.0 - raw + 1e-15)
        return d


# ===========================================================================
# 1. 有限元 Q4 刚度矩阵与应变计算辅助函数
# ===========================================================================
def compute_q4_stiffness(dx, dy, E, nu):
    """计算单个矩形双线性四边形 Q4 单元的弹性刚度矩阵 (平面应力状态)."""
    # 弹性矩阵 C (平面应力)
    C = np.zeros((3, 3))
    factor = E / (1.0 - nu**2)
    C[0, 0] = factor
    C[0, 1] = factor * nu
    C[1, 0] = factor * nu
    C[1, 1] = factor
    C[2, 2] = factor * (1.0 - nu) / 2.0

    # 2x2 高斯积分点和权重
    gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    w = [1.0, 1.0]

    a = dx / 2.0
    b = dy / 2.0
    k0 = np.zeros((8, 8))

    for xi in gp:
        for eta in gp:
            # 形状函数对局部坐标的导数
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

            # 雅可比逆变换对应的导数
            dN_dx = dN_dxi / a
            dN_dy = dN_deta / b

            # 对应 8 个自由度的 B 矩阵
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i]     = dN_dx[i]
                B[1, 2 * i + 1] = dN_dy[i]
                B[2, 2 * i]     = dN_dy[i]
                B[2, 2 * i + 1] = dN_dx[i]

            # 累加积分贡献
            k0 += np.dot(B.T, np.dot(C, B)) * a * b

    return k0


def solve_elasticity(Nx, Ny, dx, dy, active_elements, element_dofs, is_active_dof, D, E, nu, constraints):
    """基于当前损伤场 D 组装全局刚度矩阵并求解有限元位移."""
    N_nodes = Nx * Ny
    N_dofs = 2 * N_nodes

    # 单单元刚度模板
    k0 = compute_q4_stiffness(dx, dy, 1.0, nu)

    rows = []
    cols = []
    data = []

    # 1. 组装活性单元刚度 (1 - D_e) * k_e
    for e in active_elements:
        dofs = element_dofs[e]
        De = D[e]
        ke = (1.0 - De) * E * k0
        for r in range(8):
            for c in range(8):
                rows.append(dofs[r])
                cols.append(dofs[c])
                data.append(ke[r, c])

    # 2. 为非活性自由度添加对角占优保护 (使其值为 0)
    for d in range(N_dofs):
        if not is_active_dof[d]:
            rows.append(d)
            cols.append(d)
            data.append(1.0)

    # 3. 采用 Penalty 方法施加边界约束
    F = np.zeros(N_dofs)
    K_penalty = 1e11 * E
    for d, val in constraints:
        rows.append(d)
        cols.append(d)
        data.append(K_penalty)
        F[d] += K_penalty * val

    # 构建全局稀疏刚度矩阵并求解
    K_global = csr_matrix((data, (rows, cols)), shape=(N_dofs, N_dofs))
    U = spsolve(K_global, F)

    # 提取约束力 (反力)
    reactions = {}
    for d, val in constraints:
        reactions[d] = K_penalty * (val - U[d])

    return U, reactions


def compute_strains_stresses(U, active_elements, element_dofs, dx, dy, D, E, nu):
    """计算活性单元中心处的应变和应力."""
    strains = {}
    stresses = {}
    factor = E / (1.0 - nu**2)

    for e in active_elements:
        dofs = element_dofs[e]
        # 节点位移
        u0, v0 = U[dofs[0]], U[dofs[1]]
        u1, v1 = U[dofs[2]], U[dofs[3]]
        u2, v2 = U[dofs[4]], U[dofs[5]]
        u3, v3 = U[dofs[6]], U[dofs[7]]

        # 单元中心双线性插值导数 (对应 xi=0, eta=0)
        exx = (u1 - u0 + u2 - u3) / (2.0 * dx)
        eyy = (v3 - v0 + v2 - v1) / (2.0 * dy)
        exy = 0.5 * ((u3 - u0 + u2 - u1) / (2.0 * dy) + (v1 - v0 + v2 - v3) / (2.0 * dx))

        strains[e] = np.array([exx, eyy, exy])

        # 考虑损伤退化的 constitutive 应力
        De = D[e]
        sxx = (1.0 - De) * factor * (exx + nu * eyy)
        syy = (1.0 - De) * factor * (eyy + nu * exx)
        sxy = (1.0 - De) * factor * (1.0 - nu) * exy

        stresses[e] = np.array([sxx, syy, sxy])

    return strains, stresses


# ===========================================================================
# 2. 巴西圆盘压裂多尺度求解器
# ===========================================================================
class BrazilianSplittingSolver:
    """巴西圆盘压裂实验 — 物理-AI 耦合求解器."""

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

        # 网格坐标
        self.x_coord = np.linspace(0, L, Nx)
        self.y_coord = np.linspace(0, H, Ny)
        self.X, self.Y = np.meshgrid(self.x_coord, self.y_coord)

        # 活性单元判定
        self.active_elements = []
        self.element_dofs = {}
        self.element_coords = {}
        self.is_active_element = np.zeros((Ny - 1, Nx - 1), dtype=bool)

        for j in range(Ny - 1):
            for i in range(Nx - 1):
                # 单元中心坐标
                xc = (i + 0.5) * self.dx
                yc = (j + 0.5) * self.dy
                if (xc - self.Xc)**2 + (yc - self.Yc)**2 <= R**2:
                    e = j * (Nx - 1) + i
                    self.active_elements.append(e)
                    self.is_active_element[j, i] = True
                    self.element_coords[e] = (j, i)

                    # 逆时针映射 4 个节点的 ID
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

        # 活性节点与活性自由度映射
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

        # 初始化物理场
        self.D = {e: 0.0 for e in self.active_elements}
        self.U = np.zeros(2 * Nx * Ny)
        self.d_map = {e: -0.5 for e in self.active_elements}

        # 确定受载边界节点 (底侧约束面, 顶侧位移加载面)
        self.bottom_nodes = []
        self.top_nodes = []
        w_contact = 0.05 * R  # 5% 盘半径的压条接触宽度

        for j in range(Ny):
            for i in range(Nx):
                if self.is_active_node[j, i]:
                    xn = self.X[j, i]
                    yn = self.Y[j, i]
                    # 靠近中心轴加载线且在圆盘边缘的节点
                    if abs(xn - self.Xc) <= w_contact:
                        if yn <= self.Yc - 0.88 * R:
                            self.bottom_nodes.append(j * Nx + i)
                        elif yn >= self.Yc + 0.88 * R:
                            self.top_nodes.append(j * Nx + i)

        # 优化器设置
        self.net = PhysicsScaleNetSolid(input_dim=5, hidden_dim=32)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        # 记录器
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
        """计算活性单元上的局部特征向量与应变应力状态."""
        # 1. 组装损伤网格并计算空间梯度
        D_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for e in self.active_elements:
            j, i = self.element_coords[e]
            D_grid[j, i] = self.D[e]

        # 边界扩展梯度计算 (使用 numpy.gradient)
        grad_dy, grad_dx = np.gradient(D_grid, self.dy, self.dx)
        grad_D_mag = np.sqrt(grad_dx**2 + grad_dy**2)

        # 计算应变与应力
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

            # 主应变分析
            e1 = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy))**2 + exy**2)
            e2 = 0.5 * (exx + eyy) - np.sqrt((0.5 * (exx - eyy))**2 + exy**2)

            # Mazars 拉伸等效主应变
            eps_eqt = np.sqrt(max(0.0, e1)**2 + max(0.0, e2)**2)

            # 单元刚度退化驱动能量 (Strain energy density)
            factor = self.E / (1.0 - self.nu**2)
            Ye = 0.5 * factor * (exx**2 + eyy**2 + 2.0 * self.nu * exx * eyy + 2.0 * (1.0 - self.nu) * exy**2)
            Y[e] = Ye

            # 局部损伤演化驱动律 (Mazars 演化律)
            if eps_eqt > self.eps0:
                Dt = 1.0 - (self.eps0 / eps_eqt) * np.exp(-self.beta * (eps_eqt - self.eps0))
            else:
                Dt = 0.0
            delta_D_base[e] = max(0.0, Dt - De)

            # 特征提取与标准化映射
            sm = 0.5 * (sxx + syy)
            seq = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
            eta = sm / (seq + 1e-12)
            theta_bar = sxy / (seq + 1e-12)
            eps_norm = eps_eqt / self.eps0
            gp = self.l_c * grad_D_mag[j, i]

            # 构建神经网络输入特征 (各不变量范围映射在 [-1, 1] 附近)
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
        """执行自监督尺度一致性训练."""
        n_elem = len(element_indices)
        w_diss = np.zeros(n_elem)
        for idx, e in enumerate(element_indices):
            w_diss[idx] = Y[e] * delta_D_base[e]

        # 映射至网格进行双滤波操作 (Micro scale l = dx, Macro scale L = 3dx)
        w_grid = np.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            w_grid[j, i] = w_diss[idx]

        # 二维 3x3 均匀核滤波
        W1 = w_grid
        kernel = np.ones((3, 3)) / 9.0
        W2 = convolve(w_grid, kernel, mode="constant", cval=0.0)

        # 提取滤波尺度比 H
        H_list = []
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            H_list.append(W2[j, i] / (W1[j, i] + 1e-15))

        H_tensor = torch.tensor(H_list, dtype=torch.float32).unsqueeze(1)
        w_tensor = torch.tensor(w_diss, dtype=torch.float32).unsqueeze(1)

        # 网络前向传播
        d_pred = self.net(features_tensor)

        # 1. 固体尺度一致性 Germano 损失 (尺度比 λ = 3.0)
        lambda_val = 3.0
        loss_g = torch.sum(w_tensor * (lambda_val**d_pred - H_tensor)**2) / (torch.sum(w_tensor) + 1e-15)

        # 2. 物理锚点损失 (弹性区锚定 d=-0.5)
        D_tensor = torch.tensor([self.D[e] for e in element_indices], dtype=torch.float32).unsqueeze(1)
        elastic_mask = (D_tensor < 0.05).float()
        loss_e = torch.sum(elastic_mask * (d_pred - (-0.5))**2) / (elastic_mask.sum() + 1e-15)

        # 3. 物理锚点损失 (完全断裂区指数平滑约束 d -> -∞ => exp(2d)->0)
        fracture_mask = (D_tensor > 0.8).float()
        loss_f = torch.sum(fracture_mask * torch.exp(2.0 * d_pred)) / (fracture_mask.sum() + 1e-15)

        # 4. 空间平滑正则化
        d_grid = torch.zeros((self.Ny - 1, self.Nx - 1))
        for idx, e in enumerate(element_indices):
            j, i = self.element_coords[e]
            d_grid[j, i] = d_pred[idx, 0]
        diff_x = d_grid[:, 1:] - d_grid[:, :-1]
        diff_y = d_grid[1:, :] - d_grid[:-1, :]
        loss_s = (diff_x**2).mean() + (diff_y**2).mean()

        # 复合总损失
        loss_t = 1.0 * loss_g + 0.5 * loss_e + 0.3 * loss_f + 0.1 * loss_s

        # 梯度回传
        self.optimizer.zero_grad()
        if torch.isfinite(loss_t) and torch.sum(w_tensor).item() > 1e-8:
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        # 记录历史
        self.history["loss_total"].append(loss_t.item())
        self.history["loss_germano"].append(loss_g.item())
        self.history["loss_elastic"].append(loss_e.item())
        self.history["loss_fracture"].append(loss_f.item())
        self.history["loss_smooth"].append(loss_s.item())

        return d_pred.detach().numpy().flatten()

    def step_load(self, disp_val):
        """主加载循环: 施加约束 -> 求解弹性 -> 训练网络 -> 更新标度损伤场."""
        # 组装边界位移约束
        constraints = []
        for n in self.bottom_nodes:
            constraints.append((2 * n, 0.0))
            constraints.append((2 * n + 1, 0.0))  # 底部固定
        for n in self.top_nodes:
            constraints.append((2 * n, 0.0))
            constraints.append((2 * n + 1, -disp_val))  # 顶部下压

        # 1. 求解当前弹性损伤场下的力学响应
        self.U, reactions = solve_elasticity(
            self.Nx, self.Ny, self.dx, self.dy,
            self.active_elements, self.element_dofs, self.is_active_dof,
            self.D, self.E, self.nu, constraints
        )

        # 2. 统计当前总加载垂直反力大小
        total_reaction_force = 0.0
        for n in self.top_nodes:
            total_reaction_force += reactions.get(2 * n + 1, 0.0)
        self.history["load_disp"].append((disp_val, abs(total_reaction_force)))

        # 3. 计算提取局部应变特征与先验演化
        (features_tensor, element_indices, delta_D_base, Y,
         strains, stresses) = self.compute_features_and_strains()

        # 4. 执行神经网络的自监督学习优化
        d_pred_eval = self.train_scale_net(features_tensor, element_indices, delta_D_base, Y)

        # 5. 基于尺度不变算子代数更新下一时间步的损伤场
        # 设定单元-特征比 ratio = l/L_0 = 0.2
        ratio_val = 0.2
        for idx, e in enumerate(element_indices):
            de = d_pred_eval[idx]
            self.d_map[e] = de
            # 引入局部尺度幂律修正因子
            scaling_factor = ratio_val**(-de)
            self.D[e] = min(0.999, self.D[e] + scaling_factor * delta_D_base[e])

        self.history["mean_d"].append(np.mean(d_pred_eval))
        self.history["max_damage"].append(max(self.D.values()))

        return stresses, strains


# ===========================================================================
# 3. 主程序与可视化输出
# ===========================================================================
def save_snapshot(solver, step_idx, stresses):
    """绘制当前步的力学场多面板 snapshot 图并保存为文件."""
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

    # 将圆盘外元素设为 NaN 以提供干净的圆形图像
    D_grid[~solver.is_active_element] = np.nan
    sxx_grid[~solver.is_active_element] = np.nan
    d_grid[~solver.is_active_element] = np.nan

    # 建立 2x2 多面板画布
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Brazilian Disc Splitting Simulation - Step {step_idx:03d}", fontsize=14, fontweight="bold")

    # (a) 损伤场 D (裂纹垂直路径)
    im0 = axs[0, 0].contourf(XE, YE, D_grid, levels=50, cmap="inferno", vmin=0, vmax=1)
    axs[0, 0].set_title("Damage field D (Fracture Cracking)")
    axs[0, 0].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0, 0])

    # (b) 水平拉应力 σ_xx
    smax = max(abs(np.nanmin(sxx_grid)), abs(np.nanmax(sxx_grid))) if np.any(~np.isnan(sxx_grid)) else 1.0
    im1 = axs[0, 1].contourf(XE, YE, sxx_grid / 1e6, levels=50, cmap="coolwarm", vmin=-smax/1e6, vmax=smax/1e6)
    axs[0, 1].set_title("Horizontal Stress $\sigma_{xx}$ (MPa)")
    axs[0, 1].set_aspect("equal")
    fig.colorbar(im1, ax=axs[0, 1])

    # (c) 尺度谱指数 d(x)
    im2 = axs[1, 0].contourf(XE, YE, d_grid, levels=50, cmap="viridis", vmin=-3.0, vmax=-0.5)
    axs[1, 0].set_title("Scale Exponent $d(\mathbf{x})$")
    axs[1, 0].set_aspect("equal")
    fig.colorbar(im2, ax=axs[1, 0])

    # (d) 荷载-位移曲线 (Load-Displacement with Softening)
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
    """绘制最终完整的压裂多特征分析大图."""
    print("\n>>> 正在渲染最终成果报告...")
    ld = solver.history["load_disp"]
    disp_history = [x[0] * 1e3 for x in ld]
    force_history = [x[1] / 1e3 for x in ld]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 荷载位移全历史
    axs[0].plot(disp_history, force_history, "b-", lw=2, label="Load-displacement Curve")
    # 标注峰值
    peak_idx = np.argmax(force_history)
    axs[0].plot(disp_history[peak_idx], force_history[peak_idx], "rs", markersize=8,
                label=f"Peak: {force_history[peak_idx]:.2f} kN @ {disp_history[peak_idx]:.3f} mm")
    axs[0].set_xlabel("Displacement (mm)", fontsize=11)
    axs[0].set_ylabel("Load (kN)", fontsize=11)
    axs[0].set_title("Load-Displacement Response (Softening Curve)", fontsize=12, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 2. 自监督 Loss 与平均标度指数 d 的收敛曲线
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
    # ── 参数配置区 ──
    solver = BrazilianSplittingSolver(
        Nx=100, Ny=100,                # 网格分辨率
        L=2.0, H=2.0,                # 计算域尺寸 (m)
        R=0.8,                       # 巴西圆盘半径 (m)
        E=30e9,                      # 弹性模量 30 GPa
        nu=0.2,                      # 泊松比 0.2
        eps0=1.0e-4,                 # 损伤起始等效拉应变阈值 1.0e-4
        beta=4000.0,                 # 软化速率控制参数 (大则软化快)
        l_c=0.08                     # 局部化非局部正则特征长度
    )

    total_steps = 700
    disp_step = 6.0e-6  # 每加载步下压 6.0 微米 (总计下压 0.42 mm)

    print(f"\n========================================================")
    print(f"  >>> 巴西圆盘压裂实验物理-AI 多尺度有限元模型 (Q4 FEM) <<<")
    print(f"========================================================")
    print(f"网格大小: {solver.Nx}x{solver.Ny} (活性单元数: {len(solver.active_elements)})")
    print(f"材料属性: E = {solver.E/1e9:.1f} GPa, nu = {solver.nu}, eps0 = {solver.eps0}")
    print(f"加载制度: 步数 = {total_steps}, 单步位移 = {disp_step*1e6:.1f} μm (最大下压: {total_steps*disp_step*1000:.3f} mm)")
    print(f" snap输出: 每 5 步渲染并在 snapshots 文件夹保存劈裂应力-损伤双场图")
    print(f"========================================================\n")

    for step in range(1, total_steps + 1):
        disp_val = step * disp_step
        stresses, strains = solver.step_load(disp_val)

        # 记录关键指标
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

    # 导出全生命周期载荷曲线报告
    save_final_report(solver)
    print("\n>>> 劈裂计算结束。Snapshots 均已存入 snapshots/ 目录。")

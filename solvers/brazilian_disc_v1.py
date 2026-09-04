"""
巴西圆盘劈裂 — 显式 FEM + 尺度不变算子代数闭合 v1.0
==================================================
架构严格对标 bfs_projection_solver_v3.py (后台阶流动):
  Phase 1: 预热 (纯 FEM + Mazars 损伤, 无 NN)
  Phase 2: 耦合 (NN 预测 d(x), 标度修正损伤演化率)

核心: 平面应变 Q4 有限元, Mazars 脆性损伤, 位移控制加载
用法: python brazilian_disc_v1.py
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
# 0. 内嵌 PhysicsScaleNet — 固体版
# ===========================================================================
class PhysicsScaleNetSolid(nn.Module):
    """从局部力学不变量 (D, η, θ̄, ε_norm, g_D) 预测标度指数 d(x).

    输出: d(x) ∈ (-∞, -0.5]
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
        )
        # 初始化最后一层偏置为负值, 使初始 d ≈ -0.5 (弹性锚点)
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x):
        raw = self.net(x)
        # softplus 保证 d ≤ -0.5
        d = -0.5 - torch.nn.functional.softplus(raw)
        return d


# ===========================================================================
# 1. 平面应变 Q4 有限元工具
# ===========================================================================
def plane_strain_C(E, nu):
    """平面应变弹性矩阵 C (Voigt: [σ_xx, σ_yy, σ_xy])."""
    f = E / ((1 + nu) * (1 - 2 * nu))
    C = np.array([
        [f * (1 - nu), f * nu,       0.0],
        [f * nu,       f * (1 - nu), 0.0],
        [0.0,          0.0,          f * (1 - 2 * nu) / 2.0]
    ])
    return C


def q4_stiffness_template(dx, dy, C):
    """计算单个矩形 Q4 单元的刚度矩阵 (2×2 高斯积分)."""
    gp = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    a, b = dx / 2.0, dy / 2.0
    k0 = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN_dxi = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) * 0.25
            dN_deta = np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)]) * 0.25
            dN_dx = dN_dxi / a
            dN_dy = dN_deta / b
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i]     = dN_dx[i]
                B[1, 2*i + 1] = dN_dy[i]
                B[2, 2*i]     = dN_dy[i]
                B[2, 2*i + 1] = dN_dx[i]
            k0 += B.T @ C @ B * a * b
    return k0


def b_matrix_center(dx, dy):
    """单元中心 (ξ=0,η=0) 处的 B 矩阵."""
    dNdx = np.array([-1, 1, 1, -1]) / (2.0 * dx)
    dNdy = np.array([-1, -1, 1, 1]) / (2.0 * dy)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]     = dNdx[i]
        B[1, 2*i + 1] = dNdy[i]
        B[2, 2*i]     = dNdy[i]
        B[2, 2*i + 1] = dNdx[i]
    return B


# ===========================================================================
# 2. 巴西圆盘劈裂求解器
# ===========================================================================
class BrazilianDiscSolver:
    """巴西圆盘劈裂 — 显式 FEM + 尺度不变算子代数闭合."""

    def __init__(self, *,
                 Nx, Ny, L_domain, R, a_crack,
                 E, nu, sigma_t, K_Ic,
                 loading_half_width,
                 n_warmup, n_coupled, disp_step,
                 lambda_germano, lambda_elastic, lambda_fracture,
                 lambda_damage, lambda_smooth,
                 l_c, l_d, flat_height=0.0, beta_crack=0.0, hidden_dim=32, lr=1e-3):

        self.Nx, self.Ny = Nx, Ny
        self.L = L_domain
        self.R, self.a_crack = R, a_crack
        self.E, self.nu = E, nu
        self.sigma_t = sigma_t
        self.K_Ic = K_Ic
        self.n_warmup = n_warmup
        self.n_coupled = n_coupled
        self.disp_step = disp_step
        self.l_c, self.l_d = l_c, l_d
        self.flat_height = flat_height
        self.beta_crack = beta_crack
        self.lam_g = lambda_germano
        self.lam_e = lambda_elastic
        self.lam_f = lambda_fracture
        self.lam_d = lambda_damage
        self.lam_s = lambda_smooth

        # ---- 网格 ----
        self.dx = L_domain / (Nx - 1)
        self.dy = L_domain / (Ny - 1)
        self.x_node = np.linspace(0, L_domain, Nx)
        self.y_node = np.linspace(0, L_domain, Ny)
        self.Xc, self.Yc = L_domain / 2.0, L_domain / 2.0

        # ---- 材料 ----
        self.C = plane_strain_C(E, nu)
        self.k0 = q4_stiffness_template(self.dx, self.dy, self.C)
        self.k0_unit = q4_stiffness_template(self.dx, self.dy,
                                              plane_strain_C(1.0, nu))
        self.B_cen = b_matrix_center(self.dx, self.dy)
        self.eps0 = sigma_t / E   # 损伤起始阈值 (等效拉应变)
        # 断裂能 (平面应变)
        Gf = K_Ic**2 * (1 - nu**2) / E
        gf = Gf / self.dx       # 单元能量密度
        w_e = sigma_t**2 / (2 * E)
        self.beta_soft = sigma_t / max(gf - w_e, 1e-12)  # Mazars 软化指数
        print(f"    Mazars: eps0={self.eps0:.2e}, beta={self.beta_soft:.1f}, "
              f"Gf={Gf:.4e} MPa·mm")

        # ---- 圆盘活性单元 ----
        Ne_x, Ne_y = Nx - 1, Ny - 1
        self.active = []
        self.elem_ji = {}    # elem_id → (j, i)
        self.is_active_elem = np.zeros((Ne_y, Ne_x), dtype=bool)

        for j in range(Ne_y):
            for i in range(Ne_x):
                xc = (i + 0.5) * self.dx
                yc = (j + 0.5) * self.dy
                in_circle = (xc - self.Xc)**2 + (yc - self.Yc)**2 <= R**2
                if flat_height > 0:
                    in_flat = abs(yc - self.Yc) <= (R - flat_height)
                else:
                    in_flat = True
                
                if in_circle and in_flat:
                    e = j * Ne_x + i
                    self.active.append(e)
                    self.elem_ji[e] = (j, i)
                    self.is_active_elem[j, i] = True

        self.n_active = len(self.active)

        # ---- 单元自由度映射 (预计算向量化索引) ----
        self.elem_dof_array = np.zeros((self.n_active, 8), dtype=int)
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            n0 = j * Nx + i
            n1 = j * Nx + i + 1
            n2 = (j + 1) * Nx + i + 1
            n3 = (j + 1) * Nx + i
            self.elem_dof_array[idx] = [
                2*n0, 2*n0+1, 2*n1, 2*n1+1,
                2*n2, 2*n2+1, 2*n3, 2*n3+1
            ]

        # 活性 DOF 标记
        N_dof = 2 * Nx * Ny
        self.N_dof = N_dof
        self.is_active_dof = np.zeros(N_dof, dtype=bool)
        self.is_active_dof[self.elem_dof_array.flatten()] = True

        # 预计算 COO 稀疏组装索引 (一次性计算)
        k0f = self.k0_unit.flatten()  # 64 values
        self._coo_rows = np.zeros(self.n_active * 64, dtype=int)
        self._coo_cols = np.zeros(self.n_active * 64, dtype=int)
        self._k0_tile = np.tile(k0f, self.n_active)
        for idx in range(self.n_active):
            dofs = self.elem_dof_array[idx]
            r = np.repeat(dofs, 8)
            c = np.tile(dofs, 8)
            self._coo_rows[idx*64:(idx+1)*64] = r
            self._coo_cols[idx*64:(idx+1)*64] = c

        # 非活性 DOF 对角保护
        inactive = np.where(~self.is_active_dof)[0]
        self._diag_rows = inactive
        self._diag_cols = inactive
        self._diag_vals = np.ones(len(inactive))

        # ---- 预制裂缝 (以角度 beta_crack 偏转, 相对竖直加载方向) ----
        self.D = np.zeros(self.n_active)   # 损伤场 (flat array, 与 active 对应)
        if a_crack > 0:
            rad = np.radians(beta_crack)
            cos_b, sin_b = np.cos(rad), np.sin(rad)
            for idx, e in enumerate(self.active):
                j, i = self.elem_ji[e]
                xc = (i + 0.5) * self.dx
                yc = (j + 0.5) * self.dy
                dx_c = xc - self.Xc
                dy_c = yc - self.Yc
                # 旋转到裂缝局部坐标系
                x_prime = dx_c * cos_b - dy_c * sin_b
                y_prime = dx_c * sin_b + dy_c * cos_b
                if abs(x_prime) < self.dx * 0.6 and abs(y_prime) < a_crack:
                    self.D[idx] = 0.999

        # ---- 加载与支撑节点 ----
        self.top_nodes = []
        self.bottom_nodes = []
        self.top_center_node = None
        self.bottom_center_node = None
        
        min_top_dist = 1e9
        min_bot_dist = 1e9
        hw = loading_half_width
        
        for i in range(Nx):
            xn = self.x_node[i]
            if abs(xn - self.Xc) > hw:
                continue
            # 找该列中最高和最低的活性节点
            j_top, j_bot = -1, Ny
            for j in range(Ny):
                nid = j * Nx + i
                if self.is_active_dof[2 * nid]:
                    if j < j_bot: j_bot = j
                    if j > j_top: j_top = j
            if j_top > 0:
                node = j_top * Nx + i
                self.top_nodes.append(node)
                dist = abs(xn - self.Xc)
                if dist < min_top_dist:
                    min_top_dist = dist
                    self.top_center_node = node
            if j_bot < Ny:
                node = j_bot * Nx + i
                self.bottom_nodes.append(node)
                dist = abs(xn - self.Xc)
                if dist < min_bot_dist:
                    min_bot_dist = dist
                    self.bottom_center_node = node

        # ---- 位移 & 力学场 ----
        self.U = np.zeros(N_dof)
        self.strains = np.zeros((self.n_active, 3))
        self.stresses = np.zeros((self.n_active, 3))
        self._step_counter = 0

        # ---- 神经网络 ----
        self.nn = PhysicsScaleNetSolid(input_dim=5, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.nn_active = False
        self.d_field = np.full(self.n_active, -0.5)

        # ---- 历史记录 ----
        self.history = {
            "load_disp": [],
            "loss_total": [], "loss_germano": [], "loss_elastic": [],
            "loss_fracture": [], "loss_smooth": [],
            "mean_d": [], "max_damage": [],
        }

    # =====================================================================
    # I. 弹性求解 (稀疏直接法)
    # =====================================================================
    def solve_elasticity(self, disp_val):
        """组装损伤刚度 → 施加 BC → 稀疏求解."""
        # 向量化组装 (1-D)*E * k0_unit
        scale = np.repeat((1 - self.D) * self.E, 64)
        elem_vals = scale * self._k0_tile

        # 合并: 单元刚度 + 非活性 DOF 保护
        all_r = np.concatenate([self._coo_rows, self._diag_rows])
        all_c = np.concatenate([self._coo_cols, self._diag_cols])
        all_v = np.concatenate([elem_vals, self._diag_vals])

        # 惩罚法施加 BC
        K_pen = 1e10 * self.E
        F = np.zeros(self.N_dof)
        bc_r, bc_c, bc_v = [], [], []
        
        # Bottom nodes: constrain uy = 0 (allow sliding in ux)
        for n in self.bottom_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
            F[2*n+1] += K_pen * 0.0
            
        # Top nodes: constrain uy = -disp_val (allow sliding in ux)
        for n in self.top_nodes:
            bc_r.append(2*n+1); bc_c.append(2*n+1); bc_v.append(K_pen)
            F[2*n+1] += K_pen * (-disp_val)
            
        # Constrain ux = 0 only at the center top and bottom contact nodes to prevent rigid body motion
        if self.bottom_center_node is not None:
            bc_r.append(2*self.bottom_center_node)
            bc_c.append(2*self.bottom_center_node)
            bc_v.append(K_pen)
            F[2*self.bottom_center_node] += K_pen * 0.0
            
        if self.top_center_node is not None:
            bc_r.append(2*self.top_center_node)
            bc_c.append(2*self.top_center_node)
            bc_v.append(K_pen)
            F[2*self.top_center_node] += K_pen * 0.0

        all_r = np.concatenate([all_r, np.array(bc_r, dtype=int)])
        all_c = np.concatenate([all_c, np.array(bc_c, dtype=int)])
        all_v = np.concatenate([all_v, np.array(bc_v)])

        K = csr_matrix((all_v, (all_r, all_c)), shape=(self.N_dof, self.N_dof))
        self.U = spsolve(K, F)

        # 反力统计
        total_fy = 0.0
        for n in self.top_nodes:
            total_fy += K_pen * (-disp_val - self.U[2*n+1])
        return abs(total_fy)

    # =====================================================================
    # II. 应变与应力 (向量化)
    # =====================================================================
    def compute_strains_stresses(self):
        u_elem = self.U[self.elem_dof_array]    # (n_active, 8)
        self.strains = u_elem @ self.B_cen.T    # (n_active, 3) → [exx, eyy, gxy]
        sig_undamaged = self.strains @ self.C.T  # (n_active, 3) → [sxx, syy, sxy]
        self.stresses = sig_undamaged * (1 - self.D)[:, None]

    # =====================================================================
    # III. Mazars 损伤基准增量
    # =====================================================================
    def compute_damage_base(self):
        """Mazars 脆性损伤: 等效拉应变准则."""
        exx = self.strains[:, 0]
        eyy = self.strains[:, 1]
        exy = self.strains[:, 2] * 0.5   # gxy → exy

        # 主应变
        e_avg = 0.5 * (exx + eyy)
        e_diff = np.sqrt((0.5 * (exx - eyy))**2 + exy**2)
        e1 = e_avg + e_diff
        e2 = e_avg - e_diff

        # Mazars 等效拉应变 (Macaulay bracket)
        eps_eq = np.sqrt(np.maximum(e1, 0)**2 + np.maximum(e2, 0)**2)

        # 损伤目标值
        D_target = np.where(
            eps_eq > self.eps0,
            1.0 - (self.eps0 / (eps_eq + 1e-30)) * np.exp(
                -self.beta_soft * (eps_eq - self.eps0)),
            0.0
        )
        delta_D = np.maximum(0.0, D_target - self.D)
        return delta_D, eps_eq

    # =====================================================================
    # IV. 特征提取 (5D 不变量)
    # =====================================================================
    def compute_features(self):
        sxx = self.stresses[:, 0]
        syy = self.stresses[:, 1]
        sxy = self.stresses[:, 2]
        szz = self.nu * (sxx + syy)   # 平面应变 out-of-plane 应力

        # 应力不变量 (3D)
        sm = (sxx + syy + szz) / 3.0
        Sxx, Syy, Szz = sxx - sm, syy - sm, szz - sm
        J2 = 0.5 * (Sxx**2 + Syy**2 + Szz**2) + sxy**2
        seq = np.sqrt(3.0 * J2 + 1e-30)

        eta = sm / (seq + 1e-12)                   # 三轴度
        J3 = Sxx * Syy * Szz - Szz * sxy**2
        cos_arg = np.clip(27 * J3 / (2 * seq**3 + 1e-30), -1, 1)
        theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos_arg)  # Lode 角

        # 等效拉应变 (已在 compute_damage_base 中计算, 此处重新计算)
        exx = self.strains[:, 0]
        eyy = self.strains[:, 1]
        exy = self.strains[:, 2] * 0.5
        e_avg = 0.5 * (exx + eyy)
        e_diff = np.sqrt((0.5*(exx-eyy))**2 + exy**2)
        eps_eq = np.sqrt(np.maximum(e_avg+e_diff, 0)**2 +
                         np.maximum(e_avg-e_diff, 0)**2)

        # 损伤场空间梯度
        Ne_x = self.Nx - 1
        D_grid = np.zeros((self.Ny - 1, Ne_x))
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            D_grid[j, i] = self.D[idx]
        gdy, gdx = np.gradient(D_grid, self.dy, self.dx)
        grad_mag = np.sqrt(gdx**2 + gdy**2)

        # 组装特征
        f1 = self.D
        f2 = np.tanh(eta)
        f3 = np.tanh(theta_bar)
        f4 = np.tanh(eps_eq / self.eps0 - 1.0)
        gD_arr = np.zeros(self.n_active)
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            gD_arr[idx] = self.l_c * grad_mag[j, i]
        f5 = np.tanh(gD_arr)

        F_np = np.stack([f1, f2, f3, f4, f5], axis=1)
        return torch.tensor(F_np, dtype=torch.float32)

    # =====================================================================
    # V. Germano 自监督信号
    # =====================================================================
    def compute_germano_signal(self, delta_D_base):
        """双滤波耗散功差 → H_solid."""
        Ne_x = self.Nx - 1
        # 耗散功密度 φ = Y · ΔD
        # 驱动力 Y ≈ 弹性应变能密度 / (1-D)²
        exx = self.strains[:, 0]
        eyy = self.strains[:, 1]
        exy = self.strains[:, 2] * 0.5
        W = 0.5 * (self.stresses[:, 0] * exx + self.stresses[:, 1] * eyy
                    + 2 * self.stresses[:, 2] * exy)
        Y = W / ((1 - self.D)**2 + 1e-30)
        phi = Y * delta_D_base       # 基准耗散功密度

        # 映射到网格
        phi_grid = np.zeros((self.Ny - 1, Ne_x))
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            phi_grid[j, i] = phi[idx]

        # 3×3 盒式滤波 (test filter)
        kernel = np.ones((3, 3)) / 9.0
        phi_test = convolve(phi_grid, kernel, mode="constant", cval=0.0)

        # 提取 H_solid = φ_test / φ_grid (仅在 φ_grid 显著区域有效)
        H_arr = np.zeros(self.n_active)
        w_arr = np.zeros(self.n_active)
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            p_local = phi_grid[j, i]
            if p_local > 1e-15:
                H_arr[idx] = phi_test[j, i] / p_local
                w_arr[idx] = p_local  # 权重 = 耗散量

        return H_arr, w_arr

    # =====================================================================
    # VI. 混合损失函数 (5 项)
    # =====================================================================
    def compute_loss(self, d_pred, phi_grid_flat, phi_test_flat):
        n = self.n_active
        lambda_L = 3.0   # test/grid 尺度比

        # 1. 稳定的 Germano 损失 (无除法形式)
        phi_g_t = torch.tensor(phi_grid_flat, dtype=torch.float32).unsqueeze(1)
        phi_t_t = torch.tensor(phi_test_flat, dtype=torch.float32).unsqueeze(1)
        pred_ratio = lambda_L ** d_pred   # 3^d
        loss_g = torch.sum((pred_ratio * phi_g_t - phi_t_t)**2) / (torch.sum(phi_g_t**2) + 1e-15)

        # 2. 弹性锚点: D < 0.01 → d = -0.5
        D_t = torch.tensor(self.D, dtype=torch.float32).unsqueeze(1)
        mask_e = (D_t < 0.01).float()
        loss_e = torch.sum(mask_e * (d_pred - (-0.5))**2) / (mask_e.sum() + 1e-15)

        # 3. 断裂锚点: D > 0.9 → exp(2d) → 0
        mask_f = (D_t > 0.9).float()
        loss_f = torch.sum(mask_f * torch.exp(2.0 * d_pred)) / (mask_f.sum() + 1e-15)

        # 4. 本构先验: d ≈ -0.5 + ln(1-D) * c1
        c1 = 0.3
        D_np = self.D.copy()
        D_np = np.clip(D_np, 0, 0.999)
        f_const = -0.5 + np.log(1.0 - D_np) * c1
        f_t = torch.tensor(f_const, dtype=torch.float32).unsqueeze(1)
        loss_d = torch.mean((d_pred - f_t)**2)

        # 5. 空间平滑 (支持反向传播)
        Ne_x = self.Nx - 1
        Ne_y = self.Ny - 1
        d_grid_flat = torch.full((Ne_y * Ne_x, 1), -0.5, dtype=torch.float32)
        d_grid_flat[self.active] = d_pred
        d_grid = d_grid_flat.view(Ne_y, Ne_x)
        
        gdx = (d_grid[:, 1:] - d_grid[:, :-1]) / self.dx
        gdy = (d_grid[1:, :] - d_grid[:-1, :]) / self.dy
        loss_s = self.l_d**2 * (torch.mean(gdx**2) + torch.mean(gdy**2))

        loss_total = (self.lam_g * loss_g + self.lam_e * loss_e +
                      self.lam_f * loss_f + self.lam_d * loss_d +
                      self.lam_s * loss_s)
        return loss_total, loss_g, loss_e, loss_f, loss_s

    # =====================================================================
    # VII. 损伤更新 (含 (1-D) 饱和约束)
    # =====================================================================
    def update_damage(self, delta_D_base, d_field_np, use_scaling=True):
        """λ^d × ΔD_base, with saturation."""
        if use_scaling:
            ratio = 0.3        # l / L0
            scale = ratio ** d_field_np  # Correct exponent (no minus sign)
            scale = np.clip(scale, 0.1, 10.0)   # 防极端
            dD = scale * delta_D_base
        else:
            dD = delta_D_base
        self.D = np.minimum(0.999, self.D + dD)

    # =====================================================================
    # VIII. 纯 FEM 预热步 (无 NN)
    # =====================================================================
    def step_fem_only(self, disp_val):
        """Phase 1: 纯 Mazars 损伤, 不涉及神经网络."""
        F_reaction = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base()
        self.update_damage(delta_D, self.d_field, use_scaling=False)

        self.history["load_disp"].append((disp_val, F_reaction))
        self.history["max_damage"].append(np.max(self.D))
        return F_reaction

    # =====================================================================
    # IX. 耦合步 (NN + FEM)
    # =====================================================================
    def step_coupled(self, disp_val):
        """Phase 2: NN 预测 d(x), 标度修正损伤演化."""
        self._step_counter += 1

        # 1. 弹性求解
        F_reaction = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base()

        # 2. 特征提取 + NN 前向传播
        feats = self.compute_features()
        d_pred = self.nn(feats)

        # 3. 计算耗散功密度 phi
        exx = self.strains[:, 0]
        eyy = self.strains[:, 1]
        exy = self.strains[:, 2] * 0.5   # gxy → exy
        W = 0.5 * (self.stresses[:, 0] * exx + self.stresses[:, 1] * eyy
                    + 2 * self.stresses[:, 2] * exy)
        Y = W / ((1 - self.D)**2 + 1e-30)
        phi = Y * delta_D       # 基准耗散功密度

        # 4. 映射到网格并进行 3x3 盒式测试滤波
        Ne_x = self.Nx - 1
        phi_grid = np.zeros((self.Ny - 1, Ne_x))
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            phi_grid[j, i] = phi[idx]

        kernel = np.ones((3, 3)) / 9.0
        phi_test = convolve(phi_grid, kernel, mode="constant", cval=0.0)

        # 5. 提取活性单元对应的展平数组
        phi_grid_flat = phi
        phi_test_flat = np.zeros(self.n_active)
        for idx, e in enumerate(self.active):
            j, i = self.elem_ji[e]
            phi_test_flat[idx] = phi_test[j, i]

        # 6. 损失计算 & 反向传播
        loss_t, loss_g, loss_e, loss_f, loss_s = self.compute_loss(
            d_pred, phi_grid_flat, phi_test_flat)
        self.optimizer.zero_grad()
        if torch.isfinite(loss_t):
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
            self.optimizer.step()

        # 7. 以更新后的网络重新预测 d(x)
        with torch.no_grad():
            d_eval = self.nn(feats).numpy().flatten()
        self.d_field = d_eval

        # 8. 标度修正损伤更新
        self.update_damage(delta_D, d_eval, use_scaling=True)

        # 记录
        self.history["load_disp"].append((disp_val, F_reaction))
        self.history["loss_total"].append(loss_t.item())
        self.history["loss_germano"].append(loss_g.item())
        self.history["loss_elastic"].append(loss_e.item())
        self.history["loss_fracture"].append(loss_f.item())
        self.history["loss_smooth"].append(loss_s.item())
        self.history["mean_d"].append(np.mean(d_eval))
        self.history["max_damage"].append(np.max(self.D))

        return F_reaction

    # =====================================================================
    # X. 步骤分派 (对标 bfs_projection_solver_v3.py)
    # =====================================================================
    def step(self, disp_val, step_idx):
        if step_idx < self.n_warmup:
            F = self.step_fem_only(disp_val)
            return F, True
        else:
            if not self.nn_active:
                self.nn_active = True
                print(f"\n  >>> 预热完成 (step {step_idx}), 激活 NN + 标度修正")
            F = self.step_coupled(disp_val)
            return F, False


# ===========================================================================
# 3. 主程序 — 参数配置 + 运行 + 可视化
# ===========================================================================
if __name__ == "__main__":
    # ── 参数 (来自用户提供的砂岩/石灰岩典型值) ──
    solver = BrazilianDiscSolver(
        Nx=80, Ny=80,
        L_domain=60.0,         # 计算域 60×60 mm
        R=25.0,                # 圆盘半径 25mm (直径 50mm)
        flat_height=0.4,       # 平台高度 0.4mm (FBD截断)
        beta_crack=45.0,       # 预制裂缝偏角 (45度, 混合 I-II 型)
        a_crack=5.0,           # 预制裂缝半长 5mm (2a=10mm, a/R=0.2)
        E=30000.0,             # 弹性模量 30 GPa (MPa)
        nu=0.25,               # 泊松比
        sigma_t=6.0,           # 抗拉强度 6 MPa
        K_Ic=31.62,            # I型断裂韧性 1.0 MPa√m → 31.62 MPa·√mm
        loading_half_width=4.4,# 加载宽度 4.4mm (微小于平台半宽 4.45mm)
        # ── 时间步进 ──
        n_warmup=60,           # 预热步数 (纯 Mazars 损伤)
        n_coupled=200,          # 耦合步数 (NN + 标度修正)
        disp_step=3.0e-3,      # 每步下压 3 μm
        # ── 损失权重 ──
        lambda_germano=0.3,
        lambda_elastic=0.5,
        lambda_fracture=0.3,
        lambda_damage=0.2,
        lambda_smooth=0.1,
        # ── 长度尺度 ──
        l_c=0.5,               # 非局部特征长度 (mm)
        l_d=1.0,               # 标度场平滑长度 (mm)
        lr=2e-3,
    )

    total_steps = solver.n_warmup + solver.n_coupled
    print(f"\n{'='*60}")
    print(f"  巴西圆盘劈裂 — 显式 FEM + 尺度不变算子代数 v1.0")
    print(f"{'='*60}")
    print(f"  网格: {solver.Nx}×{solver.Ny}, 活性单元: {solver.n_active}")
    print(f"  材料: E={solver.E:.0f} MPa, ν={solver.nu}, σ_t={solver.sigma_t} MPa")
    print(f"  裂缝: 2a={2*solver.a_crack:.0f} mm (a/R={solver.a_crack/solver.R:.2f})")
    print(f"  加载: {total_steps} 步 × {solver.disp_step*1e3:.1f} μm = "
          f"{total_steps*solver.disp_step:.3f} mm")
    print(f"  预热: {solver.n_warmup} 步 → 耦合: {solver.n_coupled} 步")
    print(f"{'='*60}\n")

    os.makedirs("snapshots_brazilian", exist_ok=True)

    # ---- 可视化函数 ----
    def save_snapshot(step_idx):
        Ne_x = solver.Nx - 1
        xe = np.linspace(solver.dx/2, solver.L - solver.dx/2, Ne_x)
        ye = np.linspace(solver.dy/2, solver.L - solver.dy/2, solver.Ny - 1)
        XE, YE = np.meshgrid(xe, ye)

        D_g = np.full((solver.Ny-1, Ne_x), np.nan)
        sxx_g = np.full((solver.Ny-1, Ne_x), np.nan)
        d_g = np.full((solver.Ny-1, Ne_x), np.nan)

        for idx, e in enumerate(solver.active):
            j, i = solver.elem_ji[e]
            D_g[j, i] = solver.D[idx]
            sxx_g[j, i] = solver.stresses[idx, 0]
            d_g[j, i] = solver.d_field[idx]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Brazilian Disc Splitting — Step {step_idx:03d}  "
                     f"(δ={step_idx*solver.disp_step*1e3:.1f} μm)",
                     fontsize=13, fontweight="bold")

        # (a) 损伤场
        im0 = axs[0,0].contourf(XE, YE, D_g, levels=np.linspace(0,1,51),
                                 cmap="inferno")
        axs[0,0].set_title("Damage D"); axs[0,0].set_aspect("equal")
        fig.colorbar(im0, ax=axs[0,0])

        # (b) 水平应力 σ_xx
        vmax = max(abs(np.nanmin(sxx_g)), abs(np.nanmax(sxx_g)), 1e-3)
        im1 = axs[0,1].contourf(XE, YE, sxx_g, levels=50, cmap="coolwarm",
                                 vmin=-vmax, vmax=vmax)
        axs[0,1].set_title(r"$\sigma_{xx}$ (MPa)"); axs[0,1].set_aspect("equal")
        fig.colorbar(im1, ax=axs[0,1])

        # (c) 标度指数 d(x)
        im2 = axs[1,0].contourf(XE, YE, d_g, levels=50, cmap="viridis",
                                 vmin=-3, vmax=-0.3)
        axs[1,0].set_title(r"Scale exponent $d(\mathbf{x})$")
        axs[1,0].set_aspect("equal")
        fig.colorbar(im2, ax=axs[1,0])

        # (d) 荷载-位移
        ld = solver.history["load_disp"]
        axs[1,1].plot([x[0]*1e3 for x in ld], [x[1]/1e3 for x in ld],
                       "b-o", ms=2, lw=1.5)
        axs[1,1].set_xlabel("Displacement (μm)")
        axs[1,1].set_ylabel("Load (kN)")
        axs[1,1].set_title("Load–Displacement"); axs[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"snapshots_brazilian/step_{step_idx:03d}.png", dpi=120)
        plt.close()

    # ---- 主循环 ----
    for t in range(total_steps):
        disp = (t + 1) * solver.disp_step
        F, is_warmup = solver.step(disp, t)

        # 打印进度
        if t % 5 == 0 or t == total_steps - 1:
            d_val = disp * 1e3
            f_kN = F / 1e3
            maxD = solver.history["max_damage"][-1]
            tag = "预热" if is_warmup else "耦合"
            extra = ""
            if not is_warmup and len(solver.history["loss_total"]) > 0:
                lt = solver.history["loss_total"][-1]
                md = solver.history["mean_d"][-1]
                extra = f" | Loss={lt:.2e} | <d>={md:.4f}"
            print(f"  [{tag}] Step {t+1:3d}/{total_steps} | "
                  f"δ={d_val:6.1f} μm | F={f_kN:7.2f} kN | "
                  f"max(D)={maxD:.4f}{extra}")

        # 快照
        if t % 5 == 0 or t == total_steps - 1:
            save_snapshot(t + 1)

    # ---- 最终成果图 ----
    print("\n>>> 渲染最终 6 面板分析图...")
    Ne_x = solver.Nx - 1
    xe = np.linspace(solver.dx/2, solver.L - solver.dx/2, Ne_x)
    ye = np.linspace(solver.dy/2, solver.L - solver.dy/2, solver.Ny - 1)
    XE, YE = np.meshgrid(xe, ye)

    D_g = np.full((solver.Ny-1, Ne_x), np.nan)
    sxx_g = np.full((solver.Ny-1, Ne_x), np.nan)
    syy_g = np.full((solver.Ny-1, Ne_x), np.nan)
    d_g = np.full((solver.Ny-1, Ne_x), np.nan)

    for idx, e in enumerate(solver.active):
        j, i = solver.elem_ji[e]
        D_g[j, i] = solver.D[idx]
        sxx_g[j, i] = solver.stresses[idx, 0]
        syy_g[j, i] = solver.stresses[idx, 1]
        d_g[j, i] = solver.d_field[idx]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Brazilian Disc Splitting — Scale-Invariant Operator Algebra",
                 fontsize=14, fontweight="bold")

    # (a) Damage
    im0 = axs[0,0].contourf(XE, YE, D_g, levels=np.linspace(0,1,51),
                              cmap="inferno")
    axs[0,0].set_title("Damage D"); axs[0,0].set_aspect("equal")
    fig.colorbar(im0, ax=axs[0,0])

    # (b) σ_xx
    vmax = max(abs(np.nanmin(sxx_g)), abs(np.nanmax(sxx_g)), 1e-3)
    im1 = axs[0,1].contourf(XE, YE, sxx_g, levels=50, cmap="coolwarm",
                              vmin=-vmax, vmax=vmax)
    axs[0,1].set_title(r"$\sigma_{xx}$ (MPa) — tensile stress")
    axs[0,1].set_aspect("equal")
    fig.colorbar(im1, ax=axs[0,1])

    # (c) σ_yy
    vmax2 = max(abs(np.nanmin(syy_g)), abs(np.nanmax(syy_g)), 1e-3)
    im2 = axs[0,2].contourf(XE, YE, syy_g, levels=50, cmap="coolwarm",
                              vmin=-vmax2, vmax=vmax2)
    axs[0,2].set_title(r"$\sigma_{yy}$ (MPa) — compressive stress")
    axs[0,2].set_aspect("equal")
    fig.colorbar(im2, ax=axs[0,2])

    # (d) d(x)
    im3 = axs[1,0].contourf(XE, YE, d_g, levels=50, cmap="viridis",
                              vmin=-3, vmax=-0.3)
    axs[1,0].set_title(r"$d(\mathbf{x})$ scale exponent")
    axs[1,0].set_aspect("equal")
    fig.colorbar(im3, ax=axs[1,0])

    # (e) Load–Displacement
    ld = solver.history["load_disp"]
    dd = [x[0]*1e3 for x in ld]
    ff = [x[1]/1e3 for x in ld]
    axs[1,1].plot(dd, ff, "b-", lw=2)
    pk = np.argmax(ff)
    axs[1,1].plot(dd[pk], ff[pk], "rs", ms=8,
                   label=f"Peak: {ff[pk]:.1f} kN @ {dd[pk]:.1f} μm")
    axs[1,1].set_xlabel("Displacement (μm)"); axs[1,1].set_ylabel("Load (kN)")
    axs[1,1].set_title("Load–Displacement Curve"); axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)

    # (f) Loss + <d> 收敛
    nc = len(solver.history["loss_total"])
    if nc > 0:
        st = np.arange(solver.n_warmup + 1, solver.n_warmup + nc + 1)
        ax2 = axs[1,2].twinx()
        axs[1,2].plot(st, solver.history["loss_total"], "k-", lw=1.5, label="Loss")
        ax2.plot(st, solver.history["mean_d"], "g--", lw=1.5, label=r"$\langle d \rangle$")
        axs[1,2].set_xlabel("Step"); axs[1,2].set_ylabel("Loss")
        ax2.set_ylabel(r"$\langle d \rangle$", color="g")
        axs[1,2].set_yscale("log")
        axs[1,2].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    axs[1,2].set_title("Loss + Mean d Convergence"); axs[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("brazilian_disc_result.png", dpi=150)
    plt.close()
    print(">>> 完成! → brazilian_disc_result.png")
    print(f">>> 快照已存入 snapshots_brazilian/ ({len(os.listdir('snapshots_brazilian'))} 帧)")

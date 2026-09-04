"""
crack.py — 巴西圆盘劈裂 尺度不变混合 CAE 求解器 v2.0
=====================================================
架构: 符号物理 (Term A) + NN 标度修正 (Term B) + Germano 自监督
理论: dD = (l/L₀)^d(x) × ΔD_base, d(x) 由 NN 从局部不变量预测
数值: 残余刚度 + 非局部 eps_eq + 自适应阻尼 + 指数截断

Phase 1 (预热): 纯 Mazars 损伤, 无 NN
Phase 2 (耦合): NN 预测 d(x), Germano 自监督, 标度修正损伤演化

用法: python crack.py
输出: snapshots/step_XXX.png + snapshots/coupled/step_XXX.png
"""

import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter

import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs("snapshots", exist_ok=True)
os.makedirs("snapshots/coupled", exist_ok=True)


# ===========================================================================
# 1. 物理引擎基座
# ===========================================================================
def plane_strain_C(E, nu):
    f = E / ((1 + nu) * (1 - 2 * nu))
    return np.array([[f*(1-nu), f*nu, 0], [f*nu, f*(1-nu), 0], [0, 0, f*(1-2*nu)/2]])

def q4_stiffness_template(dx, dy, C):
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    k0 = np.zeros((8,8))
    for xi in gp:
        for eta in gp:
            dNdxi = np.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])*0.25
            dNdeta = np.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])*0.25
            dNdx, dNdy = dNdxi/(dx/2), dNdeta/(dy/2)
            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i]=dNdx[i]; B[1,2*i+1]=dNdy[i]; B[2,2*i]=dNdy[i]; B[2,2*i+1]=dNdx[i]
            k0 += B.T @ C @ B * dx*dy/4
    return k0

def b_matrix_center(dx, dy):
    dNdx = np.array([-1,1,1,-1])/(2*dx)
    dNdy = np.array([-1,-1,1,1])/(2*dy)
    B = np.zeros((3,8))
    for i in range(4):
        B[0,2*i]=dNdx[i]; B[1,2*i+1]=dNdy[i]; B[2,2*i]=dNdy[i]; B[2,2*i+1]=dNdx[i]
    return B


# ===========================================================================
# 2. PhysicsScaleNet — 标度指数预测器 (Term B)
# ===========================================================================
class PhysicsScaleNetSolid(nn.Module):
    """从局部力学不变量 (D, η, θ̄, ε_norm, g_D) 预测标度指数 d(x).
       输出: d(x) ∈ (-∞, -0.5]  — 弹性锚点 d=-0.5"""
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.net[-1].bias, -5.0)  # 初始 d≈-0.5

    def forward(self, x):
        return -0.5 - nn.functional.softplus(self.net(x))


# ===========================================================================
# 3. CrackSolver — 尺度不变混合求解器
# ===========================================================================
class CrackSolver:
    def __init__(self, *,
                 Nx, Ny, L_domain, R, E, nu, sigma_t, K_Ic,
                 loading_half_width, n_warmup, n_coupled, disp_step,
                 flat_height=0.0, beta_crack=0.0, a_crack=0.0,
                 hidden_dim=32, lr=2e-3,
                 lam_g=0.3, lam_e=0.5, lam_f=0.3, lam_d=0.2, lam_s=0.1,
                 l_c=0.5, l_d=1.0):
        # ── 网格 ──
        self.Nx, self.Ny = Nx, Ny
        self.L = L_domain; self.R = R
        self.dx = L_domain/(Nx-1); self.dy = L_domain/(Ny-1)
        self.Xc = self.Yc = L_domain/2

        # ── 材料 ──
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

        # ── 数值参数 ──
        self.residual_stiffness = 1e-6
        self.damping_warmup = 0.3    # 预热阶段统一阻尼 (远低于 0.8, 防止级联)
        self.damping_base = 0.5      # 耦合阶段远场阻尼
        self.damping_fast = 0.7      # 耦合阶段高驱动力区阻尼
        self.exp_clip = 50.0
        self.nonlocal_radius = 1
        self.eps_eq_cap = self.eps0*200

        # ── 标度修正参数 ──
        self.scale_ratio = 0.3      # l/L₀ (子网格/特征长度比)

        # ════════════════════════════════════════════════════════
        # 圆形几何域 Masking
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

        # ── DOF 映射 & COO 组装 ──
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

        # ── 边界条件 ──
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

        # ── 预制裂缝 (含渐变过渡带) ──
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

        # ── 位移 & 力学场 ──
        self.U = np.zeros(N_dof)
        self.strains = np.zeros((self.n_active, 3))
        self.stresses = np.zeros((self.n_active, 3))

        # ── 步进 ──
        self.n_warmup = n_warmup; self.n_coupled = n_coupled
        self.disp_step = disp_step
        self.total_steps = n_warmup + n_coupled

        # ── NN & 标度场 ──
        self.nn = PhysicsScaleNetSolid(input_dim=5, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.nn_active = False
        self.d_field = np.full(self.n_active, -0.5)  # 初始弹性锚点
        self.lam_g = lam_g; self.lam_e = lam_e; self.lam_f = lam_f
        self.lam_d = lam_d; self.lam_s = lam_s
        self.l_c = l_c; self.l_d = l_d

        # ── 历史 ──
        self.history = {"load_disp": [], "max_damage": [],
                        "loss_total": [], "mean_d": []}
        self._step_counter = 0

    # =====================================================================
    # I. 弹性求解 (Term A)
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
    # II. 应变与应力
    # =====================================================================
    def compute_strains_stresses(self):
        u_elem = self.U[self.elem_dof_array]
        self.strains = u_elem @ self.B_cen.T
        self.stresses = (self.strains @ self.C.T)*(1-self.D)[:,None]

    # =====================================================================
    # III. Mazars 损伤 (非局部 + 自适应阻尼 + eps 上限放宽)
    # =====================================================================
    def compute_damage_base(self, phase="warmup"):
        exx, eyy, exy = self.strains[:,0], self.strains[:,1], self.strains[:,2]*0.5
        e_avg = 0.5*(exx+eyy); e_diff = np.sqrt((0.5*(exx-eyy))**2+exy**2)
        eps_eq = np.sqrt(np.maximum(e_avg+e_diff,0)**2+np.maximum(e_avg-e_diff,0)**2)

        if self.nonlocal_radius > 0:
            eps_grid = np.zeros((self.Ny-1, self.Nx-1))
            for idx, e in enumerate(self.active):
                j,i = self.elem_ji[e]; eps_grid[j,i] = eps_eq[idx]
            eps_nl = uniform_filter(eps_grid, size=2*self.nonlocal_radius+1, mode='constant', cval=0)
            for idx, e in enumerate(self.active):
                j,i = self.elem_ji[e]; eps_eq[idx] = eps_nl[j,i]

        eps_clip = np.clip(eps_eq, 0, self.eps_eq_cap)
        arg = np.clip(self.beta_soft*(eps_clip-self.eps0), 0, self.exp_clip)
        D_target = np.where(eps_clip>self.eps0,
                            1-(self.eps0/(eps_clip+1e-30))*np.exp(-arg), 0.0)

        driving = np.maximum(D_target-self.D, 0)
        if phase == "warmup":
            # 预热阶段: 统一低阻尼, 防止级联扩散
            adaptive_damping = np.full_like(driving, self.damping_warmup)
        else:
            # 耦合阶段: 自适应阻尼 (尖端加速, 远场稳定)
            adaptive_damping = np.where(driving>0.1, self.damping_fast, self.damping_base)
        return driving*adaptive_damping, eps_eq

    # =====================================================================
    # IV. 特征提取 — 5D 力学不变量 → NN 输入
    # =====================================================================
    def compute_features(self):
        sxx, syy, sxy = self.stresses[:,0], self.stresses[:,1], self.stresses[:,2]
        szz = self.nu*(sxx+syy)
        sm = (sxx+syy+szz)/3; Sxx, Syy, Szz = sxx-sm, syy-sm, szz-sm
        J2 = 0.5*(Sxx**2+Syy**2+Szz**2)+sxy**2
        seq = np.sqrt(3*J2+1e-30)
        eta = sm/(seq+1e-12)
        J3 = Sxx*Syy*Szz - Szz*sxy**2
        cos_arg = np.clip(27*J3/(2*seq**3+1e-30), -1, 1)
        theta_bar = 1-(2/np.pi)*np.arccos(cos_arg)
        exx, eyy, exy2 = self.strains[:,0], self.strains[:,1], self.strains[:,2]*0.5
        e_avg=0.5*(exx+eyy); e_diff=np.sqrt((0.5*(exx-eyy))**2+exy2**2)
        eps_eq = np.sqrt(np.maximum(e_avg+e_diff,0)**2+np.maximum(e_avg-e_diff,0)**2)

        D_grid = np.zeros((self.Ny-1, self.Nx-1))
        for idx,e in enumerate(self.active):
            j,i = self.elem_ji[e]; D_grid[j,i] = self.D[idx]
        gdy, gdx = np.gradient(D_grid, self.dy, self.dx)
        grad_mag = np.sqrt(gdx**2+gdy**2)
        gD_arr = np.zeros(self.n_active)
        for idx,e in enumerate(self.active):
            j,i = self.elem_ji[e]; gD_arr[idx] = self.l_c*grad_mag[j,i]

        F_np = np.stack([self.D, np.tanh(eta), np.tanh(theta_bar),
                         np.tanh(eps_eq/self.eps0-1), np.tanh(gD_arr)], axis=1)
        return torch.tensor(F_np, dtype=torch.float32)

    # =====================================================================
    # V. Germano 自监督信号
    # =====================================================================
    def compute_germano_signal(self, delta_D_base):
        exx, eyy, exy2 = self.strains[:,0], self.strains[:,1], self.strains[:,2]*0.5
        W = 0.5*(self.stresses[:,0]*exx+self.stresses[:,1]*eyy+2*self.stresses[:,2]*exy2)
        Y = W/((1-self.D)**2+1e-30)
        phi = Y*delta_D_base

        phi_grid = np.zeros((self.Ny-1, self.Nx-1))
        for idx,e in enumerate(self.active):
            j,i = self.elem_ji[e]; phi_grid[j,i] = phi[idx]
        phi_test = uniform_filter(phi_grid, size=3, mode='constant', cval=0)*9/9

        H_arr = np.zeros(self.n_active); w_arr = np.zeros(self.n_active)
        for idx,e in enumerate(self.active):
            j,i = self.elem_ji[e]
            p_local = phi_grid[j,i]
            if p_local > 1e-15:
                H_arr[idx] = phi_test[j,i]/p_local
                w_arr[idx] = p_local
        return H_arr, w_arr, phi, phi_test

    # =====================================================================
    # VI. 混合损失函数 (5 项)
    # =====================================================================
    def compute_loss(self, d_pred, phi_grid_flat, phi_test_flat):
        lambda_L = 3.0
        phi_g_t = torch.tensor(phi_grid_flat, dtype=torch.float32).unsqueeze(1)
        phi_t_t = torch.tensor(phi_test_flat, dtype=torch.float32).unsqueeze(1)
        pred_ratio = lambda_L**d_pred
        loss_g = torch.sum((pred_ratio*phi_g_t-phi_t_t)**2)/(torch.sum(phi_g_t**2)+1e-15)

        D_t = torch.tensor(self.D, dtype=torch.float32).unsqueeze(1)
        mask_e = (D_t<0.01).float()
        loss_e = torch.sum(mask_e*(d_pred-(-0.5))**2)/(mask_e.sum()+1e-15)

        mask_f = (D_t>0.9).float()
        loss_f = torch.sum(mask_f*torch.exp(2*d_pred))/(mask_f.sum()+1e-15)

        D_np = np.clip(self.D.copy(), 0, 0.999)
        f_const = -0.5+np.log(1-D_np)*0.3
        f_t = torch.tensor(f_const, dtype=torch.float32).unsqueeze(1)
        loss_d = torch.mean((d_pred-f_t)**2)

        d_grid_flat = torch.full(((self.Ny-1)*(self.Nx-1),), -0.5, dtype=torch.float32)
        d_grid_flat[torch.tensor(self.active)] = d_pred.squeeze()
        d_grid = d_grid_flat.view(self.Ny-1, self.Nx-1)
        gdx = (d_grid[:,1:]-d_grid[:,:-1])/self.dx
        gdy = (d_grid[1:,:]-d_grid[:-1,:])/self.dy
        loss_s = self.l_d**2*(torch.mean(gdx**2)+torch.mean(gdy**2))

        loss_total = (self.lam_g*loss_g+self.lam_e*loss_e+
                      self.lam_f*loss_f+self.lam_d*loss_d+self.lam_s*loss_s)
        return loss_total, loss_g, loss_e, loss_f, loss_s

    # =====================================================================
    # VII. 标度修正损伤更新
    # =====================================================================
    def update_damage(self, delta_D_base, d_field_np, use_scaling=True):
        if use_scaling:
            scale = np.clip(self.scale_ratio**d_field_np, 0.1, 10.0)
            dD = scale*delta_D_base
        else:
            dD = delta_D_base
        self.D = np.clip(self.D+dD, 0.0, 0.99999)

    # =====================================================================
    # VIII. Phase 1 — 纯 FEM 预热步
    # =====================================================================
    def step_fem_only(self, disp_val):
        F = self.solve_elasticity(disp_val)
        self.compute_strains_stresses()
        delta_D, eps_eq = self.compute_damage_base(phase="warmup")
        self.update_damage(delta_D, self.d_field, use_scaling=False)
        return F, eps_eq

    # =====================================================================
    # IX. Phase 2 — 耦合步 (NN + Germano + 标度修正)
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
    # X. 步骤分派
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
    # XI. 可视化
    # =====================================================================
    def _get_precrack_line(self, ax):
        """计算并绘制预制裂缝线段 (如果存在)"""
        if self.a_crack <= 0:
            return None, None

        rad = np.radians(self.beta_crack)
        cos_b, sin_b = np.cos(rad), np.sin(rad)

        # 裂缝长度: 2a (从圆心向两边各延伸 a)
        crack_len = 2 * self.a_crack

        # 计算线段起点和终点 (以圆心为中点)
        # 注意: Xc, Yc 是域的几何中心坐标
        # 裂缝方向: dx = sin(beta), dy = cos(beta)
        half_dx = (crack_len / 2) * sin_b
        half_dy = (crack_len / 2) * cos_b

        x_start = self.Xc - half_dx
        y_start = self.Yc - half_dy
        x_end = self.Xc + half_dx
        y_end = self.Yc + half_dy

        # 在所有子图上绘制黑线
        for i_ax, a in enumerate(ax):
            a.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=2.5, zorder=10)

            # 在 Damage D 图上标注 beta_crack 数值
            if i_ax == 0:
                # 线段中点 (圆心: 域的几何中心)
                x_mid = self.Xc
                y_mid = self.Yc

                # 标注文字位置 (沿垂直裂缝方向偏移)
                offset = self.R * 0.08
                # 垂直于裂缝的方向 (旋转90度)
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

        # 绘制预制裂缝线段和标注
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
# 4. 主程序
# ===========================================================================
if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # 参数说明 — 巴西圆盘劈裂试验 ISRM 标准
    # ═══════════════════════════════════════════════════════════════════
    # 网格: Nx=80, Ny=80, L_domain=60 mm, R=25 mm (φ50), flat=0.4 mm
    # 材料 (砂岩): E=30 GPa, ν=0.25, σ_t=6 MPa, K_Ic=1.0 MPa·√m
    # 裂缝: 2a=10 mm (a/R=0.2), β=45° (I-II 混合型)
    # 加载: disp_step=3 μm, 总 250 步 → 0.75 mm
    # 切换: 50 步预热后激活 NN (损伤起始附近)
    # NN: hidden=32, lr=2e-3, Lamé 损失权重
    # ═══════════════════════════════════════════════════════════════════

    # 统一随机种子（NN 在线训练可复现）
    torch.manual_seed(2026)
    np.random.seed(2026)

    solver = CrackSolver(
        Nx=80, Ny=80, L_domain=60.0, R=25.0, flat_height=0.4,
        E=30000.0, nu=0.25, sigma_t=6.0, K_Ic=31.62,
        loading_half_width=4.4, disp_step=3.0e-3,
        beta_crack=90.0, a_crack=5.0,
        n_warmup=50, n_coupled=500,     # 预热到损伤起始即切换 (d~150 μm)
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

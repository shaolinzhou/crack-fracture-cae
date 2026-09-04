# 深度技术报告：AI-CAE 物理智能计算引擎数学理论与实施架构

## 1. 物理模型与切线算子推导 (Mathematical Model)

在三维各向异性损伤框架下，材料应力张量 $\boldsymbol{\sigma}$ 与应变张量 $\boldsymbol{\varepsilon}$ 的本构关系定义为：
$$\sigma_{ij} = (1 - D_{ij}) C_{ijkl} \varepsilon_{kl}$$
其中 $\mathbf{D}$ 为二阶损伤张量，$C_{ijkl}$ 为初始完整材料弹性刚度张量。

### 1.1 一致性切线算子 (Consistent Tangent Operator)
切线算子 $\mathbf{C}^{ep}$ 定义为 $\frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}$，通过对损伤演化律 $D(\boldsymbol{\varepsilon}; \theta)$ 的链式法则求导展开：
$$\mathbf{C}^{ep} = \underbrace{(1 - D_{ij}) C_{ijkl}}_{\text{Term A: Elastic Physics}} - \underbrace{C_{ijkl} \varepsilon_{kl} \frac{\partial D_{ij}(\boldsymbol{\varepsilon}; \theta)}{\partial \varepsilon_{mn}}}_{\text{Term B: Neural Damage Gradient}}$$

### 1.2 数值稳定化处理 (Regularization)
为防止算子奇异（Singularity）及指数爆炸（Overflow）：
1. **残余刚度**：$\mathbf{C}_{\text{reg}} = (1 - D + \epsilon_{\text{res}}) \mathbf{C}_{\text{elastic}}$，其中 $\epsilon_{\text{res}} \approx 10^{-6}$。
2. **指数阻尼**：将 $D$ 的演化率引入阻尼因子 $\alpha \in (0, 1]$：$D_{n+1} = D_n + \alpha \cdot \Delta D_{\text{Mazars}}$。

---

## 2. 计算算法理论 (Algorithmic Theory)

### 2.1 混合微分逻辑 (Symbolic-Neural Hybrid)
*   **符号解析 (Term A)**：$C_{ijkl}$ 及其线性组合通过解析表达式预计算，无需自动微分。
*   **局部自动微分 (Term B)**：利用局部计算图截断。对于单元 $e$，我们计算其局部的损伤贡献：
    $$\mathbb{K}_e = \int_{\Omega_e} \mathbf{B}^T \mathbf{C}^{ep} \mathbf{B} d\Omega$$
    在 PyTorch 中，通过 `torch.autograd.grad(outputs=D_e, inputs=eps_e, create_graph=False)`，在计算完梯度后立即清除 $e$ 的计算图。

### 2.2 无矩阵共轭梯度法 (Matrix-Free PCG)
不再组装全局 $\mathbf{K}$，通过算子乘法实现平衡迭代：
$$\mathbf{y} = \mathbf{K} \cdot \mathbf{u} = \sum_{e=1}^{N_e} \mathbf{A}_e^T \left( \mathbb{K}_{e, \text{sym}} + \mathbb{K}_{e, \text{AI}} \right) \mathbf{A}_e \cdot \mathbf{u}$$
其中 $\mathbf{A}_e$ 为组装映射矩阵。

---

## 3. 核心计算步骤 (Pseudo-Code)

### 步骤 A：物理初始化与算子组装
```python
# 初始化物理算子
C_fixed = get_symbolic_tensor(E, nu) # 符号解析常数

# 主求解循环
for step in range(max_steps):
    # 1. PCG 算子乘法定义
    def apply_operator(u_global):
        f_int = zeros_like(u_global)
        for e in range(n_elements):
            eps_e = B_matrix[e] @ u_global[dof_map[e]]
            # Term A (符号项)
            sigma_phys = (1 - D[e]) * (C_fixed @ eps_e)
            # Term B (AI 局部梯度)
            dD_deps = local_autograd_kernel(eps_e, net)
            sigma_ai = - (eps_e @ dD_deps) * C_fixed @ eps_e
            # 组装力矢量
            f_int[dof_map[e]] += B_matrix[e].T @ (sigma_phys + sigma_ai)
        return f_int
    
    # 2. 迭代求解 Newton-Raphson / PCG
    u_new = pcg_solve(apply_operator, force_vector)
```

### 步骤 B：损伤演化更新
```python
def update_damage(eps, net, D_old):
    # 数值稳定化计算
    eps_eq = compute_equivalent_strain(eps)
    eps_clipped = clip(eps_eq, max=eps0*50)
    
    # 物理演化 + 指数截断
    D_new = 1.0 - (eps0 / eps_clipped) * exp(-clip(beta * (eps_clipped - eps0), 0, 50))
    
    # 演化阻尼更新
    D_updated = D_old + damping * max(0, D_new - D_old)
    return clip(D_updated, 0, 0.9999)

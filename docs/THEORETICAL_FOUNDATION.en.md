# Deep Technical Report: Mathematical Theory and Implementation Architecture of the AI-CAE Physics-Intelligent Computational Engine

> English backup translation of `THEORETICAL_FOUNDATION.md`.

## 1. Physical Model and Derivation of the Tangent Operator (Mathematical Model)

Within a three-dimensional anisotropic damage framework, the constitutive relation between the material stress tensor $\boldsymbol{\sigma}$ and the strain tensor $\boldsymbol{\varepsilon}$ is defined as:
$$\sigma_{ij} = (1 - D_{ij}) C_{ijkl} \varepsilon_{kl}$$
where $\mathbf{D}$ is a second-order damage tensor and $C_{ijkl}$ is the elastic stiffness tensor of the initially undamaged material.

### 1.1 Consistent Tangent Operator
The tangent operator $\mathbf{C}^{ep}$ is defined as $\frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}$ and is expanded by applying the chain rule to the damage evolution law $D(\boldsymbol{\varepsilon}; \theta)$:
$$\mathbf{C}^{ep} = \underbrace{(1 - D_{ij}) C_{ijkl}}_{\text{Term A: Elastic Physics}} - \underbrace{C_{ijkl} \varepsilon_{kl} \frac{\partial D_{ij}(\boldsymbol{\varepsilon}; \theta)}{\partial \varepsilon_{mn}}}_{\text{Term B: Neural Damage Gradient}}$$

### 1.2 Numerical Regularization
To prevent operator singularity and exponential overflow:
1. **Residual stiffness**: $\mathbf{C}_{\text{reg}} = (1 - D + \epsilon_{\text{res}}) \mathbf{C}_{\text{elastic}}$, with $\epsilon_{\text{res}} \approx 10^{-6}$.
2. **Evolution damping**: a damping factor $\alpha \in (0, 1]$ is introduced into the evolution rate of $D$: $D_{n+1} = D_n + \alpha \cdot \Delta D_{\text{Mazars}}$.

---

## 2. Algorithmic Theory

### 2.1 Symbolic–Neural Hybrid Differentiation
*   **Symbolic (Term A)**: $C_{ijkl}$ and its linear combinations are precomputed from closed-form expressions; no automatic differentiation is required.
*   **Local automatic differentiation (Term B)**: uses truncated local computational graphs. For element $e$, its local damage contribution is computed as
    $$\mathbb{K}_e = \int_{\Omega_e} \mathbf{B}^T \mathbf{C}^{ep} \mathbf{B} d\Omega$$
    In PyTorch this is realized via `torch.autograd.grad(outputs=D_e, inputs=eps_e, create_graph=False)`, which clears the computational graph of element $e$ as soon as the gradient is obtained.

### 2.2 Matrix-Free Conjugate Gradient (PCG)
The global $\mathbf{K}$ is no longer assembled; equilibrium iterations are carried out through operator products:
$$\mathbf{y} = \mathbf{K} \cdot \mathbf{u} = \sum_{e=1}^{N_e} \mathbf{A}_e^T \left( \mathbb{K}_{e, \text{sym}} + \mathbb{K}_{e, \text{AI}} \right) \mathbf{A}_e \cdot \mathbf{u}$$
where $\mathbf{A}_e$ is the assembly (scatter) mapping matrix.

---

## 3. Core Computational Steps (Pseudo-Code)

### Step A: Physical Initialization and Operator Assembly
```python
# Initialize the physical operator
C_fixed = get_symbolic_tensor(E, nu)  # closed-form symbolic constants

# Main solve loop
for step in range(max_steps):
    # 1. Define the PCG operator product
    def apply_operator(u_global):
        f_int = zeros_like(u_global)
        for e in range(n_elements):
            eps_e = B_matrix[e] @ u_global[dof_map[e]]
            # Term A (symbolic term)
            sigma_phys = (1 - D[e]) * (C_fixed @ eps_e)
            # Term B (local AI gradient)
            dD_deps = local_autograd_kernel(eps_e, net)
            sigma_ai = - (eps_e @ dD_deps) * C_fixed @ eps_e
            # Assemble the internal force vector
            f_int[dof_map[e]] += B_matrix[e].T @ (sigma_phys + sigma_ai)
        return f_int

    # 2. Iterative solve: Newton-Raphson / PCG
    u_new = pcg_solve(apply_operator, force_vector)
```

### Step B: Damage Evolution Update
```python
def update_damage(eps, net, D_old):
    # Numerically stabilized computation
    eps_eq = compute_equivalent_strain(eps)
    eps_clipped = clip(eps_eq, max=eps0*50)

    # Physical evolution + exponential truncation
    D_new = 1.0 - (eps0 / eps_clipped) * exp(-clip(beta * (eps_clipped - eps0), 0, 50))

    # Damped evolution update
    D_updated = D_old + damping * max(0, D_new - D_old)
    return clip(D_updated, 0, 0.9999)
```

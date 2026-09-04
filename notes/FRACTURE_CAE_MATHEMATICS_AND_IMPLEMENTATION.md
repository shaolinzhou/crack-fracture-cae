# 断裂损伤 CAE 子系统：数学理论基础、本构模型、数值方法与实现

## A Mathematical Report on the Scale-Invariant Hybrid Damage CAE Subsystem

**所属项目**：Project Crack — AI-CAE 物理智能计算引擎
**子系统目录**：`fracture/`（顶层独立求解器与原型、`fracture/FEA/` 通用求解器、共享库 `src/`）
**文档性质**：数学论文级表述 —— 从连续介质平衡出发，经本构与损伤热力学、离散化与求解、尺度不变神经闭合，到逐行实现映射。

---

## 0. 摘要 (Abstract)

本子系统针对准静态脆性/准脆性断裂的巴西圆盘劈裂试验，建立并实现了一条 **"连续损伤力学 + 位移控制显式交错有限元 + 尺度不变（scale-invariant）神经算子闭合"** 的完整数值路线：

1. 在连续层面给出各向同性标量损伤框架下的边值问题，损伤演化采用 Mazars 拉应变准则，软化参数由断裂能 $\mathcal G_f$ 与网格特征长度 $l_c$ 能量标定；
2. 在离散层面采用平面应变 Q4 等参元、$2\times2$ 高斯积分、损伤降级割线刚度 $K(D)$ 的稀疏组装（罚边界 + 非活性自由度保护），并以矩阵自由 PCG 作为与稀疏直接法逐位一致的替代求解器；
3. 为克服"网格尺度依赖"与"裂纹近奇异耗散不可解析"两个根本困难，引入**损伤演化率的多尺度幂律修正** $\Delta D = (l/L_0)^{d(\boldsymbol x)}\Delta D_{\mathrm{Mazars}}$，其中局部尺度指数 $d(\boldsymbol x)\le -1/2$ 由一个小型神经网络（PhysicsScaleNetSolid）从 $5$ 个力学不变量回归；
4. 该网络以 **Germano 双滤波耗散一致性** 为自监督目标（湍流大涡模拟思想的固体类比），配合弹性/断裂锚定、本构先验与空间平滑，构成五损失在线训练；
5. 全文给出从强形式到离散代数、再到代码符号的**一致映射**（附实现函数索引与参数表），并记录稳定化措施的数学作用与验证结果。

> **范围与诚实声明**：当前求解器中，线弹性阶段使用**割线降级刚度**（非完整一致性切线）；"解析项+神经梯度项"的一致性切线 $\mathbf C^{ep}$ 分解（Term A/B）已写入理论文档，是无矩阵混合求解的目标接口，尚未在现行线性解中启用（见 §7.5）。非局部正则化、损伤率阻尼、尺度修正均已在代码中实现。

---

## 1. 记号与约定 (Notation)

| 记号 | 含义 |
| --- | --- |
| $\Omega\subset\mathbb R^2$ | 计算域（含圆盘与其外围"空"结构单元），边界 $\partial\Omega=\Gamma_u\cup\Gamma_t$ |
| $\boldsymbol u,\ \boldsymbol\varepsilon,\ \boldsymbol\sigma$ | 位移场、应变张量、应力张量 |
| $\mathbf C$ | 平面应变弹性刚度（Voigt 3×3），材料 $(E,\nu)$ |
| $D\in[0,1)$ | 各向同性标量损伤变量（单元常值离散） |
| $\varepsilon_{\mathrm{eq}}$ | Mazars 等效拉应变 |
| $\varepsilon_0,\ \beta$ | 损伤起始应变、软化指数 |
| $\mathcal G_f,\ K_{Ic},\ l_c$ | 断裂能、I 型断裂韧性、特征/单元长度 |
| $d(\boldsymbol x)\le -1/2$ | 神经网络的局部尺度指数 |
| $\lambda, \Lambda$ | 尺度比 $l/L_0$（修正）与 Germano 滤波比 $=3$ |
| $\eta,\ \bar\theta,\ \sigma_{\mathrm{eq}}$ | 应力三轴度、Lode 角参数、Mises 等效应力 |
| $\phi, \bar\phi$ | 单格耗散信号与测试滤波信号 |
| $h=\Delta x=\Delta y$ | 网格尺寸 |
| $K_{\mathrm{pen}}$ | 罚边界刚度 $=10^{10}E$ |

张量不变量约定：Voigt 剪切分量为工程剪应变 $\gamma_{xy}$，张量剪应变 $\varepsilon_{xy}=\gamma_{xy}/2$；平面应变面外应力 $\sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy})$。

---

## 2. 引言与问题设定

### 2.1 物理问题：巴西圆盘劈裂（ISRM）

圆盘半径 $R$、厚度 $t$（平面应变理想化），沿竖直直径由宽度为 $2w$ 的平台压条压缩。理想线弹性解在圆盘中心产生均匀拉应力 $\sigma_{\max}=2P/(\pi D t)$（$D=2R$），从而在中心起裂、沿加载直径劈裂。实际试验在材料破坏前以位移控制加载，结构响应呈现**峰值荷载后的软化段**；存在斜置预制裂纹（倾角 $\beta$）时为 **I–II 混合型**断裂。

### 2.2 数学建模的三层结构

| 层 | 内容 | 章节 |
| --- | --- | --- |
| 连续 | 动量守恒 + 损伤本构 + 边界条件（强/弱形式） | §3 |
| 离散 | 有限元空间离散 + 损伤场非局部正则 + 线性代数 | §4–5 |
| 闭包 | 尺度不变损伤率修正 + 神经算子自监督学习 | §6–8 |
| 工程 | 数值稳定化、BC/初值、代码实现与验证 | §9–11 |

### 2.3 出发点与设计动机

断裂模拟的两个核心困难：

1. **软化引起的病态与网格依赖**：应变软化使边值问题丧失椭圆性，解局部化带宽度由网格控制而非材料属性决定；
2. **裂尖近奇异耗散的跨尺度不可解析性**：裂纹过程带宽度远小于网格，网格尺度"看不见"真实耗散，需子网格模型。

本子系统的技术路线：损伤采用经典 Mazars 准则 + 断裂能正则；跨尺度耗散用**幂律标度（尺度指数 $d$）闭合**；标度 $d$ 由神经网络在线自监督学习，从而在**无需真值标签**的前提下，用同一框架完成损伤演化与"细观过程带效应"的建模。

---

## 3. 连续介质数学模型

### 3.1 平衡方程与边界条件（强形式）

准静态、无体力：

$$\nabla\cdot\boldsymbol\sigma=\mathbf 0\quad\text{in }\Omega,\qquad
\boldsymbol\sigma\,\mathbf n=\bar{\boldsymbol t}\ \text{on }\Gamma_t,\qquad
\boldsymbol u=\bar{\boldsymbol u}\ \text{on }\Gamma_u.$$

位移控制加载下 $\bar{\boldsymbol u}$ 为给定加载历程（下压位移 $\bar u(t)$）。

### 3.2 弱形式与有限元空间

试验函数 $\delta\boldsymbol u\in H^1_0(\Omega;\Gamma_u)^2$，虚功方程：

$$\int_\Omega \boldsymbol\sigma(\boldsymbol\varepsilon):\delta\boldsymbol\varepsilon\,d\Omega = \int_{\Gamma_t}\bar{\boldsymbol t}\cdot\delta\boldsymbol u\,d\Gamma.$$

取有限元子空间 $\boldsymbol u^h=\sum_i N_i(\boldsymbol x)\mathbf u_i$，得离散平衡：

$$\mathbf F_{\mathrm{int}}(\mathbf u,D)=\mathbf F_{\mathrm{ext}},\qquad
\mathbf F_{\mathrm{int}}=\int_\Omega \mathbf B^{\top}\boldsymbol\sigma\,d\Omega .$$

### 3.3 各向同性标量损伤本构

自由能

$$\psi(\boldsymbol\varepsilon,D)=(1-D)\,\tfrac12\boldsymbol\varepsilon:\mathbf C:\boldsymbol\varepsilon,$$

故 $\boldsymbol\sigma=\partial\psi/\partial\boldsymbol\varepsilon=(1-D)\,\mathbf C:\boldsymbol\varepsilon$，损伤驱动力（能量释放率）$Y=-\partial\psi/\partial D=\tfrac12\boldsymbol\varepsilon:\mathbf C:\boldsymbol\varepsilon$。损伤不可逆（$D$ 单调不减）。本实现中损伤"率"由**目标值驱动**：$D(\varepsilon_{\mathrm{eq}})$ 单调，仅当目标超过当前值才更新，天然满足不可逆。

### 3.4 Mazars 等效拉应变与演化

主应变 $\varepsilon_1\ge\varepsilon_2$：

$$\varepsilon_1,\varepsilon_2=\tfrac12(\varepsilon_{xx}+\varepsilon_{yy})\pm\sqrt{\big(\tfrac12(\varepsilon_{xx}-\varepsilon_{yy})\big)^2+\varepsilon_{xy}^2}.$$

**Mazars 等效拉应变**（Macaulay 括号 $\langle x\rangle_+=\max(x,0)$）：

$$\varepsilon_{\mathrm{eq}}=\sqrt{\langle\varepsilon_1\rangle_+^2+\langle\varepsilon_2\rangle_+^2}.$$

该量在单轴拉伸退化为 $\varepsilon_{xx}$，纯压缩退化为 $0$（压缩不产生 Mazars 损伤）。

**演化定律**（指数软化，目标曲线）：

$$D_{\mathrm{tar}}(\varepsilon_{\mathrm{eq}})=\begin{cases}
0, & \varepsilon_{\mathrm{eq}}\le\varepsilon_0,\\[2pt]
1-\dfrac{\varepsilon_0}{\varepsilon_{\mathrm{eq}}}\,e^{-\beta(\varepsilon_{\mathrm{eq}}-\varepsilon_0)}, & \varepsilon_{\mathrm{eq}}>\varepsilon_0.
\end{cases}$$

**一致性/不可逆（离散增量）**：由当前步应力场求出 $\varepsilon_{\mathrm{eq}}$，计算目标 $D_{\mathrm{tar}}$，则损伤驱动力增量

$$\Delta D_{\mathrm{base}}=\max\big(0,\;D_{\mathrm{tar}}-D^{\mathrm{old}}\big),$$

再乘以阻尼与（耦合期）尺度因子后累加 $D^{\mathrm{new}}=\min(0.99999,\,D^{\mathrm{old}}+\Delta D)$。这等价于在每个加载步做一次"预测-投影"的显式返回映射。

### 3.5 断裂能标定：$(\varepsilon_0,\beta)$ 的物理确定

起裂应变取抗拉强度割线：

$$\varepsilon_0=\frac{\sigma_t}{E}.$$

软化参数由能量一致确定。平面应变下按断裂韧性换算断裂能（I 型、单位厚度、单位面积）：

$$\mathcal G_f=\frac{K_{Ic}^2(1-\nu^2)}{E}.$$

令"峰值前弹性储能 + 软化尾段耗能"等于每单元断裂能密度 $\mathcal G_f/l_c$（$l_c$ 网格特征长度）。对指数软化形 $\sigma=E\varepsilon_0 e^{-\beta(\varepsilon-\varepsilon_0)}$ 积分 $\int_{\varepsilon_0}^{\infty}\sigma\,d\varepsilon=\frac{\sigma_t^2}{2E}+\frac{\sigma_t}{\beta}$，于是

$$\boxed{\;\beta=\frac{\sigma_t}{\mathcal G_f/l_c-\sigma_t^2/(2E)}\;}$$

实现：矩形规则网格 $l_c=h$；通用非结构网格 $l_c=\sqrt{\overline{A_e}}$（均值单元面积的开方）。

### 3.6 软化病态与非局部正则

局部软化损伤导致椭圆性丧失与网格依赖；本子系统采用**应变驱动场的一阶非局部平均（积分型正则化）**：

$$\bar\varepsilon_{\mathrm{eq}}(\boldsymbol x)=\frac{1}{|\mathcal N_r(\boldsymbol x)|}\int_{\mathcal N_r(\boldsymbol x)}\varepsilon_{\mathrm{eq}}(\boldsymbol y)\,d\boldsymbol y,$$

其中 $\mathcal N_r$ 为半径 $r$ 的邻域：
- 规则网格实现为 $3\times3$ 盒式 `uniform_filter`（半径 1 单元，`mode='constant'`）；
- 通用网格实现为 `cKDTree.query_ball_point`（半径 $r=2.5\,l_c$）算术平均。

以 $\bar\varepsilon_{\mathrm{eq}}$ 代替 $\varepsilon_{\mathrm{eq}}$ 进入 $D_{\mathrm{tar}}$，从而局部化带宽度由 $r$（材料型内禀长度）控制，弱化网格依赖。同理，Germano 测试滤波信号也采用该盒式滤波。

---

## 4. 有限元离散化

### 4.1 网格、自由度与域掩膜

- 规则网格：$N_x\times N_y$ 节点，方形域 $[0,L]^2$，单元数 $(N_x-1)(N_y-1)$；节点编号行主序，单元 $e=(j,i)$ 的四节点 $n_0=jN_x+i,\ n_1=n_0+1,\ n_2=n_0+N_x+1,\ n_3=n_0+N_x$；
- **几何掩膜（活性单元集）**：单元中心 $(x_c,y_c)$ 满足
$$\big(x_c-X_c\big)^2+\big(y_c-Y_c\big)^2\le R^2\quad\text{且}\quad |y_c-Y_c|\le R-f_{\mathrm{flat}}$$
才计入活性集（$f_{\mathrm{flat}}$ 为平台截断高度），其余为非活性；
- 自由度 $N_{\mathrm{dof}}=2N_xN_y$，活性单元自由度表 `elem_dof_array`（每单元 8 个 DOF）预计算；活性 DOF 掩码 $\mathcal A_{\mathrm{dof}}$。

### 4.2 Q4 等参元与积分

母单元 $(\xi,\eta)\in[-1,1]^2$，形函数 $N_i$ 双线性；物理梯度经 Jacobian $\mathbf J$ 变换：

$$\begin{bmatrix}\partial_x\\\partial_y\end{bmatrix}N_i=\mathbf J^{-1}\begin{bmatrix}\partial_\xi\\\partial_\eta\end{bmatrix}N_i,\qquad \det\mathbf J>0\ \text{需检验}.$$

应变-位移矩阵 $\mathbf B$（3×8）：$\boldsymbol\varepsilon^e=\mathbf B\mathbf u^e$，剪切行为工程约定。

单元刚度（$2\times2$ 高斯，Gauss 点 $\pm1/\sqrt3$）：

$$\mathbf k_e=\int_{\Omega_e}\mathbf B^{\top}\mathbf C\mathbf B\,d\Omega\simeq\sum_{p,q=1}^{2}w_pw_q\,\mathbf B^{\top}(\xi_p,\eta_q)\,\mathbf C\,\mathbf B(\xi_p,\eta_q)|\det\mathbf J|.$$

**关键模板化技巧**：把 $E$ 从几何解耦。矩形网格预先计算"单位模量"刚度模板

$$\mathbf k_0^{\mathrm{unit}}=\mathbf k_e\big|_{C=C(E=1,\nu)},$$

则损伤降级后的单元刚度仅是标量重放

$$\mathbf k_e(D)=(1-D_e+\epsilon_{\mathrm{res}})\,E\;\mathbf k_0^{\mathrm{unit}},$$

从而整个全局矩阵可写成"稀疏索引预计算 + 每步向量放缩"的形式（COO 缓存，$O(N)$ 存储）。

### 4.3 损伤降级全局系统

$$\mathbf K(D)=\sum_{e\in\mathcal A}A_e^{\top}\,\mathbf k_e(D)\,A_e+\mathbf K_{\mathrm{reg}}+\mathbf K_{\mathrm{pen}},$$

- $A_e$ 组装映射；$\mathbf K_{\mathrm{reg}}$ 为**非活性 DOF 对角保护**（$K_{ii}=1$），防止几何外单元污染、保证可逆；
- $\mathbf K_{\mathrm{pen}}$ 为罚边界（见 §5.1）；
- 残余刚度 $\epsilon_{\mathrm{res}}=10^{-6}$ 与饱和 $D\le 0.99999$ 联合保证 $\mathbf K(D)\succ0$ 一致正定（详见 §9.1）。

### 4.4 应变/应力恢复（单元中心）

$${\boldsymbol\varepsilon}_e=\mathbf B_c\mathbf u_e,\qquad
{\boldsymbol\sigma}_e=(1-D_e)\,\mathbf C\,{\boldsymbol\varepsilon}_e,$$

$\mathbf B_c$ 为单元中心 $(\xi,\eta)=(0,0)$ 的 $\mathbf B$。Von Mises 等效应力（平面应变）：

$$\sigma_{\mathrm{eq}}=\sqrt{\tfrac12\big[(\sigma_{xx}-\sigma_{yy})^2+(\sigma_{yy}-\sigma_{zz})^2+(\sigma_{zz}-\sigma_{xx})^2\big]+3\sigma_{xy}^2},\quad \sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy}).$$

---

## 5. 边界条件、加载与线性求解

### 5.1 罚边界与反力

位移控制下，顶部/底部平台节点沿 $y$ 约束（顶部给 $u_y=-\bar u$，底部 $u_y=0$，$u_x$ 自由），圆盘上下最中间接触节点再约束 $u_x=0$ 以消除刚体水平位移。罚处理：对受约束 DOF $i$，$K_{ii}\leftarrow K_{ii}+K_{\mathrm{pen}}$，右端 $F_i\leftarrow F_i+K_{\mathrm{pen}}\bar u_i$。反力为罚反力之和：

$$F_{\mathrm{react}}=\sum_{i\in\text{top}}\big|K_{\mathrm{pen}}(\bar u_i-u_i)\big|.$$

### 5.2 每步的弹性边值问题（割线格式）

在固定损伤场 $D^{(n)}$ 下求解

$$\mathbf K(D^{(n)})\,\mathbf U^{(n+1)}=\mathbf F,$$

这是一个**线性**（但各步变化的）系统；本子系统实际采用**显式交错（staggered）**：解弹性场 → 恢复应变/应力 → 求 $\Delta D$ → 更新 $D$，再进入下一步。不进行 Newton–Raphson 隐式平衡迭代。

### 5.3 稀疏直接法（默认）

每步将 COO 三元组重放数值 `scale=(1−D+ε_res)E` 与 `k0_tile` 逐元素相乘，`csr_matrix` 组装后 `scipy.sparse.linalg.spsolve`（KLU/UMFPACK 型稀疏直接分解）。活性 DOF 数目 $N_{\mathrm{act}}\approx 5000$（80×80 圆盘）量级时效率足够。

### 5.4 无矩阵 PCG（替代求解器，§4 与共享库）

内存优化路线：**不组装 $\mathbf K$**，仅提供矩阵-向量积

$$(\mathbf K\cdot \mathbf u)=\sum_{e\in\mathcal A} A_e^{\top}\Big[\big((1-D_e+\epsilon_{\mathrm{res}})E\big)\,\mathbf k_0^{\mathrm{unit}}\mathbf u_e\Big].$$

共轭梯度求解时（`src/pcg.py`）：
- **精确消元边界**：把受约束/非活性 DOF 从自由集剔除，解缩聚系统 $\mathbf K_{ff}\mathbf u_f=\mathbf F_f-\mathbf K_{fb}\mathbf u_b$（避免罚数 $\sim10^{10}$ 造成的病态）；
- **Jacobi 对角预条件** $M_{ii}=\sum_j|\mathbf k_{ij}|$（行绝对值和），下限保护 $10^{-12}$；
- 相对残差停机 $\|\mathbf r\|/\|\mathbf r_0\|<10^{-8}$；
- 反力由罚表达式在收敛解上求。

一致性验证（`examples/run_pcg_demo.py`，80×80，$D=0$）：PCG 解与 `spsolve` 解相对误差 $<10^{-6}$；`apply` 与显式组装矩阵的算子自洽误差 $<10^{-10}$。即**矩阵自由算子与原稀疏组装逐位一致**，为大规模网格（150×150+）预留路径。

> 注意：本节 PCG 仅含"物理项"（损伤降级弹性，Term A）；完整"神经梯度项"（Term B）进入切线算子的矩阵自由混合求解为设计目标（见 §7.5）。

---

## 6. 尺度不变算子代数：子网格闭合

### 6.1 动机（物理尺度论证）

设真实断裂过程带宽度 $\ell$ 远小于网格尺度 $h$（$\lambda=\ell/L_0$，代码取 $L_0/h$ 对应比 $0.3$）。由 Irwin 近裂尖渐近，位移奇异为 $O(\sqrt r)$（$r$ 到裂尖距离），相应应变奇异为 $O(r^{-1/2})$，能量密度奇异为 $O(r^{-1})$。若以幂律 $x^{\alpha}$ 表征跨尺度量的变化，则**能量型量的尺度指数为 $-1$、广义"场型"量的尺度指数为 $-1/2$**。这给出弹性锚点 $d=-1/2$ 的物理依据，也是整个尺度不变框架的"标定"起点。

### 6.2 幂律修正假设

对每一单元，真实（子网格）损伤率与网格可解析 Mazars 率之间假定满足

$$\boxed{\;\Delta D_{\mathrm{corr}}(\boldsymbol x)=\Big(\frac{l}{L_0}\Big)^{\,d(\boldsymbol x)}\Delta D_{\mathrm{base}}(\boldsymbol x)\;}$$

- $d(\boldsymbol x)\le-1/2$：负指数 + 尺度比 $<1$ ⇒ 修正系数 $\ge 1$，即在**损伤集中区放大网格演化率**以补偿子网格过程带的真实耗散；$|d|$ 越大放大越强（裂尖、断裂带）；
- 完好区 $d\to-1/2$ 附近，退化为一固定几何放大，避免把"弹性锚"本身学偏；
- 工程约束 $0.1\le (l/L_0)^d\le10$ 限制单步修正幅度（显式格式稳定）。

### 6.3 与湍流 Germano 恒等式的类比（自监督可行性）

LES 中，大涡方程与滤波子网格应力由 **Germano 恒等式** $\mathcal L=\widehat{\overline{T}}-\overline{\hat{T}\hat T}$ 在两层滤波之间建立可测约束。本子系统将"动量方程残差"类比为"耗散信号的跨尺度守恒"：若 $d(\boldsymbol x)$ 是正确闭包，则**在网格尺度测量并在测试（2× 网格）尺度滤波的耗散信号**，应与按 $d$ 放大后跨尺度的信号一致。由此得到**无需外部位移场真值**的自监督损失（§8.1.1）。

### 6.4 耗散信号构造（与代码逐行对应）

1. 单元内功密度（张量约定）：
$$W=\tfrac12\big(\sigma_{xx}\varepsilon_{xx}+\sigma_{yy}\varepsilon_{yy}+2\sigma_{xy}\varepsilon_{xy}\big);$$
2. 规整化驱动力（除去已损伤部分）：$Y=W\big/\big((1-D)^2+\epsilon\big)$；
3. **单格耗散信号**：$\phi=Y\,\Delta D_{\mathrm{base}}$；
4. **测试滤波**（$3\times3$ 盒式，质量守恒 cval=0）：$\bar\phi=\mathcal F[\phi]$（`uniform_filter` / `convolve(ones(3)/9)`）；
5. 局部 Hess 比与权重（诊断用）：$H=\bar\phi/\phi$（$\phi>10^{-15}$ 处），$w=\phi$。

### 6.5 尺度不变自洽目标

$$d(\boldsymbol x)\ \text{应使}\quad \Lambda^{d(\boldsymbol x)}\phi(\boldsymbol x)\approx \bar\phi(\boldsymbol x),\qquad \Lambda=3$$

$\Lambda=3$ 为测试滤波与网格滤波的尺度比（代码 `lambda_L=3.0`）。该式给出 §8 中损失项 $\mathcal L_G$。

---

## 7. PhysicsScaleNetSolid：特征、结构与输出约束

### 7.1 特征向量：$5$ 个力学不变量

对每个活性单元，构造（全部经 $\tanh$ 压缩到有界区间）：

$$\mathbf f=\Big[\;D,\quad \tanh\eta,\quad \tanh\bar\theta,\quad \tanh\Big(\tfrac{\varepsilon_{\mathrm{eq}}}{\varepsilon_0}-1\Big),\quad \tanh\big(l_c\,|\nabla D|\big)\;\Big].$$

应力不变量（含面外 $\sigma_{zz}=\nu(\sigma_{xx}+\sigma_{yy})$）：

- 静水应力 $\sigma_m=(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})/3$；
- 偏量 $S_{ij}=\sigma_{ij}-\sigma_m\delta_{ij}$，$J_2=\tfrac12 S_{ij}S_{ij}$，$J_3=\det S$；
- **Mises 等效应力** $\sigma_{\mathrm{eq}}=\sqrt{3J_2}$；
- **应力三轴度** $\eta=\sigma_m/\sigma_{\mathrm{eq}}$；
- **Lode 角参数** $\bar\theta=1-\tfrac{2}{\pi}\arccos\big(\tfrac{27}{2}\tfrac{J_3}{\sigma_{\mathrm{eq}}^3}\big)$（注意与标准 $\cos3\theta$ 关系 $\tfrac{27}{2}J_3/\sigma_{\mathrm{eq}}^3=\tfrac{3\sqrt3}{2}J_3/J_2^{3/2}$，自变量被 `clip` 到 $[-1,1]$）。

损伤梯度范数：规则网格用 `np.gradient(D_grid, dy, dx)`；通用网格用 kNN 最大差商 $\max_{j}|D_j-D_i|/\|x_j-x_i\|$。`tanh` 使特征落于 $(-1,1)$，匹配网络激活范围。

### 7.2 网络架构与输出约束

```
f ∈ R^5
  → Linear(5→32) + LayerNorm(32) + Tanh
  → Linear(32→32) + Tanh
  → Linear(32→1)，bias 初值 −5.0
  → d = −1/2 − softplus(·)
```

- `softplus` 保证 $d\le-1/2$（结构约束，不需惩罚）；
- 末层偏置初始为 $-5$ ⇒ $d_0\approx-0.5067$：网络热启动即"弹性锚点"，退化系统无破坏性瞬态；
- `LayerNorm` 抑制批间分布漂移（特征来自不断演化的应力场）。

### 7.3 在线训练组织

两阶段调度：**warmup**（前 $n_{\mathrm{warmup}}$ 步）纯 Mazars，网络不激活——此时损伤尚未起裂，学习无意义且易发散；**coupled**（后 $n_{\mathrm{coupled}}$ 步）每步：
1. 正向：特征 $\to d=\mathrm{net}(\mathbf f)$；
2. 计算 $\phi,\bar\phi$；
3. 组装五损失 $\mathcal L$，反传，`clip_grad_norm_(1.0)`，Adam($\mathrm{lr}=2\times10^{-3}$)；
4. 用更新后网络重新推理 $\bar d$，完成标度修正损伤更新。

仅当损失有限（`torch.isfinite`）才执行优化步；配合 §9 的指数截断，从机制上规避 `nan/overflow`。

### 7.4 五项混合损失函数

记 $\lambda_L=3$，$D_t$ 为单元损伤张量，$d_\theta$ 网络输出。

**(1) Germano 自监督主项**（归一化使尺度无关）：

$$\mathcal L_G=\frac{\sum_e\big(\lambda_L^{d_\theta(e)}\phi_e-\bar\phi_e\big)^2}{\sum_e \phi_e^2+\epsilon},$$

**(2) 弹性锚定**（完好单元 $D<0.01$ 锚 $d=-1/2$）：

$$\mathcal L_E=\frac{1}{|\mathcal E_0|}\sum_{e:D_e<0.01}\big(d_\theta(e)+\tfrac12\big)^2,$$

**(3) 断裂锚定**（近断裂单元 $D>0.9$ 惩罚 $d$ 不够负；$e^{2d}\to0$ 当 $d\to-\infty$）：

$$\mathcal L_F=\frac{1}{|\mathcal E_1|}\sum_{e:D_e>0.9}\exp(2\,d_\theta(e)),$$

**(4) 本构先验**（弱锚到连续损伤软化解析关系，$c_1=0.3$）：

$$f_{\mathrm{prior}}=-0.5+c_1\ln(1-D),\qquad \mathcal L_D=\frac1{N}\sum_e\big(d_\theta(e)-f_{\mathrm{prior}}(D_e)\big)^2,$$

**(5) 空间平滑**（网格差分离散梯度，$l_d$ 平滑长度）：

$$\mathcal L_S=l_d^2\left(\tfrac1{N_xN_y}\sum\big(\partial_x^h d\big)^2+\sum\big(\partial_y^h d\big)^2\right),$$

总损失（权重向量 $\boldsymbol\lambda=(\lambda_g,\lambda_e,\lambda_f,\lambda_d,\lambda_s)$）：

$$\mathcal L=\lambda_g\mathcal L_G+\lambda_e\mathcal L_E+\lambda_f\mathcal L_F+\lambda_d\mathcal L_D+\lambda_s\mathcal L_S.$$

各项物理作用汇总：

| 项 | 机理 | 防止的错误 |
| --- | --- | --- |
| Germano | 双尺度耗散守恒（唯一"真值"信号） | 闭包失真 |
| 弹性锚定 | 完好区退化为弹性基准 | 远场虚假损伤放大 |
| 断裂锚定 | 断裂区 $d\to-\infty$ | 裂尖放大不足 |
| 本构先验 | 弱正则到解析软化 | 发散/跳变 |
| 平滑 | $\|\nabla d\|^2$ 惩罚 | 逐单元噪声场 |

### 7.5 与一致性切线框架的关系（理论—实现）

理论文档把可微损伤本构的一致切线写作

$$\mathbf C^{ep}=\underbrace{(1-D)C}_{\text{Term A}}-\underbrace{C:\varepsilon\otimes\frac{\partial D}{\partial \varepsilon}}_{\text{Term B}},$$

其中 Term B 需在单元局部以自动微分计算 $\partial D/\partial\varepsilon$。这是 **“解析项预计算 + 局部 AD 图、不追踪全局图”** 的策略，对应 $O(N\times T)\to O(N)$ 显存降维的理论依据。当前实现细节：
- 求解器对弹性解使用**割线** $\mathbf K(D)$（即仅 Term A 的零梯度版本），Term B 目前经"损伤率标度因子"间接发挥作用，而未作为切线进入 Jacobian；
- 矩阵自由 PCG（§5.4）已实现 Term A 的算子乘法；把 Term B 与 `torch.vmap` 单元并行加入为后续主线。

---

## 8. 学习算法：两阶段在线自监督

### 8.1 形式化

给定加载历程 $\{\bar u^{(n)}\}$，每步执行：

```
for n = 0..N_total-1:
    # --- A. 弹性/应变恢复（固定 D） ---
    U ← solve  K(D) U = F(ū_n)
    ε_e = B_c U_e ; σ_e = (1−D_e) C ε_e
    # --- B. 非局部正则 + Mazars 目标 ---
    ε_eq ← nonlocal_average(ε_eq(ε_e))
    D_tar = mazars(ε_eq; ε0, β)
    ΔD_base = damping(phase) · max(0, D_tar − D)
    # --- C. 更新损伤 ---
    if n < n_warmup:      D ← clip(D + ΔD_base, 0, 0.99999)          # Phase 1
    else:                                                             # Phase 2
        f = features(σ, ε, D, ∇D)
        d = net(f); L = Σ λ·loss(d, φ, φ̄); backward; Adam step
        D ← clip(D + clip(λ_L^d̄, 0.1, 10) · ΔD_base, 0, 0.99999)
    record history; visualize(stride)
```

### 8.2 数值稳定化措施（数学作用）

| 措施 | 表达式 | 作用 |
| --- | --- | --- |
| 残余刚度 | $D\leftarrow\min(D,\,1-\epsilon_{\mathrm{res}})$；$k_e\leftarrow(D_\epsilon+\epsilon_{\mathrm{res}})E\,k_0$ | $\mathbf K\succ0$，条件数 $\lesssim E/\epsilon_{\mathrm{res}}E=10^6\times O(1)$ 可解；消除"完全断裂单元零刚度"奇点 |
| 指数截断 | $\mathrm{clip}(\beta(\varepsilon_{\mathrm{eq}}-\varepsilon_0),\,0,\,50)$ | $\exp$ 参数 $\le50$，杜绝 overflow |
| 等效应变上限 | $\mathrm{clip}(\varepsilon_{\mathrm{eq}},\,0,\,\varepsilon_0\times200)$ | 防单点应变尖峰放大指数项 |
| 演化阻尼 | warmup $\alpha=0.3$；coupled $\alpha=0.7$（$\Delta D>0.1$ 处）或 $0.5$ | 限制单步损伤增量；低阻尼抑制"级联扩散"，高阻尼在裂尖加速 |
| 损伤饱和 | $D\le0.99999$ | 保正定 + 可视化断裂带 |
| 尺度修正限幅 | $0.1\le\lambda^{d}\le10$ | 显式格式单步稳定 |
| 非局部正则 | 半径 $r$ 平均 | 局部化带宽与网格解耦 |

### 8.3 收敛/正则性说明（非严格证明）

1. 每步 $\mathbf K(D)$ 对称正定（$\epsilon_{\mathrm{res}}>0$、对角保护），故位移解存在唯一且 CG/直接法均可收敛；
2. $D_{\mathrm{tar}}$ 为 $\varepsilon_{\mathrm{eq}}$ 的单调、值域 $[0,1)$ 函数，经 $\Delta D=\max(0,\cdot)$ 与饱和后 $D$ 单调不减、有界，保证每个单元损伤轨迹有界，不会出现负刚度穿越；
3. 指数截断使 $D_{\mathrm{tar}}$ 计算在浮点上总是有限，配合 `isfinite(loss)` 门控，从机制上杜绝 `nan/overflow` 传播；
4. 损失各项均对 $d_\theta$ 连续可微；平滑项对 $d$ 网格差分，故 $d$ 场趋于光滑，正则化意义上限制解的振荡模。

---

## 9. 初始条件与加载历程的实现细节

### 9.1 几何与网格

| 版本 | 网格 | 域/圆盘 | 活性单元 | $f_{\mathrm{flat}}$ |
| --- | --- | --- | --- | --- |
| splitting（早期） | 100×100 | 2 m 域 / R=0.8 m | ≈6300 | 无 |
| v1.0（brazilian_disc_v1） | 80×80 | 60 mm / R=25 mm | ≈5000 | 0.4 mm |
| v2.0（crack.py） | 80×80 | 60 mm / R=25 mm | ≈5000 | 0.4 mm |

（v2.0 较 v1.0 增加：非局部 `ε_eq`、自适应阻尼、裂尖过渡带损伤、完整五损失与日志化。）

### 9.2 预制裂纹嵌入

对倾角 $\beta$、半长 $a$ 的裂纹，将单元中心旋入裂纹局部系 $(x',y')$：

$$x'=(x_c-X_c)\cos\beta-(y_c-Y_c)\sin\beta,\qquad
y'=(x_c-X_c)\sin\beta+(y_c-Y_c)\cos\beta.$$

- **核心带**：$|x'|<0.6\,h$ 且 $|y'|<a$ ⇒ $D=0.999$；
- **过渡带（v2.0）**：$|y'|<a$ 且 $|x'|<2h$ ⇒ $D=0.999\,e^{-3\,\frac{|x'|-0.6h}{2h-0.6h}}$，几何上避免裂纹面与损伤前沿的数值跳变；
- 通用 FEA 版本以 `Wall` 节点集为墙段，对墙段邻接单元置 $D=0.999$，并按到墙段距离构造指数过渡带（带宽 $2l_c$）。

### 9.3 位移控制加载与历史输出

$\bar u_n=(n+1)\Delta u_{\mathrm{step}}$（v1/v2 每步 $3\,\mu\mathrm m$；早期 $6\,\mu\mathrm m$）。反力-位移历程用于判定峰值与软化；$D_{\max}$、开裂单元计数（$D>0.99$）、平均标度 $\langle d\rangle$ 与五损失项逐耦合步记录。

---

## 10. 通用化：DAT 驱动 FEA 求解器与共享库

为把巴西圆盘专用求解器推广到任意 Q4 网格/多材料，断裂线实现了 `fracture/FEA/`：

### 10.1 DAT 前处理解析（`dat_parser.py`）

分节关键字 `coordinates` / `Element` / `Moment-Load`（约束） / `Presure`（顶部位移） / `Wall`（裂纹节点） / `MATERIAL PROPERTIES`。解析校验头部计数、单元引用材料存在性，输出 `DatModel`（节点、单元、材料、约束、预置位移、裂纹节点）。

### 10.2 通用求解器增量（`solver.py::DatCrackSolver`）

- **多材料**：每单元按其材料 $\nu$ 预计算 $\mathbf k_0^{\mathrm{unit}}$ 与 $\mathbf B_c$，放缩因子用该材料 $E$：$\mathrm{scale}_e=(1-D_e+\epsilon_{\mathrm{res}})E_{\mathrm{mat}(e)}$；
- **非局部**：质心 `cKDTree` 邻域平均（$r=2.5\,l_c$）；$\phi$ 测试滤波半径 $3\,l_c$；
- **梯度特征**：kNN 最大差商；
- **Wall/裂纹初始损伤与过渡带**；
- **自动 UX 锚点**：DAT 无水平约束时自动固定底部最近中心节点 $u_x$；
- **退化运行**：无 torch → 解析标度场 $d=-0.5+0.3\ln(1-D)$；无 matplotlib → 仅 CSV；
- **GiD 后处理**：`.msh`/`.res`（位移/应力/应变/损伤/Von Mises/标度场/反力历史）。

### 10.3 共享库 `src/`

`fem_utils.py`（$C$、$\mathbf B$、模板、不变量）、`damage_models.py`（Mazars、阻尼、断裂能标定）、`networks.py`（网络+五损失+Germano）、`pcg.py`（矩阵自由算子+PCG）、`config.py`（`SolverConfig` 参数集中）、`solver.py`（`BaseCrackSolver` 统一两阶段循环骨架）为规则网格与通用网格共用同一套算法；`src` 也服务模态线，故保留于仓库根。

---

## 11. 验证与一致性检验

### 11.1 单元级与代数级测试（`tests/`，共 16 项全部通过）

| 类别 | 断言（数学性质） | 状态 |
| --- | --- | --- |
| $C$ 矩阵 | 对称、各向同性、单轴应变解析 $\sigma_{xx}=E(1-\nu)\varepsilon_{xx}/((1+\nu)(1-2\nu))$ | PASS |
| 等效拉应变 | 单轴受拉 $\varepsilon_{\mathrm{eq}}=\varepsilon_{xx}$；全受压 $\varepsilon_{\mathrm{eq}}=0$ | PASS |
| Mazars | $D=0$ for $\varepsilon\le\varepsilon_0$；$D\in(0,1)$ for $\varepsilon>\varepsilon_0$；关于 $\varepsilon$ 单调 | PASS |
| 断裂能标定 | $\varepsilon_0=\sigma_t/E$；$\beta>0$ 有限 | PASS |
| PCG | 与稠密直接解相对误差 $<10^{-6}$；算子形状/收敛 | PASS |
| 损伤-应力 | Von Mises 单轴退化 $\sigma_{\mathrm{vm}}=|\sigma_{xx}|$（$\nu=0$） | PASS |

### 11.2 求解器级一致性（`examples/run_pcg_demo.py`）

- PCG 缩聚解 vs `spsolve`：相对误差 $<10^{-6}$（两求解器互证）；
- 矩阵自由 `apply` vs 显式组装 $K\cdot u$：$<10^{-10}$（算子自洽）。

### 11.3 端到端物理合理性（可复现输出）

`fracture/crack.py` / `fracture/brazilian_disc_v1.py` 端到端运行产出的物理轨迹（见 `fracture/snapshots*/`、`fracture/brazilian_disc_result.png`）：
- 损伤沿预制裂纹（$\beta=45°/90°$）起裂并扩展，卸载区 $D$ 保持不可逆饱和；
- 荷载-位移曲线经历近线性加载段 → 峰值 → 软化下降（峰值由可视化自动标注）；
- 耦合期 $\langle d\rangle$ 与五损失随时间单调收敛；$d$ 场在裂尖区域明显低于弹性锚点 $-1/2$（体现放大）。

> 说明：仓库未含与实验数据的定量校准报告；当前验证层级为"解析性质 + 求解器一致性 + 轨迹物理合理性"，定标/实验校核列为后续工作（见 §12）。

### 11.4 版本族谱与参数表

| 求解器文件（`fracture/`） | 定位 | 相对前代的数值增量 |
| --- | --- | --- |
| `brazilian_splitting_solver.py` | 早期原型（100×100） | Sigmoid 标度网络雏形、接触节点检测 |
| `brazilian_disc_v1.py` | v1.0（80×80） | 五面板可视化、软化学、$\mathcal L$ 五项 |
| `crack.py` | v2.0（80×80） | 非局部 $\varepsilon_{\mathrm{eq}}$、自适应阻尼、裂尖过渡带、完整损失 |
| `hybrid_cae_solver_v1.py` | 原型 | Matrix-free 逻辑框架（`apply_operator` 空） |
| `hybrid_cae_solver_final.py` | 原型 | 继承 v1，物理步+应力可视化 |
| `hybrid_cae_stable.py` | 稳定原型 | 残余刚度/截断/阻尼 $0.2$/饱和固化 |
| `run_brazilian_demo.py` / `real.py` | 演示/适配 | 局部 AD 核、网格可视化适配器 |

**默认材料与数值参数（v1/v2）**：

| 符号 | 值 | | 符号 | 值 |
| --- | --- | --- | --- | --- |
| $E$ | 30 GPa | | $K_{Ic}$ | 31.62 MPa·√mm |
| $\nu$ | 0.25 | | $\varepsilon_0=\sigma_t/E$ | $2\times10^{-4}$ |
| $\sigma_t$ | 6 MPa | | $a$（裂纹半长） | 5 mm（2a=10） |
| $w$（加载半宽） | 4.4 mm | | $\Delta u$ | 3 μm/step |
| $n_{\mathrm{warmup}}$ | 50–60 | | $n_{\mathrm{coupled}}$ | 200–500 |
| $\epsilon_{\mathrm{res}}$ | $10^{-6}$ | | $\alpha$（阻尼） | 0.3 / 0.5 / 0.7 |
| $l_c$ | $h$（规则） | | $\lambda$（尺度比） | 0.3；$\Lambda=3$ |
| $(\lambda_g,\lambda_e,\lambda_f,\lambda_d,\lambda_s)$ | $(0.3,0.5,0.3,0.2,0.1)$ | | $\mathrm{lr}$ | $2\times10^{-3}$ |

---

## 12. 局限与展望（Open Problems）

1. **一致性切线（Term B）未入线性解**：现行交错割线格式稳定且内存友好，但收敛阶为交错线性率；把 $C^{ep}$ 的神经梯度项接入矩阵自由切线 + `torch.vmap` 单元并行是既定 P0；
2. **标度闭包的物理解释仍需定标**：$\lambda^{d}$ 的幂次依赖数据自监督收敛性，尚缺与 DIC/试验的定量标定；
3. **网格无关性仅一阶非局部**：更细网格下建议推广各向异性/隐式梯度正则化；
4. **多裂纹、3D、动态裂纹分叉**未实现（roadmap Phase 6）;
5. **超参数敏感性**：阻尼 $0.3/0.5/0.7$、饱和阈值、$c_1=0.3$ 等为启发式选择，可作 UQ 主题。

---

## 附录 A：符号—代码映射（实现索引）

| 数学对象 | 实现位置 |
| --- | --- |
| $\mathbf C$（平面应变） | `src/fem_utils.py:plane_strain_C`；`fracture/crack.py:plane_strain_C` |
| $\mathbf B$、$\det\mathbf J$、$2\times2$ 高斯 | `src/fem_utils.py:q4_B_matrix/q4_unit_stiffness/rect_q4_stiffness_template` |
| $\mathbf k_0^{\mathrm{unit}}$ 模板化 | `crack.py:q4_stiffness_template`；`FEA/solver.py:_precompute_elements` |
| COO 组装/非活性 DOF 保护 | `crack.py:solve_elasticity`；`FEA/solver.py:_build_coo_cache` |
| 罚边界与反力 | `crack.py:solve_elasticity`；`src/solver.py:get_bc_rhs` |
| $\varepsilon_{\mathrm{eq}}$、主应变 | `src/fem_utils.py:mazars_equivalent_strain/compute_principal_strains` |
| Mazars $D_{\mathrm{tar}}$ | `src/damage_models.py:mazars_damage_target` |
| $\varepsilon_0,\beta$ | `src/damage_models.py:compute_damage_parameters` |
| 阻尼/饱和 | `src/damage_models.py:compute_damage_base` |
| 非局部平均 | `crack.py`（uniform_filter）；`FEA/solver.py:_nonlocal_average` |
| $(\eta,\bar\theta,\sigma_{\mathrm{eq}})$ | `src/fem_utils.py:stress_invariants`；`crack.py:compute_features` |
| 特征 $\mathbf f$ | `src/networks.py:compute_features`；`FEA/solver.py:compute_features` |
| $d=-\tfrac12-\mathrm{softplus}(\cdot)$ | `src/networks.py:PhysicsScaleNetSolid`；`crack.py` |
| $W,Y,\phi,\bar\phi$ | `src/networks.py:compute_germano_signal`；`crack.py:compute_germano_signal` |
| $\mathcal L$（五损失） | `src/networks.py:compute_loss`；`crack.py:compute_loss` |
| 尺度修正 $\lambda^d$ | `crack.py:update_damage`；`src/solver.py:update_damage` |
| Matrix-Free $K\cdot u$ + PCG | `src/pcg.py:MatrixFreeOperator/pcg_solve/_build_free_precond` |
| 通用 FEA/DAT/GiD | `fracture/FEA/dat_parser.py`、`fracture/FEA/solver.py` |
| 统一循环骨架 | `src/solver.py:BaseCrackSolver` |

## 附录 B：术语表

- **Mazars 等效拉应变**：以受拉主应变的 Macaulay 括号平方和开根定义的驱动标量；
- **割线刚度 / 显式交错**：每步以当前 $D$ 冻结刚度解弹性、再显式更新 $D$；
- **Term A/B**：一致切线 $C^{ep}$ 的解析弹性项与神经损伤梯度项；
- **Germano 恒等式**：LES 中滤波与亚格子应力之间的恒等式；此处作跨尺度耗散一致性的类比依据；
- **尺度指数 $d$**：刻画"细观过程带相对网格"耗散放大幅度的局部标度参数。

---

*文档状态：与 `fracture/` 当前代码（crack v2.0 系 + FEA + src）核对一致；共享模态线功能（`src/modal_solver.py` 等）不在本文范围。*

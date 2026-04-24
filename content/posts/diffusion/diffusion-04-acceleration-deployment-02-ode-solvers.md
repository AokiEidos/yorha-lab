---
title: "高阶 ODE 求解器"
date: 2026-04-21T20:57:40.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 高阶 ODE 求解器

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[SDE/ODE 统一视角](../01-fundamentals/07-sde-ode.md)，[DDIM 详解](../02-models-zoo/02-ddim.md)

## 从 DDIM 到高阶求解器 🔰

DDIM 的本质是扩散模型**概率流 ODE（Probability Flow ODE）** 的一阶 Euler 离散化。一阶方法的局限性在于：每一步的截断误差为 $O(\Delta t^2)$，全局误差为 $O(\Delta t)$。当步数从 1000 减少到 20 时，步长 $\Delta t$ 增大 50 倍，Euler 方法的误差急剧放大。

**高阶 ODE 求解器（Higher-Order ODE Solvers）** 通过在每步中利用更多信息（如导数的高阶近似或多步历史），将截断误差降低到 $O(\Delta t^{k+1})$（$k$ 阶方法），从而用更少的步数达到相同的精度。

### 扩散模型概率流 ODE 的一般形式

Song et al. (2021) 证明，DDPM 和 Score-based 模型对应的概率流 ODE 为：

$$\frac{dx}{dt} = f(t)x - \frac{1}{2}g^2(t)\nabla_x \log p_t(x)$$

其中 $f(t)$ 和 $g(t)$ 分别是 SDE 的漂移系数和扩散系数，$\nabla_x \log p_t(x) \approx -\epsilon_\theta(x,t)/\sigma_t$ 是学到的 score 函数。

直接用通用 ODE 求解器（如 RK4）求解这个方程可以工作，但忽略了方程的特殊结构。DPM-Solver 的核心贡献就是利用这个结构。

## DPM-Solver：半线性 ODE 的指数积分器 🔬

### 半线性结构的发现

Lu et al. (2022) 的关键洞察：在对数信噪比 $\lambda_t = \log(\alpha_t/\sigma_t)$ 参数化下，扩散概率流 ODE 可以改写为**半线性（Semi-Linear）** 形式：

$$\frac{dx_\lambda}{d\lambda} = f_\theta(x_\lambda, \lambda) = \underbrace{A(\lambda) \cdot x_\lambda}_{\text{线性部分}} + \underbrace{B(\lambda) \cdot \epsilon_\theta(x_\lambda, \lambda)}_{\text{非线性部分}}$$

具体地，在 VP（Variance Preserving）调度下：

$$\frac{dx}{d\lambda} = \frac{\alpha_\lambda'}{α_\lambda}x - \alpha_\lambda \sigma_\lambda \epsilon_\theta(x, \lambda)$$

其中 $\alpha_\lambda' = d\alpha/d\lambda$。线性部分 $A(\lambda)x$ 有解析解（简单的缩放），非线性部分涉及神经网络 $\epsilon_\theta$ 需要数值近似。

### 指数积分器（Exponential Integrator）推导

对半线性 ODE $dx/d\lambda = A(\lambda)x + g(\lambda)$，利用**常数变易法（Variation of Constants）**：

$$x(\lambda) = e^{\int_{\lambda_s}^{\lambda} A(\tau)d\tau} x(\lambda_s) + \int_{\lambda_s}^{\lambda} e^{\int_{\tau}^{\lambda} A(r)dr} g(\tau) d\tau$$

对扩散模型，线性部分的积分 $e^{\int A(\tau)d\tau} = \alpha_\lambda / \alpha_{\lambda_s}$ 有解析解。关键是对非线性积分项的近似：

$$x_\lambda = \frac{\alpha_\lambda}{\alpha_{\lambda_s}} x_{\lambda_s} - \alpha_\lambda \int_{\lambda_s}^{\lambda} e^{-\lambda} \hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda) d\lambda$$

这里 $\hat{\epsilon}_\theta$ 是对 $\epsilon_\theta$ 的多项式近似。不同阶数的 DPM-Solver 对应不同阶数的多项式近似。

### DPM-Solver 各阶公式

**DPM-Solver-1（一阶，等价于 DDIM）**：

对 $\epsilon_\theta$ 用零阶近似（常数），得到：

$$x_{\lambda_{i+1}} = \frac{\alpha_{i+1}}{\alpha_i} x_{\lambda_i} - \sigma_{i+1}(e^{h_i} - 1)\epsilon_\theta(x_{\lambda_i}, \lambda_i)$$

其中 $h_i = \lambda_{i+1} - \lambda_i$。这与 DDIM 更新公式完全等价。

**DPM-Solver-2（二阶，多步法）**：

对 $\epsilon_\theta$ 用一阶近似（线性外推），利用前一步的值：

$$x_{\lambda_{i+1}} = \frac{\alpha_{i+1}}{\alpha_i} x_{\lambda_i} - \sigma_{i+1}(e^{h_i} - 1)\epsilon_\theta(x_{\lambda_i}, \lambda_i) - \sigma_{i+1}\frac{e^{h_i} - 1 - h_i}{h_i} \cdot D_i$$

其中 $D_i = \epsilon_\theta(x_{\lambda_i}, \lambda_i) - \epsilon_\theta(x_{\lambda_{i-1}}, \lambda_{i-1})$ 是一阶差分。

**DPM-Solver-3（三阶，多步法）**：

进一步用二阶近似（二次多项式），利用前两步的值：

$$x_{\lambda_{i+1}} = (\text{DPM-Solver-2 项}) - \sigma_{i+1}\frac{e^{h_i} - 1 - h_i - h_i^2/2}{2h_i^2} \cdot D_i^{(2)}$$

其中 $D_i^{(2)}$ 是 $\epsilon_\theta$ 的二阶差分。

### PyTorch 伪代码：DPM-Solver-2 多步法

```python
def dpm_solver_2_multistep(model, x_T, timesteps, alphas, sigmas):
    """
    DPM-Solver-2 多步法采样器
    Args:
        model: 噪声预测网络 ε_θ
        x_T: 初始噪声 ~ N(0, I)
        timesteps: 离散时间步序列 [t_0, t_1, ..., t_N]，从大到小
        alphas, sigmas: 噪声调度参数
    """
    lambdas = torch.log(alphas / sigmas)  # 对数信噪比
    x = x_T
    eps_prev = None  # 存储前一步的 ε 预测
    
    for i in range(len(timesteps) - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        lam_cur, lam_next = lambdas[i], lambdas[i + 1]
        h = lam_next - lam_cur  # 步长（在 λ 空间）
        
        # 当前步的噪声预测
        eps_cur = model(x, t_cur)
        
        if eps_prev is None:
            # 第一步退化为 DPM-Solver-1 (DDIM)
            x = (alphas[i+1] / alphas[i]) * x \
                - sigmas[i+1] * (torch.exp(h) - 1) * eps_cur
        else:
            # 二阶多步更新
            h_prev = lam_cur - lambdas[i - 1]
            r = h / h_prev  # 步长比
            D = eps_cur - eps_prev  # 一阶差分
            
            x = (alphas[i+1] / alphas[i]) * x \
                - sigmas[i+1] * (torch.exp(h) - 1) * eps_cur \
                - sigmas[i+1] * (torch.exp(h) - 1 - h) / (2 * r) * D
        
        eps_prev = eps_cur
    
    return x
```

## DPM-Solver++：数据预测参数化 ⚙️

### 从噪声预测到数据预测

DPM-Solver++ (Lu et al., 2022b) 的改进核心是**参数化重整（Reparameterization）**。原始 DPM-Solver 对 $\epsilon_\theta$（噪声预测）做多项式近似，而 DPM-Solver++ 对 $x_\theta$（数据预测）做近似：

$$\hat{x}_\theta(x_t, t) = \frac{x_t - \sigma_t \epsilon_\theta(x_t, t)}{\alpha_t}$$

**为什么这更好？** 在使用 **CFG（Classifier-Free Guidance）** 时，有效的噪声预测变为：

$$\tilde{\epsilon}(x_t, t, c) = (1 + w)\epsilon_\theta(x_t, t, c) - w\epsilon_\theta(x_t, t, \varnothing)$$

当引导强度 $w$ 较大时（通常 $w = 7.5$），$\tilde{\epsilon}$ 的幅度远大于训练时的分布，导致 $\epsilon$ 空间的多项式近似不准确。而数据预测 $\hat{x}_\theta$ 的幅度相对稳定，多项式近似更精确。

### DPM-Solver++ 2M 更新公式

$$x_{t_{i+1}} = \frac{\sigma_{i+1}}{\sigma_i} x_{t_i} + \alpha_{i+1}\left(e^{-h_i} - 1\right)\hat{x}_\theta^{(i)} + \alpha_{i+1}\frac{e^{-h_i} - 1 + h_i}{2r_i}(\hat{x}_\theta^{(i)} - \hat{x}_\theta^{(i-1)})$$

这里 "2M" 表示 2 阶多步法（Multistep），每步只需 1 次网络评估。

## Heun 方法：预测-校正法 ⚙️

### 算法原理

**Heun 方法（Heun's Method）** 是经典的二阶 ODE 求解器，也叫**改进 Euler 法（Improved Euler Method）** 或**预测-校正法（Predictor-Corrector）**：

1. **预测（Predict）**：用 Euler 方法估计下一步 $\tilde{x}_{t+h} = x_t + h \cdot f(x_t, t)$
2. **校正（Correct）**：在预测点再求一次导数，用梯形法则修正：$x_{t+h} = x_t + \frac{h}{2}[f(x_t, t) + f(\tilde{x}_{t+h}, t+h)]$

```python
def heun_sampler(model, x_T, timesteps, noise_schedule):
    """
    Heun 预测-校正采样器
    每步需要 2 次网络评估（NFE=2）
    """
    x = x_T
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur
        
        # 步骤 1: 预测（Euler 步）
        d_cur = compute_ode_derivative(model, x, t_cur, noise_schedule)
        x_pred = x + dt * d_cur
        
        # 步骤 2: 校正（梯形法则）
        # 在预测点处再次评估导数
        if t_next > 0:  # 最后一步不需要校正
            d_next = compute_ode_derivative(model, x_pred, t_next, noise_schedule)
            x = x + dt * 0.5 * (d_cur + d_next)  # 梯形法则
        else:
            x = x_pred
    
    return x

def compute_ode_derivative(model, x, t, ns):
    """计算概率流 ODE 的导数 dx/dt"""
    eps = model(x, t)
    alpha_t, sigma_t = ns.alpha(t), ns.sigma(t)
    # dx/dt = f(t)x + g(t)ε_θ（具体形式取决于噪声调度）
    return (alpha_t * eps - x) / sigma_t
```

代价是每步 2 次 NFE，但精度从 $O(\Delta t)$ 提升到 $O(\Delta t^2)$。在步数充裕时（>20步），Heun 因为 NFE 翻倍不一定比 DPM-Solver-2 更划算。

## Karras 噪声调度 ⚙️

Karras et al. (2022) 提出了一套经过精心设计的**噪声调度（Noise Schedule）**，与高阶求解器配合使用：

### 调度公式

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)\right)^\rho$$

其中 $\rho = 7$ 是推荐值。这使得时间步在对数空间中分布更均匀，步长在高噪声区域更大（误差容忍度高），低噪声区域更小（需要精细控制）。

### 预处理与后处理

Karras 还建议对输入和输出做缩放：

$$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \quad c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}$$

$$F_\theta(x; \sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot f_\theta\left(\frac{x}{c_{\text{in}}(\sigma)}; c_{\text{noise}}(\sigma)\right)$$

这确保了网络输入输出的方差在所有噪声水平上都保持接近 1，改善了训练稳定性和采样质量。

## UniPC：统一预测-校正框架 🔬

**UniPC（Unified Predictor-Corrector）**（Zhao et al., 2023）是一个统一框架，将多步法和预测-校正法结合：

- **预测器（Predictor）**：用多步法预测下一步的值
- **校正器（Corrector）**：利用预测值处的梯度信息进行校正

UniPC 的关键创新在于：校正步不需要额外的 NFE——它复用预测步中已经计算的网络评估值和历史值，通过更精确的插值公式来校正。

这意味着 UniPC 在**相同 NFE 下比纯多步法更精确**，在 5-10 步的极低步数场景表现尤为突出。

## 自适应步长方法：RK45 🔬

**RK45（Runge-Kutta-Fehlberg）** 是一种自适应步长的 ODE 求解器。核心思想：

1. 同时用 4 阶和 5 阶方法计算结果
2. 两者之差作为误差估计
3. 如果误差 > 容差，缩小步长重新计算
4. 如果误差 << 容差，增大步长

$$h_{\text{new}} = h_{\text{old}} \cdot \min\left(\beta_{\max}, \max\left(\beta_{\min}, \beta \cdot \left(\frac{\text{tol}}{|e|}\right)^{1/5}\right)\right)$$

对扩散模型的优势：

- 在"容易"的区域（高噪声，特征平滑）自动用大步长
- 在"困难"的区域（低噪声，精细结构）自动缩小步长
- 无需手动选择步数

劣势：NFE 不确定，不利于延迟预测和批处理优化。实际中通常 NFE 在 50-150 之间。

## 求解器综合对比 🔬

### CIFAR-10 无条件生成 (FID↓)

| 求解器 | 5 步 | 10 步 | 20 步 | 50 步 | NFE/步 |
|--------|------|-------|-------|-------|--------|
| Euler (DDIM) | 42.17 | 13.36 | 4.70 | 2.86 | 1 |
| Heun (2阶) | 23.18 | 5.27 | 3.15 | 2.78 | 2 |
| DPM-Solver-2 | 15.42 | 4.15 | 3.42 | 2.84 | 1 (多步) |
| DPM-Solver-3 | 8.73 | 3.45 | 3.12 | 2.82 | 1 (多步) |
| DPM-Solver++ 2M | 14.30 | 3.95 | 3.24 | 2.82 | 1 (多步) |
| UniPC (3阶) | 7.51 | 3.30 | 3.08 | 2.80 | 1 (多步) |
| RK45 (自适应) | — | — | — | 2.78 | ~100 (总) |

> 数据来源：DPM-Solver、UniPC 原论文及社区复现。FID 在 CIFAR-10 50K 样本上评估。

### ImageNet 256×256 条件生成 (FID↓, CFG=1.5)

| 求解器 | 10 步 | 20 步 | 50 步 | 100 步 |
|--------|-------|-------|-------|--------|
| DDIM | 18.25 | 6.82 | 3.95 | 3.10 |
| DPM-Solver++ 2M | 5.47 | 3.42 | 2.88 | 2.75 |
| DPM-Solver++ 3M | 4.82 | 3.18 | 2.82 | 2.74 |
| UniPC (3阶) | 4.35 | 3.05 | 2.80 | 2.73 |
| Heun | 6.10 | 3.55 | 2.85 | 2.74 |

> 在条件生成+CFG 场景下，DPM-Solver++ 的数据预测参数化优势明显。

## 选择求解器的实践指南 ⚙️

### 决策流程

```
你的场景是什么？
│
├── 步数 ≥ 20，追求最佳质量
│   └── DPM-Solver++ 2M（稳定、快速、质量好）
│       或 Heun（略好质量，NFE 翻倍）
│
├── 步数 10-20，平衡速度和质量
│   └── DPM-Solver++ 2M（最佳性价比）
│       或 UniPC 3阶（极致质量）
│
├── 步数 5-10，速度优先
│   └── UniPC 3阶（低步数最佳）
│       或 DPM-Solver-3（接近）
│
├── 步数 < 5
│   └── 不建议用 ODE 求解器 → 转向蒸馏方法
│
├── 需要确定性 NFE（生产部署）
│   └── DPM-Solver++ 2M 或 UniPC
│
└── 学术研究/不关心 NFE
    └── RK45 自适应（最精确参考基线）
```

### 与其他加速方法的协同

| 组合 | 效果 | 注意事项 |
|------|------|---------|
| DPM-Solver++ + TensorRT | 步数少 + 单步快 | 推荐的默认方案 |
| DPM-Solver++ + DeepCache | 进一步减少 NFE | 可能需要调整缓存间隔 |
| 高阶求解器 + 量化 | 步数少 + 精度低 | 低步数下量化误差更敏感 |
| 高阶求解器 + 蒸馏 | **不兼容** | 蒸馏模型有专用步数 |

## 小结

| 要点 | 说明 |
|------|------|
| 核心原理 | 利用扩散 ODE 的半线性结构，精确解析线性部分 |
| 关键公式 | 指数积分器 + 多项式近似非线性部分 |
| 实际收益 | 50 步 → 10-20 步，质量几乎无损 |
| 推荐方案 | DPM-Solver++ 2M 是大多数场景的最佳选择 |
| 低步数极限 | 10 步以下质量下降明显，需要蒸馏方法 |

---

> **下一篇**：[蒸馏方法](./03-distillation.md) — 突破 ODE 求解器的步数下限

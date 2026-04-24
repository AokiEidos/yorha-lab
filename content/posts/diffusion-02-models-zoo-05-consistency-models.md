---
title: "Consistency Models"
date: 2026-04-20T17:14:10.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Consistency Models

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[SDE/ODE 统一视角](../01-fundamentals/07-sde-ode.md)，[DDIM 详解](./02-ddim.md)

## 一步生成的追求

扩散模型最大的实际痛点是采样速度——即使用 DDIM 或 DPM-Solver 加速，也需要 10-50 步。能否实现**单步生成**（一次网络前向传播就得到高质量图像）？

**一致性模型（Consistency Models）**（Song et al., 2023）正是为此设计的。它利用概率流 ODE 轨迹上的"一致性"约束，实现 1-2 步的高质量生成。与渐进式蒸馏（Progressive Distillation）需要多轮迭代不同，Consistency Models 提供了更优雅的理论框架。

## 核心思想：自一致性约束

### 概率流 ODE 轨迹

回顾：概率流 ODE 定义了从噪声 $x_T$ 到数据 $x_0$ 的确定性路径。这条路径上的每个点 $x_t$（$t \in [\epsilon, T]$）都对应同一个数据样本 $x_0$。

```
ODE 轨迹:  x_T ─── x₃ ─── x₂ ─── x₁ ─── x_ε ≈ x₀
               所有这些点都应该映射到同一个 x₀
```

### 一致性函数

定义**一致性函数（Consistency Function）** $f_\theta$：将 ODE 轨迹上的任意点映射到轨迹的起点：

$$f_\theta: (x_t, t) \mapsto x_\epsilon$$

**自一致性约束（Self-Consistency Property）**：同一条 ODE 轨迹上的任意两点 $x_t$ 和 $x_{t'}$，映射结果必须相同：

$$\boxed{f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \forall \, t, t' \in [\epsilon, T]}$$

### 边界条件与参数化

一致性函数必须满足：$f_\theta(x_\epsilon, \epsilon) = x_\epsilon$（在起点处是恒等映射）。

实现方式（skip connection）：

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

其中 $c_{\text{skip}}(\epsilon) = 1$，$c_{\text{out}}(\epsilon) = 0$，确保边界条件成立。具体地，Song et al. 使用：

$$c_{\text{skip}}(t) = \frac{\sigma_{\text{data}}^2}{(t - \epsilon)^2 + \sigma_{\text{data}}^2}, \quad c_{\text{out}}(t) = \frac{\sigma_{\text{data}}(t - \epsilon)}{\sqrt{\sigma_{\text{data}}^2 + t^2}}$$

其中 $\sigma_{\text{data}} = 0.5$ 是数据标准差的经验值。这种参数化保证了在 $t = \epsilon$ 时输出恒等，在 $t$ 较大时主要依赖网络 $F_\theta$ 的预测。

## 两种训练范式

### 一致性蒸馏（Consistency Distillation, CD）

**需要**：一个预训练的扩散模型（教师模型）

思路：利用教师模型的 ODE 求解器生成相邻点对，训练一致性模型使它们的输出一致。

1. 取 ODE 轨迹上的相邻点 $(x_{t_{n+1}}, x_{t_n})$
2. $x_{t_n}$ 通过教师模型的一步 ODE 求解从 $x_{t_{n+1}}$ 得到
3. 训练损失：$\| f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(x_{t_n}, t_n) \|^2$

其中 $\theta^-$ 是目标网络（EMA 更新），类似 DQN 的目标网络技巧，防止训练崩溃。EMA 衰减率 $\mu$ 随训练进行从 0 逐渐增大到接近 1。

```python
def consistency_distillation_loss(model, model_ema, teacher, x_0, t_n, t_n1, noise_schedule):
    """一致性蒸馏训练步骤"""
    # 构造 x_{t_{n+1}}
    noise = torch.randn_like(x_0)
    x_tn1 = noise_schedule.add_noise(x_0, noise, t_n1)  # q(x_{t_{n+1}} | x_0)
    
    # 教师模型 ODE 一步：x_{t_{n+1}} → x_{t_n}（使用 Euler 或 Heun 求解器）
    with torch.no_grad():
        teacher_eps = teacher(x_tn1, t_n1)
        # Euler step: x_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) * dx/dt
        x_tn = x_tn1 + (t_n - t_n1) * (x_tn1 - teacher_eps) / t_n1
    
    # 一致性损失：两个时间点映射到同一起点
    pred_tn1 = model(x_tn1, t_n1)       # 在线网络
    with torch.no_grad():
        pred_tn = model_ema(x_tn, t_n)   # 目标网络 (EMA)
    
    loss = F.mse_loss(pred_tn1, pred_tn)
    return loss
```

### 一致性训练（Consistency Training, CT）

**不需要**预训练扩散模型——从头训练一致性模型。

CT 的关键洞察：当相邻时间步 $t_n$ 和 $t_{n+1}$ 足够接近时，可以用当前模型自身的得分估计来近似 ODE 一步。具体地，利用：

$$x_{t_n} \approx x_{t_{n+1}} + (t_n - t_{n+1}) \cdot \left(\frac{x_{t_{n+1}} - F_\theta(x_{t_{n+1}}, t_{n+1})}{t_{n+1}}\right)$$

```python
def consistency_training_loss(model, model_ema, x_0, t_n, t_n1, noise_schedule):
    """一致性训练：无需教师模型"""
    noise = torch.randn_like(x_0)
    x_tn1 = noise_schedule.add_noise(x_0, noise, t_n1)
    
    # 关键区别：用当前模型自身做 ODE 一步估计（而非教师模型）
    with torch.no_grad():
        # 利用当前模型的输出做 Euler 步
        model_output = model(x_tn1, t_n1)
        # 从模型输出中提取得分估计
        score_est = (x_tn1 - model_output) / t_n1
        x_tn = x_tn1 + (t_n - t_n1) * score_est
    
    # 一致性损失
    pred_tn1 = model(x_tn1, t_n1)
    with torch.no_grad():
        pred_tn = model_ema(x_tn, t_n)
    
    loss = F.mse_loss(pred_tn1, pred_tn)
    return loss
```

CT 的训练需要一个关键的**课程策略（Curriculum）**：随训练进行逐步增加离散化步数 $N$（从小到大）。初始时 $N$ 小（如 2），相邻步间距大但噪声估计误差也大；训练后期 $N$ 增大（如 150），间距小且估计更准确。

### CT vs CD 对比

| 方面 | 一致性蒸馏 (CD) | 一致性训练 (CT) |
|------|----------------|----------------|
| 需要教师模型 | 是（预训练扩散模型） | 否 |
| ODE 步估计质量 | 高（教师模型精确） | 较低（自身估计） |
| FID (ImageNet 64, 1步) | 3.55 | 7.20 |
| 训练复杂度 | 较低 | 较高（需要课程策略） |
| 独立性 | 依赖预训练 | 完全独立 |

## 采样过程

### 单步生成

```python
def one_step_generation(model, shape, T=80.0, eps=0.002):
    """一步生成：最快但质量略低"""
    z = torch.randn(shape) * T             # 采样噪声（缩放到 T）
    x = model(z, T)                        # 一步直接映射到 x_epsilon
    return x
```

### 多步生成（详细算法）

可以用 2-4 步进一步提升质量——交替"去噪 -> 加噪 -> 去噪"：

```python
def multi_step_generation(model, shape, time_steps=[80.0, 24.4, 5.84, 0.9], eps=0.002):
    """
    多步采样算法（Multistep Consistency Sampling）
    
    time_steps: 递减的时间步序列 [T = tau_1 > tau_2 > ... > tau_K > eps]
    每一步：(1) 去噪到 x_eps  (2) 重新加噪到 tau_{k+1}  (3) 再次去噪
    """
    # 第 1 步：从纯噪声开始
    z = torch.randn(shape) * time_steps[0]
    x_denoised = model(z, time_steps[0])      # 去噪到 x_eps
    
    # 第 2..K 步：加噪 → 去噪循环
    for k in range(1, len(time_steps)):
        tau = time_steps[k]
        # 加噪：将干净预测重新扰动到 tau_k
        noise = torch.randn_like(x_denoised)
        x_noisy = x_denoised + torch.sqrt(tau**2 - eps**2) * noise
        
        # 去噪：用一致性模型映射回 x_eps
        x_denoised = model(x_noisy, tau)
    
    return x_denoised
```

**为什么多步有效？** 每次加噪后重新去噪，模型有机会从不同的噪声水平重新审视图像，修正之前的错误。类似于"草稿 -> 加噪糊化 -> 重新精修"的过程。

```
详细流程（3 步采样）:
步骤 1: z ~ N(0, T²I) ──→ f_θ(z, T) ──→ x̂₀⁽¹⁾  [粗略初稿]
步骤 2: x̂₀⁽¹⁾ + noise·√(τ₂²-ε²) ──→ x_τ₂ ──→ f_θ(x_τ₂, τ₂) ──→ x̂₀⁽²⁾  [精修一轮]
步骤 3: x̂₀⁽²⁾ + noise·√(τ₃²-ε²) ──→ x_τ₃ ──→ f_θ(x_τ₃, τ₃) ──→ x̂₀⁽³⁾  [最终结果]
```

## 改进版本

### iCT（Improved Consistency Training, 2024）

Song & Dhariwal 的改进版（2024），解决了原始 CT 的多个训练不稳定问题：

**1. 连续时间公式**：消除离散化误差，不再需要手动设定离散步数 $N$ 的课程。

**2. Lognormal 时间分布**：替代均匀采样。时间 $t$ 从 lognormal 分布中采样：

$$\ln t \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$$

其中 $P_{\text{mean}} = -1.1$，$P_{\text{std}} = 2.0$。这使得训练在中间噪声水平（信息量最大的区域）获得更多样本，避免在极高或极低噪声处浪费计算。

**3. Pseudo-Huber 损失函数**：替代 MSE/LPIPS，对离群值更鲁棒：

$$L_{\text{Pseudo-Huber}}(\hat{x}, x) = \sqrt{\|\hat{x} - x\|_2^2 + c^2} - c$$

其中 $c$ 是常数（Song & Dhariwal 使用 $c = 0.00054 \sqrt{d}$，$d$ 是数据维度）。当误差小时行为接近 L2，误差大时接近 L1——避免 MSE 对离群点的过度惩罚，同时保持小误差处的平滑梯度。

**4. Scaling Laws**：iCT 展示了 Consistency Models 的首个 Scaling Laws 结果：

| 模型大小 | 参数量 | FID (ImageNet 512, 2步) |
|---------|--------|------------------------|
| Small | 152M | 4.02 |
| Medium | 300M | 3.20 |
| Large | 600M | 2.77 |
| XL | 1.5B | 2.30 |

FID 与模型参数量呈现清晰的幂律关系，说明 Consistency Models 可以通过增大模型继续获益。

### sCT（Simplified Consistency Training）

Lu & Song（2024）进一步简化训练流程，核心贡献：

- **TrigFlow 框架**：用三角函数统一了扩散训练和一致性训练的参数化
- **去除 EMA 目标网络**：发现在适当设置下可以不用 EMA，简化训练
- 在 ImageNet 512 上单步 FID 达到 2.06，两步 FID 达到 1.88

### LCM（Latent Consistency Model）

**潜在一致性模型（Latent Consistency Model, LCM）**（Luo et al., 2023）将 Consistency Model 思想应用到**潜在空间（Latent Space）**，直接在 Stable Diffusion 的 VAE 潜在空间中训练：

```
LCM 架构:
┌───────────┐     ┌─────────────────────┐     ┌───────────┐
│  z_T      │ ──→ │  一致性模型 (潜在空间) │ ──→ │  z_0 预测  │ ──→ VAE Decode ──→ 图像
│ (潜在噪声) │     │  基于 SD 的 U-Net     │     │           │
└───────────┘     └─────────────────────┘     └───────────┘
```

LCM 的关键创新：
- 在潜在空间训练，直接兼容 SD/SDXL 生态
- **LCM-LoRA**：只需训练一个 LoRA 适配器（~67M 参数），即可将任何 SD 模型转化为 4 步生成
- 社区广泛采用，成为实时生成的主流方案

## 与渐进式蒸馏的对比

**渐进式蒸馏（Progressive Distillation）**（Salimans & Ho, 2022）是另一种加速采样的方法。对比如下：

| 方面 | Consistency Models (CD) | Progressive Distillation |
|------|------------------------|--------------------------|
| 蒸馏轮数 | 1 轮 | 需要多轮（每轮减半步数） |
| 理论基础 | ODE 轨迹一致性 | 步数折半蒸馏 |
| 单步 FID (ImageNet 64) | 3.55 | 9.12 |
| 两步 FID (ImageNet 64) | 2.93 | 4.51 |
| 灵活性 | 可选任意步数 | 步数固定为 $2^n$ |
| 训练效率 | 高（一次训练） | 低（每轮需要重新训练） |

Consistency Models 在几乎所有维度上优于渐进式蒸馏，尤其是在极少步数（1-2 步）的情况下优势显著。

## 完整方法族谱

```
Consistency Models 方法演进:

Song et al. 2023 ─┬─ CD (Consistency Distillation) ─────→ LCM (Luo 2023)
 (原始论文)        │   需要教师模型                          潜在空间 + LoRA
                   │                                        → LCM-LoRA 社区广泛使用
                   ├─ CT (Consistency Training) ─────────→ iCT (Song & Dhariwal 2024)
                   │   独立训练                              连续时间 + Pseudo-Huber + Scaling
                   │                                        
                   └─────────────────────────────────────→ sCT (Lu & Song 2024)
                                                            TrigFlow 统一框架
```

## 与其他方法的对比

| 方法 | 采样步数 | FID (ImageNet 64) | FID (ImageNet 512) | 需要预训练 |
|------|---------|-------------------|--------------------|----------|
| DDPM | 250 | 2.07 | — | 否 |
| DDIM | 50 | 4.67 | — | 否 |
| Prog. Distillation | 1 | 9.12 | — | 是 |
| Prog. Distillation | 2 | 4.51 | — | 是 |
| CD | 1 | 3.55 | — | 是 |
| CD | 2 | 2.93 | — | 是 |
| CT | 1 | 7.20 | — | 否 |
| iCT | 1 | 3.25 | 3.20 (XL) | 否 |
| iCT | 2 | 2.77 | 2.30 (XL) | 否 |
| sCT | 1 | — | 2.06 | 否 |
| sCT | 2 | — | 1.88 | 否 |

## 小结

| 概念 | 要点 |
|------|------|
| 自一致性 | ODE 轨迹上任意点映射到同一起点 |
| 一致性蒸馏 (CD) | 需要教师模型，质量好，单轮蒸馏 |
| 一致性训练 (CT) | 独立训练，无需预训练，课程策略 |
| 单步生成 | 一次前向传播即可生成 |
| 多步改进 | 2-4 步交替去噪加噪，进一步提升质量 |
| iCT 改进 | Pseudo-Huber 损失 + lognormal 时间 + Scaling Laws |
| LCM 落地 | 潜在空间一致性模型，LCM-LoRA 社区广泛采用 |
| vs 渐进蒸馏 | Consistency Models 在 1-2 步场景全面优于渐进式蒸馏 |

---

> **下一篇**：[Flow Matching](./06-flow-matching.md)

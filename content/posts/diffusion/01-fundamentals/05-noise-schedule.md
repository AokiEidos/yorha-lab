---
title: "噪声调度"
date: 2026-04-17T17:43:24.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 噪声调度

> ⚙️ 进阶 | 前置知识：[前向过程详解](./02-forward-process)，了解 $\bar{\alpha}_t$ 和 SNR 的定义

## 为什么噪声调度重要

**噪声调度（Noise Schedule）** 定义了噪声系数 $\beta_t$（或等价地，$\bar{\alpha}_t$）随时间步 $t$ 的变化规律。它决定了：

- 前向过程中数据被"摧毁"的速度
- 模型在不同噪声水平下需要分配多少学习能力
- 最终的生成质量和多样性

一个好的噪声调度应该让信噪比（SNR）平滑地从无穷大下降到接近零，并且在各个噪声水平上都给模型足够的学习信号。

## 线性调度（Linear Schedule）

DDPM 原始论文使用的调度方式，最简单直观：

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

其中 $\beta_{\min} = 10^{-4}$，$\beta_{\max} = 0.02$，$T = 1000$。

```python
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
```

**特点**：
- $\bar{\alpha}_t$ 呈现近似指数衰减
- 前半段 SNR 下降缓慢，大量步数浪费在高 SNR 区域（图像几乎没变化）
- 后半段 SNR 急剧下降，过渡到纯噪声的过程太快

**问题**：在高分辨率图像上，线性调度会导致中间噪声水平（信号和噪声比较均衡的区域）覆盖不足，影响生成质量。

## 余弦调度（Cosine Schedule）

Nichol & Dhariwal（2021, Improved DDPM）提出的改进调度，基于 $\bar{\alpha}_t$ 的直接定义：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是一个小偏移量，防止 $\beta_t$ 在 $t$ 接近 $T$ 时过大。

```python
def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.clamp(betas, max=0.999)  # 防止极端值
    return betas
```

**特点**：
- $\bar{\alpha}_t$ 遵循余弦函数，衰减更加均匀
- 中间噪声水平覆盖充分，生成质量显著提升
- 前后两端的变化更平缓

### 线性 vs 余弦调度对比

```
ᾱ_t
1.0 ┤ ●●●●●●●
    │        ●●    ← 线性：前半段变化缓慢
0.8 ┤   ○       ●
    │     ○      ●●
0.6 ┤       ○       ●●
    │         ○        ●●
0.4 ┤           ○         ●●
    │             ○          ●●●
0.2 ┤               ○            ●●●●
    │                  ○○             ●●●●●
0.0 ┤                     ○○○○○○○○○○        ●●●
    └──────────────────────────────────────────
    t=0                                    t=T
    
    ○ = 余弦调度（均匀衰减）
    ● = 线性调度（前慢后快）
```

## 信噪比（SNR）统一视角

⚙️ 进阶

不同的噪声调度可以用**信噪比（Signal-to-Noise Ratio）** 统一描述：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

对数 SNR 提供了更直观的视角：

$$\log \text{SNR}(t) = \log \bar{\alpha}_t - \log(1 - \bar{\alpha}_t)$$

**SNR 视角的好处**：

1. **调度无关**：不管用什么调度策略，模型本质上都是在不同 SNR 水平下学习去噪。好的调度应该让 $\log \text{SNR}$ 在时间轴上近似线性下降。

2. **损失加权**：Min-SNR 加权策略（Hang et al., 2023）提出根据 SNR 自适应调整损失权重：
   $$w(t) = \min(\text{SNR}(t), \gamma)$$
   其中 $\gamma$ 是截断阈值（通常为 5）。这避免了高 SNR 时间步（简单任务）主导训练。

3. **跨模型比较**：用 SNR 作为横轴可以公平比较不同步数、不同调度的模型。

## 高分辨率图像的调度调整

当图像分辨率提高时（如从 64×64 到 512×512），同样的噪声调度会产生问题：

**问题**：高分辨率图像在相同 $\beta_t$ 下，**感知噪声水平更低**。原因是人眼（和神经网络）对空间相关结构敏感，而高分辨率图像在加入少量独立噪声后，空间结构（低频信息）仍然大量保留。

**解决方案**：

1. **偏移噪声（Offset Noise）**（Lin et al., 2023）：在独立像素噪声之外，额外添加一个空间均匀的低频噪声项，确保模型能学到纯黑/纯白等均匀色彩的图像。

2. **缩放调度**：根据分辨率调整 SNR 范围，例如 Stable Diffusion XL 使用了比 SD 1.x 更大的噪声范围。

3. **Zero Terminal SNR**（Lin et al., 2023）：确保 $\bar{\alpha}_T = 0$（即 $\text{SNR}(T) = 0$），使终端分布精确为 $\mathcal{N}(0, \mathbf{I})$，避免训练和推理之间的分布不匹配。

## 连续时间调度

🔬 深入

在 SDE 框架（详见 [SDE/ODE 统一视角](./07-sde-ode)）中，噪声调度被推广到连续时间 $t \in [0, 1]$：

$$\text{SNR}(t) = e^{-\lambda(t)}$$

其中 $\lambda(t)$ 是对数 SNR 的调度函数。常见选择包括：

| 调度 | $\lambda(t)$ | 来源 |
|------|-------------|------|
| 线性 | $\lambda_{\min} + t(\lambda_{\max} - \lambda_{\min})$ | VP-SDE |
| 余弦 | $-2\log\tan\left(\frac{\pi t}{2}\right)$ | Improved DDPM 连续版 |
| 可学习 | 单调神经网络 | Variational Diffusion Models |

连续时间框架让噪声调度的设计更加灵活，也方便了理论分析。

## 实践建议

| 场景 | 推荐调度 | 理由 |
|------|---------|------|
| 低分辨率（64×64） | 线性或余弦 | 差异不大，线性实现更简单 |
| 中高分辨率（256-512） | 余弦 | 中间 SNR 覆盖更均匀 |
| 高分辨率（1024+） | 余弦 + 偏移噪声 | 需要额外低频噪声 |
| Latent Diffusion | 缩放后的线性/余弦 | 潜空间的噪声尺度不同于像素空间 |
| 追求最优 | Min-SNR 加权 + Zero Terminal SNR | 综合收益最大 |

## 小结

| 概念 | 要点 |
|------|------|
| 线性调度 | 简单但前慢后快，中间 SNR 覆盖不均匀 |
| 余弦调度 | $\bar{\alpha}_t$ 均匀衰减，生成质量更好 |
| SNR | 统一描述不同调度，$\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$ |
| 高分辨率问题 | 需要偏移噪声和/或调整 SNR 范围 |
| Zero Terminal SNR | 确保 $\bar{\alpha}_T = 0$，避免训练/推理不匹配 |

---

> **下一篇**：[得分函数与得分匹配](./06-score-matching) — 从另一个视角理解扩散模型：得分函数和去噪得分匹配。

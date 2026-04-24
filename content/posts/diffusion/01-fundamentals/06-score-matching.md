---
title: "得分函数与得分匹配"
date: 2026-04-17T17:44:29.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 得分函数与得分匹配

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[反向过程与训练目标](./03-reverse-process)，基本微积分（梯度概念）

## 另一种看扩散模型的方式

到目前为止，我们从"加噪→去噪"的概率框架理解扩散模型。本文介绍一条平行的技术路线——**基于得分的生成模型（Score-based Generative Models）**——它从不同角度出发，最终与 DDPM 殊途同归。

这两条路线的统一是扩散模型理论的重要里程碑（Song et al., 2021），理解这种统一对深入掌握扩散模型至关重要。

## 什么是得分函数

### 定义

**得分函数（Score Function）** 是数据概率分布的对数的梯度：

$$s(x) = \nabla_x \log p(x)$$

其中 $p(x)$ 是数据的概率密度函数，$\nabla_x$ 表示对 $x$ 求梯度。

### 直觉理解

想象一张地形图，其中海拔高度代表数据的概率密度——数据集中的地方像山峰，数据稀少的地方像山谷。

- **概率密度** $p(x)$：地形图的海拔
- **得分函数** $\nabla_x \log p(x)$：在地形图上每一点指向"上坡方向"的箭头

得分函数告诉你："从当前位置出发，往哪个方向走能到达概率更高（数据更密集）的地方。"

```
概率密度 p(x)              得分向量场 ∇log p(x)

     ╱╲                        ↗ ↑ ↖
    ╱  ╲                     → 山峰 ←
   ╱    ╲                      ↘ ↓ ↙
  ╱      ╲                   ↗       ↖
 ╱   ╱╲   ╲               → 山峰 ←
╱   ╱  ╲   ╲                ↘     ↙
```

### 为什么用得分函数而非概率密度

直接学习 $p(x)$ 需要计算**配分函数（Partition Function / Normalization Constant，即确保概率分布总和为 1 的归一化常数）** $Z = \int p_{\text{unnorm}}(x) dx$——在高维空间中这通常是不可计算的（intractable）。

得分函数 $\nabla_x \log p(x)$ 的优势：归一化常数 $Z$ 在取对数后变成常数，求梯度时消失了：

$$\nabla_x \log p(x) = \nabla_x \log \frac{p_{\text{unnorm}}(x)}{Z} = \nabla_x \log p_{\text{unnorm}}(x)$$

因此，我们可以绕过配分函数，直接学习得分函数。

## 得分匹配（Score Matching）

### 朴素得分匹配

**得分匹配（Score Matching）** 的目标是训练一个**得分网络（Score Network）** $s_\theta(x) \approx \nabla_x \log p(x)$。

最直觉的损失函数：

$$L = \mathbb{E}_{x \sim p(x)} \left[ \| s_\theta(x) - \nabla_x \log p(x) \|^2 \right]$$

**问题**：我们不知道真实的 $\nabla_x \log p(x)$——如果知道的话就不需要学了。

Hyvärinen（2005）证明了一个巧妙的恒等式，将上式转化为不依赖真实得分的等价形式（涉及 $s_\theta$ 的雅可比矩阵的迹），但这个公式计算代价太高（需要对每个维度做反向传播），不适合高维数据如图像。

### 去噪得分匹配（Denoising Score Matching, DSM）

⚙️ 进阶

Vincent（2011）提出了一个优雅的替代方案——**去噪得分匹配（Denoising Score Matching, DSM）**：

**核心思想**：不直接匹配干净数据的得分，而是匹配**加噪数据**的得分。

给定噪声水平 $\sigma$，加噪后的分布为：

$$q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x}|x) p(x) dx, \quad q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 \mathbf{I})$$

加噪分布的得分有一个简洁的形式：

$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = \frac{x - \tilde{x}}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

其中 $\epsilon = \tilde{x} - x$ 是加入的噪声。

**DSM 损失**：

$$L_{\text{DSM}} = \mathbb{E}_{x \sim p(x), \tilde{x} \sim q_\sigma(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) \right\|^2 \right]$$

$$= \mathbb{E}_{x, \epsilon} \left[ \left\| s_\theta(x + \sigma\epsilon) + \frac{\epsilon}{\sigma} \right\|^2 \right]$$

**关键洞察**：DSM 损失等价于让网络预测"从噪声数据指向干净数据的方向"——这正是**去噪**！

## 与 DDPM 的等价性

🔬 深入

现在让我们揭示两条路线的深层联系。

在 DDPM 的前向过程中，$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$。

对应的得分函数为：

$$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

其中我们使用了 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 的关系。

因此：

$$\boxed{s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}}$$

**得分函数和噪声预测只差一个与时间步相关的缩放因子！** 训练一个噪声预测网络 $\epsilon_\theta$，等价于训练一个得分估计网络 $s_\theta$。

DDPM 的训练损失 $\|\epsilon - \epsilon_\theta\|^2$ 等价于（差一个常数系数）：

$$\| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0) \|^2$$

这正是多尺度 DSM 损失。两条路线在数学上完全等价。

## NCSN：多尺度去噪得分匹配

**NCSN（Noise Conditional Score Network，噪声条件得分网络）**（Song & Ermon, 2019）是基于得分的生成模型的开山之作。

### 为什么需要多尺度

单一噪声水平的得分估计有严重问题：

- **低密度区域的得分估计不准确**：数据稀疏的区域几乎没有训练样本，得分估计不可靠
- **"流形假说"问题**：真实数据通常集中在高维空间的低维流形（Manifold，即高维空间中数据实际分布的低维曲面）上，流形之外的得分没有定义

**解决方案**：使用多个噪声水平 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$，从大噪声到小噪声逐级估计得分：

- 大噪声 $\sigma_1$：加噪后数据"膨胀"到整个空间，低密度区域也有训练信号
- 小噪声 $\sigma_L$：接近真实数据分布，得分估计精确

这与 DDPM 的多步时间步异曲同工——不同时间步 $t$ 对应不同的噪声水平。

### 朗之万动力学采样

**朗之万动力学（Langevin Dynamics）** 是利用得分函数采样的经典方法：

$$x_{k+1} = x_k + \frac{\delta}{2} \nabla_x \log p(x_k) + \sqrt{\delta} \, z_k, \quad z_k \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\delta$ 是步长。直觉上：
- $\nabla_x \log p(x_k)$ 将样本推向高概率区域（上山）
- $\sqrt{\delta} z_k$ 添加随机扰动，避免陷入局部模式
- 当 $\delta \to 0$ 且步数 $\to \infty$ 时，采样分布收敛到 $p(x)$

NCSN 的采样使用**退火朗之万动力学（Annealed Langevin Dynamics）**：先用大噪声的得分做粗略采样，逐步切换到小噪声的得分做精细调整。

```python
def annealed_langevin_dynamics(score_model, sigmas, n_steps, delta):
    x = torch.randn(...)  # 随机初始化
    for sigma in sigmas:  # 从大噪声到小噪声
        for _ in range(n_steps):
            z = torch.randn_like(x)
            score = score_model(x, sigma)
            x = x + (delta / 2) * score + torch.sqrt(delta) * z
    return x
```

## 两条路线的历史和统一

| 时间 | DDPM 路线 | Score-based 路线 |
|------|----------|-----------------|
| 2015 | Sohl-Dickstein et al.（扩散概率模型） | — |
| 2019 | — | Song & Ermon（NCSN） |
| 2020 | Ho et al.（DDPM） | — |
| 2021 | — | Song et al.（Score SDE，统一两条路线） |

Song et al. (2021) 的 Score SDE 论文用随机微分方程（SDE）框架将两条路线完全统一：
- DDPM = VP-SDE（方差保持 SDE）的 Euler-Maruyama 离散化
- NCSN = VE-SDE（方差爆炸 SDE）的 Euler-Maruyama 离散化

这个统一视角将在 [SDE/ODE 统一视角](./07-sde-ode) 中详细展开。

## 小结

| 概念 | 要点 |
|------|------|
| 得分函数 | $\nabla_x \log p(x)$，指向数据密度增大的方向 |
| 得分匹配 | 训练网络估计得分函数 |
| 去噪得分匹配 | 匹配加噪数据的得分 ≡ 预测噪声/去噪方向 |
| 与 DDPM 等价 | $s_\theta = -\epsilon_\theta / \sqrt{1-\bar{\alpha}_t}$ |
| NCSN | 多噪声水平的得分网络 + 退火朗之万动力学 |
| 朗之万动力学 | 沿得分方向迭代采样 + 随机扰动 |

---

> **下一篇**：[SDE/ODE 统一视角](./07-sde-ode) — 用连续时间的随机/常微分方程统一 DDPM 和 Score-based 模型。

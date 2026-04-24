---
title: "反向过程与训练目标"
date: 2026-04-17T17:41:20.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 反向过程与训练目标

> ⚙️ 进阶 | 前置知识：[前向过程详解](./02-forward-process)，了解封闭解公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

## 核心问题

前向过程将数据变成噪声——反向过程需要把噪声变回数据。数学上，我们需要求解：

$$q(x_{t-1} | x_t) = \text{?}$$

即给定一个带噪声的 $x_t$，推断它在前一时刻 $x_{t-1}$ 应该长什么样。

**难点**：直接计算 $q(x_{t-1} | x_t)$ 需要知道整个数据分布 $q(x_0)$——这正是我们试图学习的东西。这是一个"鸡生蛋"的问题。

## 贝叶斯反转：已知 $x_0$ 时的反向过程

🔬 深入

虽然 $q(x_{t-1} | x_t)$ 难以直接计算，但如果我们**同时知道原始数据** $x_0$，那么**后验分布（Posterior Distribution）** $q(x_{t-1} | x_t, x_0)$ 是可以精确计算的。

利用贝叶斯定理：

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

由于马尔可夫性质，$q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})$。而三个分布都是高斯分布（前向过程保证），所以它们的乘积/商也是高斯分布。

经过代数运算（展开高斯分布的指数部分，配方），得到：

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$

其中**后验均值**和**后验方差**为：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

**直觉理解**：后验均值 $\tilde{\mu}_t$ 是 $x_0$ 和 $x_t$ 的加权平均——它在"估计的原始数据"和"当前噪声数据"之间做插值。后验方差 $\tilde{\beta}_t$ 是确定的，只依赖于噪声调度参数。

## 训练目标：让神经网络近似反向过程

在实际生成时我们不知道 $x_0$，所以用一个参数化的神经网络 $p_\theta$ 来近似反向过程：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

训练目标来自**变分下界（Variational Lower Bound, VLB，也叫 ELBO）**——最大化数据的对数似然的下界。分解后，VLB 变成一系列 **KL 散度（KL Divergence，衡量两个概率分布之间差异的度量）**的和：

$$L_{\text{VLB}} = \sum_{t=2}^{T} \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{D_{\text{KL}}(q(x_T|x_0) \| p(x_T))}_{L_T} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}$$

核心项 $L_{t-1}$ 要求我们的模型 $p_\theta$ 逼近真实后验 $q(x_{t-1}|x_t,x_0)$。由于两者都是高斯分布，$D_{\text{KL}}$ 有解析解，最终归结为让模型预测的均值 $\mu_\theta$ 逼近后验均值 $\tilde{\mu}_t$。

## 三种参数化方式

后验均值 $\tilde{\mu}_t$ 同时依赖于 $x_0$ 和 $x_t$，而 $x_0$ 在生成时是未知的。利用封闭解关系 $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon) / \sqrt{\bar{\alpha}_t}$，我们可以选择让网络预测不同的"目标量"。

### $\epsilon$-预测（预测噪声） ⭐ 最常用

让网络 $\epsilon_\theta(x_t, t)$ 预测加入的噪声 $\epsilon$，然后通过封闭解反推 $x_0$。

将 $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon) / \sqrt{\bar{\alpha}_t}$ 代入后验均值：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$

因此模型预测的均值为：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

**简化训练损失**（Ho et al., DDPM 论文的关键发现）：

$$\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]}$$

这就是一个简单的**均方误差（MSE）** ——预测噪声与真实噪声之间的差距。DDPM 论文发现，这个简化损失虽然丢弃了 VLB 中的权重系数，但实际效果更好。

### $x_0$-预测（预测干净数据）

让网络 $x_{0,\theta}(x_t, t)$ 直接预测原始干净数据 $x_0$。

$$L_{x_0} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| x_0 - x_{0,\theta}(x_t, t) \|^2 \right]$$

**特点**：
- 在低噪声水平（小 $t$）时效果好，因为 $x_t$ 接近 $x_0$
- 在高噪声水平（大 $t$）时困难，因为 $x_t$ 几乎是纯噪声，直接预测 $x_0$ 不稳定
- 常用于 Latent Diffusion 和某些条件生成场景

### $v$-预测（速度预测）

**$v$-参数化（Velocity Parameterization）** 由 Salimans & Ho（2022）提出，定义：

$$v_t = \sqrt{\bar{\alpha}_t} \, \epsilon - \sqrt{1-\bar{\alpha}_t} \, x_0$$

让网络预测这个"速度" $v_t$，可以同时恢复 $\epsilon$ 和 $x_0$：
- $\hat{\epsilon} = \sqrt{\bar{\alpha}_t} \, x_t + \sqrt{1-\bar{\alpha}_t} \, v_\theta$
- $\hat{x}_0 = \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{1-\bar{\alpha}_t} \, v_\theta$

**特点**：
- 在 SNR 的两个极端都比较稳定
- Stable Diffusion 2.x、Imagen Video 等采用了 $v$-预测
- 名字来源于概率流 ODE 中的"速度场"概念

### 三种参数化对比

| 参数化 | 网络输出 | 优势 | 劣势 | 典型应用 |
|--------|---------|------|------|---------|
| $\epsilon$-预测 | 噪声 $\epsilon$ | 简单、效果好、最广泛使用 | 高 SNR 时梯度消失 | DDPM, SD 1.x |
| $x_0$-预测 | 干净数据 $x_0$ | 直觉直接 | 低 SNR 时不稳定 | 部分 LDM |
| $v$-预测 | 速度 $v$ | 两端稳定、理论优雅 | 相对较新、生态略少 | SD 2.x, Imagen |

⚙️ 它们在数学上是**等价的**——给定一个预测，可以通过封闭解关系推导出其他两个。差异仅在于训练时的梯度分布和数值稳定性。

## 完整训练算法

```python
# DDPM 训练算法（ε-预测）
def train_step(model, x_0, alpha_bars, T):
    # 1. 随机采样时间步
    t = torch.randint(0, T, (batch_size,))
    
    # 2. 采样随机噪声
    epsilon = torch.randn_like(x_0)
    
    # 3. 用封闭解构造 x_t
    a_bar = alpha_bars[t].reshape(-1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * epsilon
    
    # 4. 让模型预测噪声
    epsilon_pred = model(x_t, t)
    
    # 5. 计算简化 MSE 损失
    loss = F.mse_loss(epsilon_pred, epsilon)
    
    return loss
```

**训练流程总结**：
1. 从数据集采样一批干净数据 $x_0$
2. 对每个样本随机抽一个时间步 $t \sim \text{Uniform}(1, T)$
3. 采样噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
4. 用封闭解计算 $x_t$
5. 网络接收 $(x_t, t)$，输出噪声预测 $\epsilon_\theta$
6. 损失 = $\|\epsilon - \epsilon_\theta\|^2$
7. 反向传播，更新参数

## 方差的选择

⚙️ 进阶

在模型预测的反向分布 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ 中，方差 $\Sigma_\theta$ 有几种选择：

1. **固定方差**（DDPM 原始设置）：$\Sigma = \beta_t \mathbf{I}$ 或 $\Sigma = \tilde{\beta}_t \mathbf{I}$
   - 两者分别对应上界和下界
   - 实践中差异不大，DDPM 使用 $\beta_t$

2. **可学习方差**（Improved DDPM）：$\Sigma_\theta = \exp(v \log \beta_t + (1-v) \log \tilde{\beta}_t)$
   - 网络额外输出一个插值系数 $v$
   - 可以改善对数似然（但 FID 改善有限）

3. **DDIM**：方差由超参 $\eta$ 控制，$\eta=0$ 时方差为零（确定性采样）

## ELBO 分解的完整图景

🔬 深入

回顾 VLB 的三部分：

$$L = L_T + \sum_{t=2}^{T} L_{t-1} + L_0$$

| 项 | 公式 | 含义 |
|----|------|------|
| $L_T$ | $D_{\text{KL}}(q(x_T\|x_0) \| p(x_T))$ | 终端分布匹配，无可学习参数，可忽略 |
| $L_{t-1}$ | $D_{\text{KL}}(q(x_{t-1}\|x_t,x_0) \| p_\theta(x_{t-1}\|x_t))$ | 去噪匹配，核心训练项 |
| $L_0$ | $-\log p_\theta(x_0\|x_1)$ | 重建项，最后一步的解码质量 |

DDPM 的简化损失 $L_{\text{simple}}$ 等价于对 $L_{t-1}$ 去掉了依赖于 $t$ 的权重系数 $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$。去掉权重后，高噪声时间步（大 $t$，权重本应较大）和低噪声时间步被平等对待，实验中这反而有更好的样本质量（虽然理论上牺牲了似然）。

## 小结

| 概念 | 要点 |
|------|------|
| 后验分布 | $q(x_{t-1}\|x_t,x_0)$ 是高斯分布，均值和方差可精确计算 |
| 训练思路 | 用神经网络近似后验均值，通过 VLB/简化 MSE 训练 |
| $\epsilon$-预测 | 最常用，预测噪声 → 反推均值，损失为简单 MSE |
| $x_0$-预测 | 直接预测干净数据，部分场景使用 |
| $v$-预测 | 速度参数化，数值稳定性好 |
| 简化损失 | $\|\epsilon - \epsilon_\theta\|^2$，去掉权重系数，效果更好 |

---

> **下一篇**：[采样算法](./04-sampling-algorithms) — 如何用训练好的模型从噪声生成数据，以及不同采样器的特性。

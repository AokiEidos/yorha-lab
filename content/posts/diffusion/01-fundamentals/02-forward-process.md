---
title: "前向过程详解"
date: 2026-04-17T17:39:59.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 前向过程详解

> 🔰 入门 → ⚙️ 进阶 | 前置知识：[扩散模型核心直觉](./01-core-intuition)，基本概率论（高斯分布、条件概率）

## 核心思想回顾

前向过程（Forward Process）是扩散模型中"不需要学习"的那一半——它只是一个预定义的加噪过程，将数据逐步变成纯噪声。数学上，它被定义为一条**马尔可夫链（Markov Chain，即每一步的状态只依赖前一步，与更早的历史无关）**。

## 逐步加噪公式

### 单步加噪

给定时刻 $t-1$ 的数据 $x_{t-1}$，时刻 $t$ 的数据通过添加高斯噪声得到：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \, \beta_t \mathbf{I})$$

其中：
- $q$ 表示前向过程的分布（不含可学习参数）
- $\mathcal{N}(\mu, \sigma^2)$ 表示均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布
- $\beta_t$ 是第 $t$ 步的**噪声系数（Noise Coefficient）**，是一个预先设定的小正数（通常 $\beta_t \in [0.0001, 0.02]$）
- $\mathbf{I}$ 是单位矩阵（表示每个维度独立加噪）

**直觉理解**：这个公式做了两件事：
1. **缩小信号**：将 $x_{t-1}$ 乘以 $\sqrt{1-\beta_t}$（略小于 1），使信号稍微衰减
2. **添加噪声**：加上方差为 $\beta_t$ 的高斯噪声

这就像用一个略微模糊的复印机复印一张照片——每复印一次，照片就变模糊一点点，最终变成纯噪点。

### 等价的采样形式

上面的分布可以用**重参数化技巧（Reparameterization Trick，即用 $x = \mu + \sigma \cdot \epsilon$ 的形式从高斯分布采样，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$）** 写成：

$$x_t = \sqrt{1-\beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

这是实际编程中使用的形式——一次乘法、一次加法。

## 关键参数定义

为了后续推导方便，定义以下参数：

| 符号 | 定义 | 含义 |
|------|------|------|
| $\beta_t$ | 预设的噪声系数 | 第 $t$ 步添加的噪声量 |
| $\alpha_t$ | $1 - \beta_t$ | 第 $t$ 步保留的信号比例 |
| $\bar{\alpha}_t$ | $\prod_{s=1}^{t} \alpha_s$ | 从第 1 步到第 $t$ 步的累积信号保留比例 |

注意 $\bar{\alpha}_t$（读作 "alpha bar t"）是所有 $\alpha_s$ 的**累积乘积**。由于每个 $\alpha_s < 1$，所以 $\bar{\alpha}_t$ 随 $t$ 增大而单调递减，从接近 1 递减到接近 0。

**直觉**：$\bar{\alpha}_t$ 衡量"到时刻 $t$ 时，原始信号还剩下多少"。
- $t = 0$：$\bar{\alpha}_0 = 1$，信号完整保留
- $t = T$：$\bar{\alpha}_T \approx 0$，信号几乎完全消失

## 封闭解：一步到位

⚙️ 进阶

前向过程的一个关键优势是：我们不需要逐步加噪 $T$ 次，而是可以从 $x_0$ 直接跳到任意时刻 $x_t$。

### 推导过程

从 $x_1$ 开始展开：

$$x_1 = \sqrt{\alpha_1} \, x_0 + \sqrt{1-\alpha_1} \, \epsilon_1$$

$$x_2 = \sqrt{\alpha_2} \, x_1 + \sqrt{1-\alpha_2} \, \epsilon_2$$

将 $x_1$ 代入 $x_2$：

$$x_2 = \sqrt{\alpha_2} (\sqrt{\alpha_1} \, x_0 + \sqrt{1-\alpha_1} \, \epsilon_1) + \sqrt{1-\alpha_2} \, \epsilon_2$$

$$= \sqrt{\alpha_1 \alpha_2} \, x_0 + \sqrt{\alpha_2(1-\alpha_1)} \, \epsilon_1 + \sqrt{1-\alpha_2} \, \epsilon_2$$

由于两个独立高斯噪声的和仍然是高斯噪声（方差相加），合并后方差为：

$$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$$

因此：

$$x_2 = \sqrt{\bar{\alpha}_2} \, x_0 + \sqrt{1-\bar{\alpha}_2} \, \bar{\epsilon}_2, \quad \bar{\epsilon}_2 \sim \mathcal{N}(0, \mathbf{I})$$

### 一般形式（封闭解）

通过**数学归纳法（Mathematical Induction）** 推广到任意时刻 $t$：

$$\boxed{q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1-\bar{\alpha}_t) \mathbf{I})}$$

等价的采样形式：

$$\boxed{x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})}$$

**这就是前向过程最重要的公式。** 它意味着：
- 我们可以从 $x_0$ **一步**到达 $x_t$，无需逐步模拟
- $x_t$ 是原始数据 $\sqrt{\bar{\alpha}_t} \, x_0$ 和噪声 $\sqrt{1-\bar{\alpha}_t} \, \epsilon$ 的加权混合
- 训练时，只需对 $t$ 和 $\epsilon$ 采样，即可高效地构造任意噪声水平的训练样本

### 信噪比视角

从封闭解可以看出，$x_t$ 中：
- 信号部分的系数：$\sqrt{\bar{\alpha}_t}$，功率为 $\bar{\alpha}_t$
- 噪声部分的系数：$\sqrt{1-\bar{\alpha}_t}$，功率为 $1 - \bar{\alpha}_t$

因此**信噪比（Signal-to-Noise Ratio, SNR）** 为：

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- $t \to 0$：$\text{SNR} \to \infty$（纯信号）
- $t \to T$：$\text{SNR} \to 0$（纯噪声）

SNR 提供了一个与具体调度策略无关的统一视角来理解噪声水平（详见 [噪声调度](./05-noise-schedule)）。

## 伪代码

### 训练时构造噪声样本

```python
def forward_diffusion_sample(x_0, t, alpha_bar):
    """
    从 x_0 直接采样 x_t（用于训练）
    x_0: 干净数据, shape [B, C, H, W]
    t: 时间步, shape [B]
    alpha_bar: 预计算的 ᾱ_t 序列, shape [T]
    """
    # 取出当前时间步的 ᾱ_t
    a_bar = alpha_bar[t]  # shape [B, 1, 1, 1]
    
    # 采样随机噪声
    epsilon = torch.randn_like(x_0)
    
    # 封闭解公式
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * epsilon
    
    return x_t, epsilon  # 返回噪声样本和真实噪声（用于计算损失）
```

### 预计算噪声调度参数

```python
def prepare_schedule(beta_start=0.0001, beta_end=0.02, T=1000):
    """预计算所有时间步的参数"""
    betas = torch.linspace(beta_start, beta_end, T)  # 线性调度
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # 累积乘积
    return betas, alphas, alpha_bars
```

## 可视化理解

想象一张清晰的猫咪照片经过前向过程：

```
t=0        t=250      t=500      t=750      t=1000
[清晰猫咪] → [略有噪声] → [模糊轮廓] → [几乎噪声] → [纯噪声]
SNR=∞      SNR≈10     SNR≈1      SNR≈0.1    SNR≈0

ᾱ_t ≈ 1.0   ᾱ_t ≈ 0.9  ᾱ_t ≈ 0.5  ᾱ_t ≈ 0.1  ᾱ_t ≈ 0.0
```

关键观察：
- $\bar{\alpha}_t \approx 0.5$ 是信号和噪声势均力敌的临界点
- 前半段（$t$ 小时）噪声增长缓慢，信号衰减温和
- 后半段（$t$ 大时）噪声快速占主导，信号迅速消失
- 这种"S 型曲线"特征对噪声调度的设计有重要启示

## 为什么前向过程如此重要

虽然前向过程本身不含可学习参数，但它为扩散模型奠定了以下基础：

1. **训练效率**：封闭解让训练时可以对 $(t, \epsilon)$ 均匀采样，无需逐步模拟
2. **数学框架**：前向过程的高斯性质使得反向过程也具有（近似）高斯形式，大大简化了数学推导
3. **终端分布**：确保 $x_T$ 足够接近 $\mathcal{N}(0, \mathbf{I})$，这是反向过程的起点
4. **噪声调度设计**：$\beta_t$ 的选择直接影响信号衰减速率和生成质量

## 小结

| 概念 | 公式/含义 |
|------|----------|
| 单步加噪 | $x_t = \sqrt{1-\beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t$ |
| 封闭解 | $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon$ |
| $\alpha_t$ | $1 - \beta_t$，单步信号保留比例 |
| $\bar{\alpha}_t$ | $\prod \alpha_s$，累积信号保留比例 |
| SNR | $\bar{\alpha}_t / (1-\bar{\alpha}_t)$，信噪比 |

---

> **下一篇**：[反向过程与训练目标](./03-reverse-process) — 如何训练神经网络逆转加噪过程，以及三种主要的参数化方式。

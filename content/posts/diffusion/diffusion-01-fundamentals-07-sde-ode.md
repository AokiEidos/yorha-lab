---
title: "SDE/ODE 统一视角"
date: 2026-04-17T17:45:26.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# SDE/ODE 统一视角

> 🔬 深入 | 前置知识：[得分函数与得分匹配](./06-score-matching.md)，基本微分方程概念

## 从离散到连续

前面几篇文档中，我们使用离散的时间步 $t = 0, 1, \ldots, T$。当 $T \to \infty$（步数趋于无穷），步长 $\Delta t \to 0$，离散过程趋近于连续时间过程。

**连续时间框架**的好处：
- 统一 DDPM 和 Score-based 模型这两条看似不同的路线
- 提供更灵活的采样器设计空间（不受离散步数限制）
- 引入概率流 ODE，实现确定性采样和精确的似然计算

## 随机微分方程（SDE）

### 什么是 SDE

**随机微分方程（Stochastic Differential Equation, SDE）** 描述一个受确定性力和随机扰动共同影响的连续时间过程：

$$dx = f(x, t) \, dt + g(t) \, dw$$

其中：
- $f(x, t)$：**漂移系数（Drift Coefficient）**，确定性趋势（像河流的水流方向）
- $g(t)$：**扩散系数（Diffusion Coefficient）**，随机扰动的强度（像水面的波浪大小）
- $dw$：**维纳过程（Wiener Process，也叫布朗运动）** 的微分——数学上对"连续时间白噪声"的严格定义

**直觉**：想象一个醉汉沿着河流漂流——河流的方向是漂移（确定性部分），醉汉的摇晃是扩散（随机部分）。

### 前向 SDE

扩散模型的前向过程可以统一写成 SDE 形式：

$$dx = f(x, t) \, dt + g(t) \, dw$$

不同的扩散模型对应不同的 $f$ 和 $g$。

### 反向 SDE

Anderson（1982）证明了一个深刻的结果：给定前向 SDE，其**反向时间 SDE** 为：

$$\boxed{dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) \, d\bar{w}}$$

其中：
- $d\bar{w}$ 是反向时间的维纳过程
- $\nabla_x \log p_t(x)$ 正是时刻 $t$ 的**得分函数**

**关键洞察**：如果我们有一个训练好的得分网络 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$，就可以模拟反向 SDE，从噪声生成数据！

## 三种经典 SDE

Song et al.（2021）定义了三种 SDE，统一了已有的扩散模型：

### VP-SDE（Variance Preserving SDE，方差保持 SDE）

$$dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dw$$

- $f(x,t) = -\frac{1}{2}\beta(t)x$（线性漂移，将数据推向零）
- $g(t) = \sqrt{\beta(t)}$
- **DDPM 是 VP-SDE 的 Euler-Maruyama 离散化**
- "方差保持"指过程中 $x_t$ 的总方差大致不变（信号衰减的同时噪声增加）

### VE-SDE（Variance Exploding SDE，方差爆炸 SDE）

$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dw$$

- $f(x,t) = 0$（无漂移）
- $g(t) = \sigma(t)\sqrt{2\dot{\sigma}(t)/\sigma(t)}$
- **NCSN/SMLD 是 VE-SDE 的离散化**
- "方差爆炸"指噪声方差 $\sigma^2(t)$ 随时间持续增长

### sub-VP-SDE

$$dx = -\frac{1}{2}\beta(t)x \, dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})} \, dw$$

- VP-SDE 的变体，理论上可获得更好的似然
- 实践中较少使用

### 对比

| SDE 类型 | 漂移 $f$ | 扩散 $g$ | 离散化 | 边际方差 |
|---------|---------|---------|--------|---------|
| VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | DDPM | ≈ 1（保持） |
| VE-SDE | 0 | $\propto \dot{\sigma}$ | NCSN | $\sigma^2(t)$（增长） |
| sub-VP | 同 VP | 修正的 $g(t)$ | — | ≤ VP |

## 概率流 ODE

### 从 SDE 到 ODE

Song et al. 证明了另一个关键结果：对于任何 SDE，存在一个**确定性**的**常微分方程（Ordinary Differential Equation, ODE）**——**概率流 ODE（Probability Flow ODE）**——它产生与 SDE 完全相同的边际分布 $p_t(x)$：

$$\boxed{\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)}$$

对比反向 SDE：$dx = [f - g^2 \nabla_x \log p_t] dt + g \, d\bar{w}$

概率流 ODE 就是反向 SDE 去掉随机项后，漂移项减半的结果。

### 概率流 ODE 的重要性

1. **确定性采样**：同一个初始噪声 $x_T$ 总是映射到同一个 $x_0$——这对图像编辑（DDIM Inversion）至关重要

2. **精确似然计算**：通过**瞬时变量变换公式（Instantaneous Change of Variables）** 可以精确计算 $\log p(x_0)$：
   $$\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde{f}(x_t, t) \, dt$$
   其中 $\tilde{f}$ 是概率流 ODE 的漂移，$\nabla \cdot$ 是散度

3. **潜空间插值**：$x_T$ 空间中的线性插值对应数据空间中的平滑语义插值

4. **灵活采样步数**：ODE 可以用各种数值求解器（Euler、Heun、RK45 等）以不同精度求解，天然支持不同步数的采样

### DDIM 是概率流 ODE 的离散化

VP-SDE 对应的概率流 ODE 在 Euler 离散化后，恰好等价于 DDIM（$\eta=0$）的采样公式。这解释了 DDIM 的确定性采样为什么有效，以及为什么 DDIM 可以用更少的步数采样。

## 数值求解

### Euler-Maruyama（SDE 求解器）

SDE 的最简单数值方法：

$$x_{t+\Delta t} = x_t + f(x_t, t)\Delta t + g(t)\sqrt{\Delta t} \, z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

这就是 DDPM 采样的底层原理——每步采样 = Euler-Maruyama 一步。

### Euler（ODE 求解器）

$$x_{t+\Delta t} = x_t + f_{\text{ODE}}(x_t, t)\Delta t$$

无随机项，这就是 DDIM 采样的底层原理。

### 高阶求解器

| 求解器 | 阶数 | 精度 | 每步网络调用 |
|--------|------|------|------------|
| Euler | 1 阶 | $O(\Delta t)$ | 1 次 |
| Heun（改进 Euler） | 2 阶 | $O(\Delta t^2)$ | 2 次 |
| RK45 | 4-5 阶 | 自适应 | 多次 |
| DPM-Solver | 专用高阶 | $O(\Delta t^3)$ | 1 次/步 |

更高阶的求解器允许用更少的步数达到相同精度，这是 DPM-Solver 等加速方法的理论基础（详见 [加速与部署](../04-acceleration-deployment/) 模块）。

## 统一图景

```
                    连续时间 SDE 框架
                   ╱                ╲
           前向 SDE                反向 SDE
          (加噪过程)              (去噪过程)
               │                      │
    ┌──────────┼──────────┐    ┌──────┴──────┐
    │          │          │    │             │
  VP-SDE    VE-SDE   sub-VP  反向 SDE   概率流 ODE
    │          │              (随机)     (确定性)
    │          │               │           │
  离散化      离散化          离散化       离散化
    │          │               │           │
  DDPM       NCSN          祖先采样     DDIM(η=0)
                                      DPM-Solver
```

**核心信息**：DDPM 和 NCSN 是同一个连续时间框架（SDE）的两种不同离散化。它们的训练目标等价（都是去噪得分匹配），只是采样过程和噪声调度不同。

## 小结

| 概念 | 要点 |
|------|------|
| SDE | 连续时间的加噪过程，$dx = f \, dt + g \, dw$ |
| 反向 SDE | 需要得分函数 $\nabla_x \log p_t(x)$ |
| VP-SDE / VE-SDE | 对应 DDPM / NCSN 的连续版本 |
| 概率流 ODE | 去掉随机项的确定性版本，同边际分布 |
| 确定性采样 | ODE → DDIM，支持可逆映射和精确似然 |
| 高阶求解器 | 更少步数 + 更高精度，加速采样的理论基础 |

---

> **下一篇**：[DDPM 详解](../02-models-zoo/01-ddpm.md) — 进入模块二，详细解析 DDPM 的 U-Net 架构和完整实现。

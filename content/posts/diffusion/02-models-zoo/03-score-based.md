---
title: "Score-based Generative Models 详解"
date: 2026-04-17T17:48:31.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Score-based Generative Models 详解

> 🔬 深入 | 前置知识：[得分函数与得分匹配](../01-fundamentals/06-score-matching)，[SDE/ODE 统一视角](../01-fundamentals/07-sde-ode)

## 两条路线的交汇

在模块一中我们已介绍，扩散模型有两条历史路线：
1. **DDPM 路线**（概率视角）：最大化数据似然的变分下界
2. **Score-based 路线**（得分匹配视角）：学习数据分布的梯度场

本文详解 Score-based 路线的代表性工作，并展示两条路线如何被 SDE 框架统一。

## NCSN（Noise Conditional Score Network）

### 核心问题：为什么单一噪声水平不够

直接学习 $\nabla_x \log p(x)$ 面临两个困难：

1. **低密度区域**：数据分布的"沙漠地带"几乎没有训练样本，得分估计不准确。但采样过程恰恰需要从这些区域出发。
2. **流形问题**：如果数据集中在低维流形上，流形外的得分没有定义。

### NCSN 的解决方案

Song & Ermon（2019）提出使用一系列递减的噪声水平 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$：

- **大 $\sigma$**：加噪后数据"膨胀"到整个空间，覆盖低密度区域
- **小 $\sigma$**：接近真实分布，得分估计精确

训练一个**条件得分网络** $s_\theta(x, \sigma)$，联合估计所有噪声水平的得分：

$$L_{\text{NCSN}} = \sum_{i=1}^{L} \lambda(\sigma_i) \mathbb{E}_{x \sim p(x)} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)} \left[ \left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2} \right\|^2 \right]$$

其中 $\lambda(\sigma_i) = \sigma_i^2$ 是权重系数，使不同噪声水平的损失量级一致。

### 退火朗之万动力学采样

采样时，从大噪声到小噪声逐级使用朗之万动力学：

```
算法：退火朗之万动力学（Annealed Langevin Dynamics）
1. 初始化 x ~ N(0, σ₁²I) 或均匀分布
2. for i = 1, 2, ..., L:        # 从大噪声到小噪声
3.     步长 δᵢ = ε · σᵢ² / σ_L²  # 步长与噪声水平成正比
4.     for k = 1, ..., K:         # 每个噪声水平跑 K 步
5.         z ~ N(0, I)
6.         x = x + (δᵢ/2) · sθ(x, σᵢ) + √δᵢ · z
7. return x
```

直觉：像雕刻一样——先用大锤敲出大致形状（大 $\sigma$），再用小锤精细打磨（小 $\sigma$）。

## Song et al. SDE 统一框架

Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole（2021）的里程碑论文用 SDE 框架将两条路线完全统一。

### 统一映射

| 离散模型 | 对应的连续 SDE | 漂移 $f(x,t)$ | 扩散 $g(t)$ |
|---------|-------------|-------------|-----------|
| DDPM | VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ |
| NCSN/SMLD | VE-SDE | $0$ | $\sigma(t)\sqrt{2\dot{\sigma}(t)/\sigma(t)}$ |

SMLD = Score Matching with Langevin Dynamics，是 NCSN 的另一个名字。

### 统一训练

两种 SDE 的训练目标都等价于：

$$L = \mathbb{E}_{t \sim \mathcal{U}(0,T)} \mathbb{E}_{x_0 \sim p_0} \mathbb{E}_{x_t \sim q(x_t|x_0)} \left[ \lambda(t) \| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0) \|^2 \right]$$

其中 $\lambda(t)$ 是权重函数。

### 统一采样

训练好得分网络后，有三种采样策略：

1. **反向 SDE 采样**：模拟反向 SDE（含随机项），类似 DDPM 祖先采样
2. **概率流 ODE 采样**：求解概率流 ODE（无随机项），类似 DDIM
3. **预测-校正采样器（Predictor-Corrector, PC Sampler）**：交替使用
   - **预测步（Predictor）**：SDE/ODE 数值积分一步
   - **校正步（Corrector）**：在当前噪声水平做几步朗之万动力学，精炼得分估计

PC 采样器在 Song et al. 的实验中取得了当时最好的 FID。

```python
def pc_sampler(score_model, sde, shape, n_steps, n_corrector_steps):
    x = sde.prior_sampling(shape)  # 从先验分布采样
    timesteps = torch.linspace(sde.T, 1e-3, n_steps)
    
    for i in range(n_steps - 1):
        t = timesteps[i]
        dt = timesteps[i+1] - t
        
        # Predictor: 反向 SDE 的 Euler-Maruyama 一步
        score = score_model(x, t)
        drift = sde.reverse_drift(x, t, score)
        diffusion = sde.diffusion(t)
        x = x + drift * dt + diffusion * torch.sqrt(-dt) * torch.randn_like(x)
        
        # Corrector: 朗之万动力学几步
        for _ in range(n_corrector_steps):
            score = score_model(x, timesteps[i+1])
            noise = torch.randn_like(x)
            step_size = compute_step_size(score, noise)
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    
    return x
```

## 两条路线的关系图

```
2015: Sohl-Dickstein et al.
  │   (扩散概率模型雏形)
  │
  ├──────── DDPM 路线 ────────┐
  │                           │
  │  2019: Song & Ermon       │  2020: Ho et al.
  │  (NCSN, 多尺度得分匹配)    │  (DDPM, 简化训练)
  │         │                 │       │
  │         │                 │  2020: Song et al.
  │         │                 │  (DDIM, 确定性采样)
  │         │                 │       │
  ├─── Score 路线 ────────────┤       │
  │                           │       │
  │     2021: Song et al.     │       │
  │     (Score SDE) ←─────────┴───────┘
  │     统一两条路线
  │         │
  │    ┌────┴────┐
  │  VP-SDE    VE-SDE
  │  (=DDPM)   (=NCSN)
  │    │
  │  概率流 ODE
  │  (=DDIM)
```

## NCSN++ 与架构改进

在 Score SDE 论文中，作者还提出了改进的架构 NCSN++/DDPM++：

- **更大的网络**：增加通道数和注意力层
- **渐进式增长**：逐步增加分辨率训练
- **FIR 滤波器**：用于上下采样，减少伪影
- **Skip Scaling**：缩放跳跃连接，改善高分辨率稳定性

NCSN++ 在 CIFAR-10 上实现了 FID 2.20（无条件生成），显著超越了 DDPM 和之前的所有方法。

## 小结

| 概念 | 要点 |
|------|------|
| NCSN | 多噪声水平得分网络 + 退火朗之万采样 |
| 退火采样 | 从大噪声到小噪声逐级精炼 |
| Score SDE | 用 SDE 统一 DDPM(VP-SDE) 和 NCSN(VE-SDE) |
| PC 采样器 | 预测步(SDE/ODE) + 校正步(朗之万)，交替进行 |
| 核心洞察 | 噪声预测 ≡ 得分估计 ≡ 去噪方向，三者等价 |

---

> **下一篇**：[Latent Diffusion 与 Stable Diffusion](./04-latent-diffusion)

---
title: "DDIM 详解"
date: 2026-04-17T17:47:38.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# DDIM 详解

> ⚙️ 进阶 | 前置知识：[DDPM 详解](./01-ddpm)，[SDE/ODE 统一视角](../01-fundamentals/07-sde-ode)

## 核心动机

DDPM 生成一张图需要 1000 步——太慢了。DDIM（Denoising Diffusion Implicit Models，去噪扩散隐式模型）（Song et al., 2020b）在**不重新训练**的前提下，将采样步数降到 50-100 步，同时引入了确定性采样这一重要能力。

关键洞察：DDPM 的训练目标只约束了边际分布 $q(x_t|x_0)$，并不要求前向过程必须是马尔可夫链。DDIM 构造了一族非马尔可夫的前向过程，它们共享相同的边际分布（因此可以复用 DDPM 训练好的模型），但拥有不同的采样特性。

## 非马尔可夫前向过程

### DDPM vs DDIM 的前向过程

**DDPM**（马尔可夫）：$q(x_{t-1}|x_t, x_0)$ 是固定的后验分布

**DDIM**（非马尔可夫）：构造一族新的后验分布 $q_\sigma(x_{t-1}|x_t, x_0)$

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \; \sigma_t^2 \mathbf{I}\right)$$

这个分布族由参数 $\sigma_t$ 控制：
- $\sigma_t = \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t}$ 时，退化为 DDPM
- $\sigma_t = 0$ 时，得到 DDIM（确定性采样）

### $\eta$ 参数

实践中用 $\eta \in [0, 1]$ 统一控制：

$$\sigma_t = \eta \sqrt{\frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t}$$

| $\eta$ | 含义 |
|--------|------|
| 0 | DDIM（纯确定性） |
| 1 | DDPM（标准随机采样） |
| (0, 1) | 介于两者之间的半随机采样 |

## DDIM 采样公式

将噪声预测 $\epsilon_\theta(x_t, t)$ 代入（利用 $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta) / \sqrt{\bar{\alpha}_t}$），得到 DDIM 的核心采样公式：

$$\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 } x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta}_{\text{指向 } x_t \text{ 的方向}} + \underbrace{\sigma_t \epsilon_t}_{\text{随机噪声}}}$$

当 $\sigma_t = 0$（$\eta = 0$）时，最后一项消失，采样变成完全确定性的。

### 三项的直觉

1. **预测的 $x_0$**：用当前的噪声预测反推出干净数据估计
2. **指向 $x_t$ 的方向**：保留一部分当前噪声的方向信息
3. **随机噪声**：额外的随机扰动（DDIM 中为零）

## 子序列采样（Subsequence Sampling）

DDIM 最实用的特性是**子序列采样**：不需要遍历 $\{T, T-1, \ldots, 1\}$ 全部 1000 步，可以选一个子序列。

例如，选择 $S = 50$ 步的等距子序列：$\tau = \{1, 21, 41, \ldots, 981\}$

```python
def ddim_sample(model, shape, alpha_bars, timesteps, eta=0.0):
    """
    timesteps: 子序列，如 [999, 979, 959, ..., 19, 0]
    """
    x = torch.randn(shape)
    
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        # 网络预测噪声
        eps = model(x, t)
        
        # 预测 x_0
        a_t = alpha_bars[t]
        a_prev = alpha_bars[t_prev]
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        
        # 计算方向
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma**2) * eps
        
        # DDIM 更新
        noise = sigma * torch.randn_like(x) if sigma > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    
    return x
```

### 步数与质量的关系

| 步数 $S$ | DDPM (η=1) FID | DDIM (η=0) FID | 速度提升 |
|----------|---------------|---------------|---------|
| 1000 | 3.17 | ~3.3 | 1× |
| 100 | 很差 | ~4.2 | 10× |
| 50 | 极差 | ~4.7 | 20× |
| 20 | 不可用 | ~6.8 | 50× |
| 10 | 不可用 | ~13.4 | 100× |

DDIM 在 50-100 步时仍能保持合理质量，而 DDPM 在减少步数时质量急剧下降。

## 确定性采样的意义

当 $\eta = 0$ 时，DDIM 采样是确定性的：相同的初始噪声 $x_T$ 总是生成相同的 $x_0$。这建立了噪声空间与数据空间之间的双射映射，带来几个重要应用。

### 语义插值

在噪声空间做线性插值，对应数据空间的平滑语义变化：

$$x_T^{(\lambda)} = (1-\lambda) x_T^{(A)} + \lambda x_T^{(B)}$$

对 $x_T^{(\lambda)}$ 做 DDIM 采样，得到的图像在 $A$ 和 $B$ 之间平滑过渡。

### DDIM Inversion（反演）

确定性采样是可逆的——给定一张真实图像 $x_0$，可以用 ODE 的反向（将去噪过程反过来做）推回噪声 $x_T$：

```python
def ddim_inversion(model, x_0, alpha_bars, timesteps):
    """从 x_0 反推 x_T"""
    x = x_0
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]  # 往噪声方向走
        
        eps = model(x, t)
        
        a_t = alpha_bars[t]
        a_next = alpha_bars[t_next]
        
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps
    
    return x  # 得到 x_T
```

DDIM Inversion 是图像编辑的基础——先将真实图像反演到噪声空间，修改条件，再用 DDIM 采样回来，就能对真实图像进行编辑（详见 [图像编辑](../05-llm-era/04-image-editing)）。

## 与概率流 ODE 的联系

🔬 深入

DDIM（$\eta=0$）的采样公式实际上就是 VP-SDE 对应的**概率流 ODE** 的 Euler 离散化。

概率流 ODE：$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g^2(t)\nabla_x \log p_t(x)$

用 Euler 方法离散化，步长选为 DDIM 的时间子序列间距，即可恢复 DDIM 的更新公式。

这个联系意味着：
- 更高阶的 ODE 求解器（如 Heun、DPM-Solver）可以进一步提升采样质量/减少步数
- DDIM 的 Inversion 精确性受限于 Euler 方法的一阶精度
- 自适应步长 ODE 求解器（如 RK45）可以自动选择最优步数

## 小结

| 概念 | 要点 |
|------|------|
| 核心创新 | 非马尔可夫前向过程，复用 DDPM 模型 |
| $\eta$ 参数 | 0=确定性（DDIM），1=随机（DDPM） |
| 子序列采样 | 50-100 步仍有不错质量，20× 加速 |
| 确定性采样 | 同噪声→同结果，支持插值和反演 |
| DDIM Inversion | 真实图像→噪声空间，图像编辑的基础 |
| 与 ODE 联系 | DDIM = 概率流 ODE 的 Euler 离散化 |

---

> **下一篇**：[Score-based Models 详解](./03-score-based)

---
title: "Flow Matching"
date: 2026-04-20T17:15:54.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Flow Matching

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[SDE/ODE 统一视角](../01-fundamentals/07-sde-ode)，基本向量场概念

## 从扩散到流

**流匹配（Flow Matching）**（Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2022）是一种与扩散模型密切相关但更简洁的生成框架。其核心思想：学习一个**速度场（Velocity Field）**，将噪声分布"流动"到数据分布。

如果说扩散模型是"加噪 -> 去噪"，那么 Flow Matching 就是"学习从噪声到数据的流动路径"。FM 已经成为 SD3、FLUX、Sora 等下一代模型的核心范式，也是机器人 VLA 领域（如 $\pi_0$）的动作生成方法（详见 [VLA: Diffusion 作为动作生成器](../../../..//posts/vla/07-foundation-models/03-diffusion-action)）。

## 连续标准化流（Continuous Normalizing Flow, CNF）

### 基本框架

定义一个时间相关的速度场 $v_\theta(x, t)$，它驱动 ODE：

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

- $t=0$：数据分布 $p_0 = p_{\text{data}}$
- $t=1$：噪声分布 $p_1 = \mathcal{N}(0, \mathbf{I})$（或反过来，不同论文约定不同）

训练好 $v_\theta$ 后，采样就是求解这个 ODE：从 $x_1 \sim \mathcal{N}(0, \mathbf{I})$ 出发，用 ODE 求解器从 $t=1$ 积分到 $t=0$。

```
┌────────────────────────────────────────────────────────────┐
│                 Flow Matching 采样过程                       │
│                                                            │
│  t=1 (噪声)                                    t=0 (数据)  │
│     x₁ ~ N(0,I)  ─── v_θ(x,t) 驱动 ───→  x₀ ~ p_data    │
│                                                            │
│  dx/dt = v_θ(x_t, t)，用 ODE 求解器积分                     │
│  常用: Euler (1阶), Midpoint (2阶), RK4 (4阶)               │
└────────────────────────────────────────────────────────────┘
```

### 与扩散模型的联系

概率流 ODE 也是一个 CNF，其速度场：

$$v_{\text{diff}}(x, t) = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

Flow Matching 的区别在于：它不从 SDE 推导速度场，而是直接学习一个将两个分布连接的速度场。

### 数学联系：速度场与得分函数的互转

🔬 深入

给定 Flow Matching 的速度场 $v_\theta$ 和边际分布 $p_t$，可以推导出等价的得分函数：

$$\nabla_x \log p_t(x) = \frac{v_\theta(x, t) - \mathbb{E}[\dot{\sigma}_t \epsilon | x_t = x]}{\sigma_t \dot{\sigma}_t}$$

反之，给定扩散模型的得分函数 $s_\theta = \nabla_x \log p_t$，也可以推导出速度场。两者在数学上等价，差异在于训练路径的几何性质。

## 条件流匹配（Conditional Flow Matching, CFM）

### 直觉

直接训练 $v_\theta$ 匹配全局速度场需要知道 $p_t(x)$——但这是未知的。

**条件流匹配（Conditional Flow Matching, CFM）** 的技巧：定义每个数据点 $x_0$ 对应的**条件路径** $x_t = \phi_t(x_0, \epsilon)$，训练 $v_\theta$ 匹配条件速度。

最简单的条件路径是**线性插值**：

$$x_t = (1-t) x_0 + t \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

对应的条件速度场：

$$u_t(x_t | x_0) = \frac{dx_t}{dt} = \epsilon - x_0$$

### CFM 训练目标

$$\boxed{L_{\text{CFM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1), x_0 \sim p_0, \epsilon \sim \mathcal{N}(0,I)} \left[ \| v_\theta(x_t, t) - (\epsilon - x_0) \|^2 \right]}$$

与 DDPM 的训练惊人地相似——都是预测一个与噪声/数据相关的目标。但 Flow Matching 更直接：目标就是 $\epsilon - x_0$（噪声减去数据），不需要复杂的噪声调度参数。

```python
def flow_matching_loss(model, x_0):
    """标准 Flow Matching 训练损失"""
    t = torch.rand(x_0.shape[0], 1, 1, 1)     # 均匀采样 t in [0,1]
    eps = torch.randn_like(x_0)                 # 噪声
    x_t = (1 - t) * x_0 + t * eps              # 线性插值
    target = eps - x_0                          # 条件速度
    pred = model(x_t, t.squeeze())              # 网络预测速度
    loss = F.mse_loss(pred, target)
    return loss
```

### 为什么 CFM 等价于 FM？

🔬 深入

关键定理（Lipman et al., 2022）：条件流匹配损失 $L_{\text{CFM}}$ 与全局流匹配损失 $L_{\text{FM}}$ 的梯度相同：

$$\nabla_\theta L_{\text{CFM}} = \nabla_\theta L_{\text{FM}}$$

直觉：尽管每个训练样本只看到一条条件路径，但对所有数据点 $x_0$ 求期望后，条件速度的平均恰好等于全局速度场。这就是为什么可以用简单的线性插值训练，却能学到正确的全局流。

## 最优传输路径（Optimal Transport Path）

### OT 路径的数学推导

**最优传输（Optimal Transport, OT）** 问题：寻找从分布 $p_0$ 到 $p_1$ 的映射 $T$，使总传输代价最小：

$$\min_{T} \mathbb{E}_{x \sim p_0}[\|x - T(x)\|^2]$$

对于高斯分布对 $p_0 = \mathcal{N}(\mu_0, \Sigma_0)$，$p_1 = \mathcal{N}(\mu_1, \Sigma_1)$，最优传输映射是**线性的**：

$$T(x) = \mu_1 + A(x - \mu_0), \quad A = \Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}\Sigma_0^{-1/2}$$

当 $p_0 = p_{\text{data}}$，$p_1 = \mathcal{N}(0, I)$ 时，**条件 OT 路径**（以单点 $x_0$ 为条件）简化为线性插值：

$$x_t = (1-t) x_0 + t \epsilon$$

这恰好就是 CFM 使用的路径！因此，线性插值不是任意选择，而是**条件最优传输的精确解**。

```
扩散路径 (曲线):           Flow Matching (直线):
                            
x₁(噪声) ╮                 x₁(噪声)
          ╲                    \
           ╲                    \
            ╲                    \  <-- OT 直线路径
             ╲                    \     代价最小
              ╲                    \
               ╲                    \
x₀(数据) <----╯                 x₀(数据)
```

**直线路径的优势**：
- 路径更短 → ODE 求解需要更少的步数
- 路径不交叉 → 数值积分更稳定
- 条件最优传输 → 理论上的效率最优

## Rectified Flow（直线流）与 Reflow

### Reflow 算法

Liu et al.（2022）提出的 **Reflow** 过程可以进一步拉直路径。核心思想：用训练好的模型生成新的 $(x_0, x_1)$ 配对，重新训练模型使路径更直。

```python
def reflow_training(model_k, model_k1, dataloader, num_steps=100000):
    """
    Reflow 算法：用第 k 代模型生成配对，训练第 k+1 代模型
    
    model_k:  已训练好的第 k 代 flow model
    model_k1: 待训练的第 k+1 代 flow model（随机初始化或从 k 热启动）
    """
    for step in range(num_steps):
        # 1. 从数据集采样 x_0
        x_0 = next(dataloader)
        
        # 2. 用第 k 代模型将 x_0 映射到 x_1（正向 ODE）
        with torch.no_grad():
            x_1 = ode_solve(model_k, x_0, t_start=0, t_end=1, steps=50)
        
        # 3. 训练第 k+1 代模型连接 (x_0, x_1) 对
        t = torch.rand(x_0.shape[0], 1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1           # 线性插值
        target = x_1 - x_0                        # 连接两端的直线速度
        
        pred = model_k1(x_t, t.squeeze())
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
```

**Reflow 迭代效果**：

| Reflow 迭代次数 | 路径曲直度 | 1步采样 FID | 5步采样 FID |
|-----------------|----------|------------|------------|
| 0 (初始) | 曲线 | 很差 | ~5.0 |
| 1 | 较直 | ~12.0 | ~3.5 |
| 2 | 接近直线 | ~6.0 | ~2.8 |
| 3 | 几乎直线 | ~4.5 | ~2.6 |

经过 2-3 次 Reflow，路径几乎变成直线，1-2 步采样就能获得较高质量结果。

### Stochastic Interpolants 框架

Albergo & Vanden-Eijnden（2022）提出的**随机插值（Stochastic Interpolants）** 框架为 Flow Matching 提供了更一般的理论基础：

$$x_t = \alpha_t x_0 + \beta_t \epsilon, \quad \alpha_0 = 1, \beta_0 = 0, \alpha_1 = 0, \beta_1 = 1$$

线性插值是其特例（$\alpha_t = 1-t, \beta_t = t$）。该框架还支持：
- 带额外随机性的插值路径
- VP/VE 扩散作为特例的统一描述
- 更灵活的端点分布选择

## 与传统扩散的关键差异

| 方面 | 扩散模型 (DDPM) | DDIM | Flow Matching |
|------|---------------|------|---------------|
| 路径形状 | 曲线（由 SDE 决定） | 曲线（ODE） | 直线（线性插值） |
| 时间范围 | $t \in \{0, ..., T\}$ 离散 | $t \in \{0, ..., T\}$ | $t \in [0, 1]$ 连续 |
| 噪声调度 | 需要精心设计 $\beta_t$ | 继承 DDPM 的 $\bar{\alpha}_t$ | 不需要（线性自然定义） |
| 训练目标 | 预测噪声 $\epsilon$ | 同 DDPM | 预测速度 $v = \epsilon - x_0$ |
| 理论框架 | SDE → ODE | SDE → ODE | 直接 ODE（CNF） |
| 采样效率 | 差（需要 50-1000 步） | 中（20-50 步） | 好（10-25 步理论最优） |

### FID / 步数 / 速度详细对比

| 方法 | 步数 | FID (CIFAR-10) | FID (ImageNet 256) | 每步计算量 | 总时间(相对) |
|------|------|---------------|-------------------|-----------|-------------|
| DDPM | 1000 | 3.17 | — | 1x | 1000x |
| DDIM | 50 | 4.67 | — | 1x | 50x |
| DDIM | 10 | 13.36 | — | 1x | 10x |
| DPM-Solver | 20 | 2.87 | — | 1x | 20x |
| FM (Euler) | 100 | 2.99 | 2.36 (SD3-8B) | 1x | 100x |
| FM (Euler) | 25 | 3.82 | 2.50 (SD3-8B) | 1x | 25x |
| FM + Reflow | 10 | 4.85 | — | 1x | 10x |
| FM + Reflow | 1 | 6.18 | — | 1x | 1x |

**关键观察**：FM 在相同步数下通常优于 DDIM，在少步数场景（10-25 步）优势尤其明显，因为直线路径允许更大的步长而不引入过多截断误差。

## Stable Diffusion 3 中的 Flow Matching

SD3（Esser et al., 2024）是首个大规模采用 Flow Matching 的商业级模型，标志着 FM 从研究走向工业落地。

### Logit-Normal 时间采样

SD3 使用 **logit-normal 时间采样** 替代均匀采样。时间 $t$ 从以下分布采样：

$$t = \sigma(\mu + s \cdot z), \quad z \sim \mathcal{N}(0, 1)$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数，$\mu = 0$，$s = 1$。

**为什么不用均匀采样？** 实验发现不同时间步对生成质量的贡献不同：
- 接近 $t=0$（几乎无噪声）和 $t=1$（几乎纯噪声）的时间步，训练信号弱
- 中间时间步（$t \approx 0.3 \sim 0.7$）承载最多语义信息

Logit-normal 在中间集中更多采样概率，训练效率提升约 15-20%。

```python
def logit_normal_sampling(batch_size, mu=0.0, s=1.0):
    """SD3 的 logit-normal 时间采样"""
    z = torch.randn(batch_size)
    t = torch.sigmoid(mu + s * z)       # logit-normal 分布
    # t 集中在 0.3~0.7 之间，两端概率密度低
    return t
```

### SD3 架构：MM-DiT + Rectified Flow

```
SD3 完整架构:
┌─────────────────────────────────────────────────────────────┐
│  文本编码: CLIP-L + CLIP-G + T5-XXL → 文本嵌入              │
│  (三编码器融合，捕获不同粒度的语义)                            │
├─────────────────────────────────────────────────────────────┤
│  MM-DiT (多模态 DiT):                                       │
│  ┌──────────┐   ┌──────────┐                                │
│  │ 文本流    │   │ 图像流    │  ← 双流 Transformer            │
│  │ (text)   │ ↔ │ (image)  │  ← Joint Attention 交互       │
│  └──────────┘   └──────────┘                                │
│  参数: 2B / 8B                                               │
├─────────────────────────────────────────────────────────────┤
│  训练: Rectified Flow + logit-normal 时间采样                │
│  采样: Euler 求解器, 28-50 步                                 │
│  CFG scale: 4.0-7.0                                         │
└─────────────────────────────────────────────────────────────┘
```

## Flow Matching 在 VLA 中的应用

Flow Matching 不仅用于图像生成，也已成为机器人**动作生成（Action Generation）** 的核心方法。

**$\pi_0$（Physical Intelligence, 2024）** 使用 Flow Matching 作为动作解码头（详见 [VLA: $\pi_0$ 详解](../../../..//posts/vla/03-models-zoo/06-pi-zero)）：
- 将动作序列视为"数据"，高斯噪声视为"起点"
- 用 FM 的速度场将噪声"流动"到合理的动作序列
- 天然支持多模态动作分布（如左抓/右抓都是合理策略）
- 比离散 Token 自回归更适合连续动作空间

这一应用表明 Flow Matching 的简洁性使其成为**通用的连续分布建模工具**，不局限于图像生成。更多细节参见 [VLA: Diffusion 作为动作生成器](../../../..//posts/vla/07-foundation-models/03-diffusion-action)。

## 小结

| 概念 | 要点 |
|------|------|
| Flow Matching | 学习速度场连接噪声和数据分布，训练目标 $v = \epsilon - x_0$ |
| 条件流匹配 (CFM) | 用线性插值定义条件路径，梯度等价于全局 FM |
| 最优传输路径 | 线性插值是条件 OT 的精确解，路径最短 |
| Rectified Flow + Reflow | 通过迭代拉直路径，2-3 轮后接近直线 |
| Stochastic Interpolants | 更一般的理论框架，统一 FM 和扩散 |
| SD3 落地 | MM-DiT + logit-normal 时间 + Rectified Flow |
| VLA 应用 | $\pi_0$ 用 FM 做动作生成，天然支持多模态分布 |
| vs DDPM/DDIM | 更简洁（无噪声调度）、更高效（少步数）、理论更优雅 |

---

> **下一篇**：[模型演化关系图](./07-evolution-map)

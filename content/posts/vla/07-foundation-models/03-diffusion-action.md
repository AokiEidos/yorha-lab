---
title: "Diffusion 作为动作生成器"
date: 2026-04-20T16:31:31.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# Diffusion 作为动作生成器

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[动作解码头设计](../02-architecture/03-action-head)
> 交叉引用：[Diffusion 系列 · 基础理论](..//posts/diffusion/01-fundamentals/01-core-intuition)，[Diffusion 系列 · Flow Matching](..//posts/diffusion/02-models-zoo/06-flow-matching)

## 为什么动作生成需要 Diffusion

行为克隆（BC）中最常见的动作输出方式是 MLP 回归或离散 Token 自回归。这两种方式在简单任务上表现不错，但在面对**多模态动作分布（Multi-modal Action Distribution）** 时会失败。

### 多模态动作问题的直觉

想象这个场景：机器人面前有一个杯子，需要抓取。从左边抓和从右边抓都是合理的策略。

```
观测: [杯子在正前方]

合理动作 1: 从左边接近 → Δy = -0.05
合理动作 2: 从右边接近 → Δy = +0.05

MLP 回归输出: Δy = 0.0 (两者平均值) → 直接撞向杯子中间！
```

MLP 回归输出所有合理动作的**均值（Mean）**——当分布是多峰的，均值恰好落在两个峰之间的低概率区域。这就是所谓的**模式平均化问题（Mode Averaging Problem）**。

更严重的例子发生在灵巧操作中：折衣服时，先折左边还是先折右边都可以，但做到一半改主意（混合两种策略）会导致衣服变成一团。

### 为什么 Diffusion/Flow Matching 能解决

Diffusion 模型天然建模**整个分布**而非单一输出——通过去噪过程中的随机性，每次采样可以落在不同的模式上：

```
采样 1: 噪声种子 A → [去噪] → 从左边抓 (Δy = -0.05)
采样 2: 噪声种子 B → [去噪] → 从右边抓 (Δy = +0.05)
```

每次采样都是一个完整的、一致的动作策略——不存在模式平均化。

## Diffusion Policy（Chi et al., 2023）

### 核心架构

**Diffusion Policy** 将条件去噪过程应用于动作序列生成。训练时向动作序列加噪然后学习去噪，推理时从纯噪声出发条件生成动作。

```
┌──────────────────────────────────────────────────────┐
│                  Diffusion Policy                     │
│                                                       │
│  推理过程:                                             │
│                                                       │
│  观测 o_t ─→ [观测编码器] ─→ 条件特征 c                 │
│                                  │                    │
│  噪声动作 a_T ~ N(0,I)          │                    │
│  shape: [chunk_size, action_dim]  │                    │
│              │                    │                    │
│              ▼                    ▼                    │
│         ┌─────────────────────────────┐               │
│         │    条件去噪网络               │               │
│         │    ε_θ(a_t, t, c)           │               │
│         │    (1D U-Net 或 Transformer) │               │
│         └─────────────┬───────────────┘               │
│                       │ 重复 T 步去噪                   │
│                       ▼                               │
│  干净动作序列 a_0: [a¹, a², ..., a^H]                  │
│  shape: [chunk_size, action_dim]                      │
│  例如: [16 步, 7 维] = 16 个未来动作                    │
└──────────────────────────────────────────────────────┘
```

### 两种条件化架构

Diffusion Policy 论文提出了两种架构变体：

#### CNN-based（1D U-Net）

```
条件特征 c (来自 ResNet/ViT 视觉编码器)
    │
    ▼ FiLM 条件化
噪声动作序列 a_t: [B, H, A]  (H=chunk_size, A=action_dim)
    │
    ▼ reshape to [B, A, H] (将时间维度视为空间维度)
    │
[1D U-Net]  ← 时间步嵌入 t
    │           ← FiLM(条件特征 c) 在每层注入
    ▼
预测噪声 ε: [B, A, H]
    │
    ▼ DDPM/DDIM 去噪步
    │
干净动作: [B, H, A]
```

#### Transformer-based

```
[观测 Token] [时间步 Token] [噪声动作 Token₁...Token_H]
                    │
                    ▼
            [Transformer Decoder]
            (因果 mask: 观测可见,
             动作 Token 间可见)
                    │
                    ▼
            [预测的去噪动作 Token₁...Token_H]
```

论文发现 CNN-based 版本在大多数任务上表现更好且更稳定。

### 训练过程

```python
def diffusion_policy_training_step(model, obs_encoder, batch):
    """Diffusion Policy 训练一步"""
    obs = batch['observations']        # [B, obs_dim] 或图像
    actions = batch['actions']          # [B, H, A] 动作序列 (H步 chunk)
    
    # 1. 编码观测为条件特征
    cond = obs_encoder(obs)            # [B, cond_dim]
    
    # 2. 采样随机时间步和噪声
    t = torch.randint(0, T, (B,))      # 随机时间步
    noise = torch.randn_like(actions)   # 与动作同形的噪声
    
    # 3. 前向加噪
    alpha_bar = alpha_bars[t].reshape(B, 1, 1)
    noisy_actions = torch.sqrt(alpha_bar) * actions + torch.sqrt(1 - alpha_bar) * noise
    
    # 4. 预测噪声
    noise_pred = model(noisy_actions, t, cond)
    
    # 5. MSE 损失
    loss = F.mse_loss(noise_pred, noise)
    return loss
```

### 推理过程

```python
@torch.no_grad()
def diffusion_policy_inference(model, obs_encoder, obs, num_steps=100):
    """Diffusion Policy 推理：从噪声生成动作序列"""
    cond = obs_encoder(obs)
    
    # 从纯噪声开始
    actions = torch.randn(1, chunk_size, action_dim)  # [1, H, A]
    
    # DDPM 采样（或 DDIM 加速）
    for t in reversed(range(num_steps)):
        noise_pred = model(actions, t, cond)
        actions = ddpm_step(actions, noise_pred, t)  # 一步去噪
    
    return actions  # [1, H, A] 干净动作序列
```

### 实验结果

Diffusion Policy 在 RobotMimic 基准上的表现：

| 任务 | BC (MLP) | BC-RNN | IBC | Diffusion Policy |
|------|---------|--------|-----|-----------------|
| Lift | 100% | 100% | 96.7% | **100%** |
| Can | 90.0% | 96.7% | 93.3% | **100%** |
| Square | 76.7% | 90.0% | 50.0% | **96.7%** |
| Transport | 43.3% | 76.7% | 36.7% | **93.3%** |
| Tool Hang | 0.0% | 10.0% | 0.0% | **73.3%** |

在 Tool Hang（需要精确对准后挂上）这类多模态任务上，Diffusion Policy 的优势最为显著——BC/MLP 完全失败，Diffusion Policy 达到 73%。

## Flow Matching：π₀ 的选择

### 为什么 π₀ 选 Flow Matching 而非 Diffusion

π₀ 使用 **Flow Matching** 替代传统 Diffusion，有三个关键原因：

#### 1. 训练目标更简洁

**Diffusion 训练目标**：预测噪声 $\epsilon$
$$L_{\text{Diff}} = \mathbb{E}_{t,\epsilon} \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2$$

**Flow Matching 训练目标**：预测速度 $v = \epsilon - x_0$
$$L_{\text{FM}} = \mathbb{E}_{t,\epsilon} \| (\epsilon - x_0) - v_\theta(x_t, t, c) \|^2$$

其中 $x_t = (1-t)x_0 + t\epsilon$（线性插值）。

Flow Matching 不需要噪声调度参数（$\beta_t$, $\bar{\alpha}_t$ 等），训练更简洁。

#### 2. 采样路径更直

```
Diffusion 路径:     Flow Matching 路径:

噪声 x_T            噪声 x_1
  │                    │
  ╲                    ╲
   ╲                    ╲  ← 直线路径
    ╲                    ╲
     ╲                    ╲
      ╲                    ╲
数据 x_0               数据 x_0
(曲线路径,需更多步)    (直线路径,需更少步)
```

直线路径意味着用更少的 ODE 求解步数就能达到相同质量。

#### 3. 推理速度更快

| 方法 | 采样步数 | 动作质量 |
|------|---------|---------|
| DDPM (Diffusion) | 100 步 | 最优 |
| DDIM (Diffusion) | 10-20 步 | 接近最优 |
| **Flow Matching** | **5-10 步** | 接近最优 |

在 π₀ 中，Flow Matching 只需 ~10 步采样就能生成高质量的 50 步动作块。

### π₀ 的 Flow Matching 动作头

```python
class PiZeroFlowMatchingHead(nn.Module):
    """π₀ 的 Flow Matching 动作解码头"""
    
    def __init__(self, cond_dim, action_dim, chunk_size=50):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # 速度预测网络 (Action Expert)
        # 接收: 噪声动作 + VLM条件特征 + 时间步
        self.velocity_net = TransformerDecoder(
            input_dim=action_dim,
            cond_dim=cond_dim,
            num_layers=4,
            hidden_dim=512,
        )
    
    def compute_loss(self, cond, actions):
        """训练: Flow Matching 损失"""
        B = actions.shape[0]
        t = torch.rand(B, 1, 1)                    # [B, 1, 1]
        noise = torch.randn_like(actions)           # [B, chunk_size, action_dim]
        
        # 线性插值
        x_t = (1 - t) * actions + t * noise         # [B, 50, action_dim]
        
        # 目标速度 = noise - data
        target_velocity = noise - actions
        
        # 预测速度
        pred_velocity = self.velocity_net(x_t, t, cond)
        
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss
    
    @torch.no_grad()
    def sample(self, cond, num_steps=10):
        """推理: 从噪声采样动作"""
        B = cond.shape[0]
        x = torch.randn(B, self.chunk_size, self.action_dim)
        
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B, 1, 1), 1.0 - i * dt)
            velocity = self.velocity_net(x, t, cond)
            x = x - velocity * dt  # Euler 步 (从 t=1 到 t=0)
        
        return x  # [B, 50, action_dim] 干净动作序列
```

## π₀ 的 Action Expert 设计

π₀ 的 Flow Matching 头不是一个简单的 MLP——它是一个**Action Expert** Transformer，与 VLM 的 Transformer 层交织在一起：

```
VLM Transformer 层:
  Layer 1: [视觉Token, 语言Token] → Self-Attention → 输出
  Layer 2: [视觉Token, 语言Token] → Self-Attention → 输出
  ...
  
Action Expert 层 (交织在 VLM 层之间):
  Action Layer 1: [动作Token] → Cross-Attention(条件=VLM Layer 2 输出) → 输出
  Action Layer 2: [动作Token] → Cross-Attention(条件=VLM Layer 4 输出) → 输出
  ...
```

这种交织设计让动作生成能够利用 VLM 不同层级的特征——低层的空间细节和高层的语义理解。

## Diffusion/FM vs 自回归 Token 的深度对比

| 维度 | 自回归 Token (OpenVLA) | Diffusion Policy | Flow Matching (π₀) |
|------|----------------------|-----------------|-------------------|
| **动作精度** | 离散 (~0.4mm/bin) | 连续（无损） | 连续（无损） |
| **多模态** | ❌ 单峰 argmax | ✅ 多峰采样 | ✅ 多峰采样 |
| **Action Chunk** | 通常 1 步 | 天然 (16步) | 天然 (50步) |
| **推理步数** | 7 token 串行 | 10-100 步去噪 | 5-10 步 ODE |
| **架构改动** | 最小（扩展词表） | 需加 1D U-Net | 需加 Action Expert |
| **训练损失** | 交叉熵 | MSE（ε-预测） | MSE（v-预测） |
| **训练难度** | 低 | 中等 | 中等 |
| **实时推理** | ~200ms (7 token) | ~500ms (100步) | ~100ms (10步) |
| **灵巧操作** | 有限 | 好 | **最好** |
| **理论基础** | next-token prediction | SDE/ODE 理论 | 最优传输理论 |

### 推理延迟的实际比较

以 50 步动作 chunk、7 维动作空间为例（A100 GPU）：

| 方法 | 每次推理延迟 | 等效控制频率 | 备注 |
|------|------------|------------|------|
| OpenVLA (7B, 自回归) | ~1.4s (350个token) | ~0.7 Hz | 50步需350个token |
| Diffusion Policy (CNN) | ~500ms (100步去噪) | ~2 Hz | 可用DDIM加速 |
| Diffusion Policy (DDIM 10步) | ~50ms | ~20 Hz | 质量略降 |
| **Flow Matching (π₀, 10步)** | ~30ms | **~33 Hz** | 最快 |

## 何时使用 Diffusion/FM

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速原型/验证 | 自回归 Token | 最简单，无需额外组件 |
| 多模态动作（灵巧/双臂） | Flow Matching | 多峰建模 + 快速采样 |
| 接触丰富的操作 | Diffusion/FM | 时间一致的动作序列 |
| 延迟敏感部署 | Flow Matching | 5-10 步采样足够 |
| 资源极度受限 | MLP 回归 | 单步前向传播 |

## 小结

| 概念 | 要点 |
|------|------|
| 多模态问题 | MLP 回归/自回归 Token 无法表达多峰动作分布 |
| Diffusion Policy | 条件去噪生成动作序列，天然多模态 |
| Flow Matching | 直线路径替代曲线，更少步数、更快采样 |
| π₀ Action Expert | 与 VLM 交织的 Transformer 动作解码器 |
| 速度对比 | FM(10步) ~30ms < Diffusion(10步) ~50ms << 自回归 ~1.4s |

---

> **下一篇**：[具身智能 Agent 框架](./04-embodied-agent) — VLA 在完整 Agent 系统中的定位。

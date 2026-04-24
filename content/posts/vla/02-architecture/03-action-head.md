---
title: "动作解码头设计"
date: 2026-04-20T15:31:13.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 动作解码头设计

> ⚙️ 进阶 | 前置知识：[动作表示与 Token 化](./02-action-representation)

## 动作解码头的角色

**动作解码头（Action Head / Action Decoder）** 是 VLA 中负责将 VLM 的内部表示转化为具体机器人动作的模块。它是 VLA 区别于 VLM 的关键组件。

## 自回归 Token 预测头

### 设计

最简单的方案——直接复用 LLM 的**语言模型头（LM Head）**，将动作 Token 作为词表的一部分。

```python
class AutoregressiveActionHead:
    def __init__(self, llm):
        # 扩展词表：原始词表 + 256×动作维度 个新 Token
        self.llm = llm
        self.llm.resize_token_embeddings(
            original_vocab_size + 256 * action_dim
        )
    
    def generate_action(self, features):
        action_tokens = []
        for dim in range(action_dim):
            logits = self.llm.lm_head(features)
            token = logits.argmax(dim=-1)  # 或采样
            action_tokens.append(token)
            features = self.llm.forward_one_step(token)
        return dequantize(action_tokens)
```

**使用模型**：RT-2, OpenVLA

## MLP 回归头

### 设计

在 VLM 的输出特征上接 2-3 层 MLP，直接回归连续动作值。

```python
class MLPActionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, chunk_size=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * chunk_size),
        )
    
    def forward(self, features):
        return self.mlp(features)  # [B, action_dim * chunk_size]
```

**使用模型**：Octo（带 Readout Token）

### Octo 的 Readout Token

Octo 引入了**Readout Token**——一组可学习的特殊 Token，用于从 Transformer 中"读出"动作相关的信息：

```
[视觉 Token] [语言 Token] [Readout Token₁...Readoutₖ] → Transformer → Readout 输出 → MLP → 动作
```

Readout Token 使得同一个 Transformer 可以灵活适配不同的动作空间。

## Diffusion 解码头

### 设计

用条件去噪网络从噪声中生成动作序列。VLM 的输出特征作为条件。

```python
class DiffusionActionHead(nn.Module):
    def __init__(self, action_dim, chunk_size, hidden_dim):
        super().__init__()
        self.noise_pred_net = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=hidden_dim,
            diffusion_step_embed_dim=128,
        )
    
    def forward(self, features, noisy_actions, timestep):
        # 预测噪声
        return self.noise_pred_net(noisy_actions, timestep, features)
    
    @torch.no_grad()
    def sample(self, features, num_steps=10):
        actions = torch.randn(B, chunk_size, action_dim)
        for t in reversed(range(num_steps)):
            noise_pred = self.forward(features, actions, t)
            actions = ddpm_step(actions, noise_pred, t)
        return actions
```

**使用模型**：Diffusion Policy（独立使用）、部分 VLA 变体

## Flow Matching 解码头

### 设计

Flow Matching 用速度场替代噪声预测——从噪声到动作的直线路径。

```python
class FlowMatchingActionHead(nn.Module):
    def __init__(self, action_dim, chunk_size, hidden_dim):
        super().__init__()
        self.velocity_net = MLP(
            input_dim=action_dim + hidden_dim + 1,  # 动作 + 条件 + 时间
            output_dim=action_dim * chunk_size,
        )
    
    def forward(self, features, noisy_actions, t):
        return self.velocity_net(noisy_actions, features, t)
    
    @torch.no_grad()
    def sample(self, features, num_steps=10):
        actions = torch.randn(B, chunk_size, action_dim)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = i * dt
            velocity = self.forward(features, actions, t)
            actions = actions + velocity * dt  # Euler 步
        return actions
```

**使用模型**：π₀

## Action Chunking：一次预测多步

### 为什么需要 Action Chunking

**Action Chunking（动作分块）** 是一次预测未来 $H$ 步动作（$H$ 称为 chunk size 或 horizon），而非每次只预测 1 步。

**动机**：
1. **减少推理频率**：VLM 推理慢（~100ms），如果每步都推理一次则控制频率受限于 ~10Hz。预测 16 步后一次性执行，等效控制频率可达 ~160Hz
2. **时间一致性**：一次生成的多步动作天然在时间上连贯
3. **减少累积误差**：推理次数减少 = 误差累积机会减少

### 典型 Chunk Size

| 模型 | Chunk Size | 控制频率 |
|------|-----------|---------|
| RT-2 | 1 | ~3 Hz |
| OpenVLA | 1 | ~5 Hz |
| ACT (ALOHA) | 100 | 50 Hz |
| π₀ | 50 | 50 Hz |
| Diffusion Policy | 16 | 10 Hz |

### 执行策略

预测 $H$ 步后如何执行？

- **全部执行**：执行所有 $H$ 步后再推理下一块
- **部分执行**：只执行前 $k < H$ 步，然后重新推理（重叠执行）
- **时间集成（Temporal Ensemble）**：多次预测的重叠部分取加权平均

## 解码头对比

| 解码头 | 动作类型 | 多模态 | 推理速度 | Action Chunking | 代表 |
|--------|---------|--------|---------|----------------|------|
| 自回归 Token | 离散 | 否 | 中等 | 不常用 | RT-2, OpenVLA |
| MLP 回归 | 连续 | 否 | 快 | 可选 | Octo |
| Diffusion | 连续 | ✅ | 慢 | 天然适配 | Diffusion Policy |
| Flow Matching | 连续 | ✅ | 较快 | 天然适配 | π₀ |

---

> **下一篇**：[多模态输入融合](./04-multimodal-fusion)

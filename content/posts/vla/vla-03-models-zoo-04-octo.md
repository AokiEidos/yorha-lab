---
title: "Octo 详解"
date: 2026-04-20T16:32:38.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# Octo 详解

> ⚙️ 进阶 | 前置知识：[RT-2 详解](./03-rt2.md)

## 定位：开源通用机器人策略

**Octo**（UC Berkeley, 2024，发表于 *RSS 2024*）是第一个在 Open X-Embodiment 数据集上预训练的**开源**通用机器人策略。它不是严格意义上的 VLA（没有大型 VLM 骨干），而是一个**基于 Transformer 的通用机器人策略模型**。但其设计思想——跨具身体预训练、灵活的任务条件化、可更换的动作解码头——深刻影响了后续 VLA（如 OpenVLA、π₀）的发展。

核心定位：**一个可微调的机器人基座模型（Foundation Model）**——预训练后在新机器人/新任务上用少量数据（~100 条轨迹）快速适配。

## 完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Octo Architecture                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    输入 Tokenization                      │   │
│  │                                                          │   │
│  │  语言指令: "pick up the red block"                        │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  [T5-base Encoder] → N_lang 个 Task Token                │   │
│  │                                                          │   │
│  │  ─ OR ─                                                  │   │
│  │                                                          │   │
│  │  目标图像 (goal image):                                    │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  [ViT Patch Embedding] → N_goal 个 Task Token             │   │
│  │                                                          │   │
│  │  ─────────────────────────────────────────                │   │
│  │                                                          │   │
│  │  当前图像 (1-2 视角, 256×256):                              │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  [ViT-B/16] → 每视角 256 个 Observation Token              │   │
│  │  (patch_size=16, 256/16=16, 16×16=256)                   │   │
│  │                                                          │   │
│  │  本体感觉 (joint positions, etc):                          │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  [MLP 投影] → 1 个 Proprio Token                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Transformer Backbone                         │   │
│  │                                                          │   │
│  │  输入序列:                                                 │   │
│  │  [Task Token...] [Obs Token (cam1)...] [Obs Token (cam2)] │   │
│  │  [Proprio Token] [Readout Token_1...Readout Token_R]      │   │
│  │                                                          │   │
│  │  结构:                                                     │   │
│  │    Octo-Small: 12 层, d=384,  6 头, 27M 参数              │   │
│  │    Octo-Base:  24 层, d=768, 12 头, 93M 参数              │   │
│  │                                                          │   │
│  │  注意力模式: 分组因果注意力                                   │   │
│  │    - Task Token: 只看自己 (不参与全局注意力)                  │   │
│  │    - Obs Token: 看 Task + 自己 (不看 Readout)              │   │
│  │    - Readout Token: 看所有 Token (全局注意力)               │   │
│  │                                                          │   │
│  │  位置编码: Learned 2D (图像) + Learned 1D (序列)           │   │
│  └──────────────────────────────┬───────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Readout Token → Action Head                  │   │
│  │                                                          │   │
│  │  Readout Token 输出 (R 个, d 维)                          │   │
│  │         │                                                │   │
│  │    ┌────┴────┐                                           │   │
│  │    │  选择一  │                                           │   │
│  │    ▼         ▼                                           │   │
│  │  [MLP 头]  [Diffusion 头]                                │   │
│  │  简单回归   多模态生成                                      │   │
│  │    │         │                                           │   │
│  │    ▼         ▼                                           │   │
│  │  动作 (action_dim × chunk_size)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Readout Token 机制深度解析

**Readout Token（读出 Token）** 是 Octo 最重要的设计创新。它是一组**可学习的特殊 Token**（不对应任何输入数据），附加在 Transformer 输入序列的末尾。

### 为什么需要 Readout Token

传统方法会用 Transformer 的 [CLS] Token 或所有 Token 的平均来获取输出表示，但这两种方式都有问题：
- **[CLS] Token**：对所有下游任务共享同一个输出接口，灵活性差
- **Token 平均**：丢失了空间和模态信息

Readout Token 的设计灵感来自 Perceiver：它是专门为"读出动作信息"而设计的查询 Token。

### 注意力模式

```
               Task    Obs    Proprio  Readout
              ┌─────┬──────┬────────┬─────────┐
 Task         │  ✓  │  ✗   │   ✗    │   ✗     │  只看自己（预计算条件）
 Obs          │  ✓  │  ✓   │   ✓    │   ✗     │  看 Task + 自己 + Proprio
 Proprio      │  ✓  │  ✓   │   ✓    │   ✗     │  看 Task + Obs + 自己
 Readout      │  ✓  │  ✓   │   ✓    │   ✓     │  看所有（全局聚合）
              └─────┴──────┴────────┴─────────┘
```

**关键设计决策**：Readout Token 是唯一能看到所有信息的 Token 组。这意味着只有 Readout Token 的输出包含了任务、视觉、本体感觉的完整融合信息。其他 Token 组通过限制注意力范围减少了计算量。

### Readout Token 的灵活性

不同机器人使用**不同的 Readout Token + Action Head**，而 Transformer 骨干共享：

```python
# 伪代码：不同机器人的适配
class OctoModel:
    def __init__(self, backbone, readout_tokens, action_head):
        self.backbone = backbone          # 共享的 Transformer
        self.readout_tokens = readout_tokens  # 可替换
        self.action_head = action_head    # 可替换

# 机器人 A: 7-DOF 臂
model_A = OctoModel(
    backbone=shared_transformer,
    readout_tokens=nn.Parameter(torch.randn(4, 768)),  # 4 个 Readout
    action_head=MLPHead(768 * 4, action_dim=7, chunk_size=4)
)

# 机器人 B: 14-DOF 双臂
model_B = OctoModel(
    backbone=shared_transformer,
    readout_tokens=nn.Parameter(torch.randn(8, 768)),  # 8 个 Readout
    action_head=DiffusionHead(768 * 8, action_dim=14, chunk_size=8)
)
```

## Action Head 对比：MLP vs Diffusion

Octo 支持两种 Action Head，论文中做了详细的对比实验：

### MLP 回归头

```python
class MLPActionHead(nn.Module):
    def __init__(self, input_dim, action_dim, chunk_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_dim * chunk_size),
        )
    
    def forward(self, readout_features):
        # readout_features: (B, R*d) 拼接的 Readout 输出
        return self.net(readout_features).reshape(B, chunk_size, action_dim)
    
    def loss(self, pred, target):
        return F.mse_loss(pred, target)
```

### Diffusion 解码头

```python
class DiffusionActionHead(nn.Module):
    def __init__(self, input_dim, action_dim, chunk_size, 
                 num_diffusion_steps=20):
        super().__init__()
        self.denoise_net = nn.Sequential(
            # 输入: noisy_action (chunk_size*action_dim) + 
            #       condition (input_dim) + timestep_embed (128)
            nn.Linear(chunk_size * action_dim + input_dim + 128, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, chunk_size * action_dim),
        )
        self.num_steps = num_diffusion_steps
    
    def loss(self, readout_features, target_actions):
        # DDPM 训练
        t = torch.randint(0, self.num_steps, (B,))
        noise = torch.randn_like(target_actions)
        noisy_actions = add_noise(target_actions, noise, t)
        pred_noise = self.denoise_net(noisy_actions, readout_features, t)
        return F.mse_loss(pred_noise, noise)
    
    @torch.no_grad()
    def sample(self, readout_features, num_steps=20):
        actions = torch.randn(B, chunk_size, action_dim)
        for t in reversed(range(num_steps)):
            pred_noise = self.denoise_net(actions, readout_features, t)
            actions = ddpm_reverse_step(actions, pred_noise, t)
        return actions
```

### 对比实验结果

| 维度 | MLP Head | Diffusion Head |
|------|---------|---------------|
| 简单拾取任务 | 82% | 80% |
| 多模态任务（多路径） | 58% | **74%** |
| 精细操作 | 65% | **73%** |
| 推理速度 | 1ms | 50ms (20步去噪) |
| 训练稳定性 | 高 | 中 |
| **推荐场景** | 简单任务/实时控制 | 复杂/多模态任务 |

**结论**：Diffusion Head 在需要多模态动作建模的复杂任务上显著优于 MLP Head，但在简单任务上差异不大。Octo 默认使用 Diffusion Head。

## 预训练数据与训练计算

### Open X-Embodiment 数据集

Octo 在 **Open X-Embodiment** 数据集的子集上预训练：

| 数据维度 | 规格 |
|---------|------|
| 总轨迹数 | ~800K |
| 机器人种类 | 22 种 |
| 数据集来源 | 25+ 个不同实验室 |
| 任务种类 | 数百种 |

### 数据分布

| 数据来源 | 轨迹数 | 机器人 | 占比 |
|---------|-------|-------|------|
| Bridge V2 | ~200K | WidowX | 25% |
| RT-1 数据 | ~130K | Everyday Robot | 16% |
| TACO-Play | ~120K | Panda | 15% |
| Language Table | ~100K | xArm | 12% |
| BC-Z | ~100K | Everyday Robot | 12% |
| 其他 20+ 数据集 | ~150K | 各种 | 20% |

### 训练计算详情

| 配置 | Octo-Small | Octo-Base |
|------|-----------|----------|
| 参数量 | 27M | 93M |
| Transformer 层 | 12 | 24 |
| Hidden dim | 384 | 768 |
| 注意力头数 | 6 | 12 |
| 训练步数 | 300K | 300K |
| Batch size | 1024 | 1024 |
| 学习率 | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| 优化器 | AdamW | AdamW |
| 权重衰减 | 0.01 | 0.01 |
| 训练硬件 | TPU v4-128 | TPU v4-128 |
| 训练时间 | ~12 小时 | ~36 小时 |
| 总 FLOPs | ~$2 \times 10^{18}$ | ~$8 \times 10^{18}$ |

训练损失：
$$\mathcal{L}_{\text{Octo}} = \mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(a_t, c, t)\|^2 \right]$$

其中 $a_t$ 是加噪后的动作，$c$ 是 Readout Token 的输出（条件），$t$ 是扩散时间步。

## 微调流程与 API

Octo 提供了简洁的微调 API，这是其核心竞争力之一：

```python
# Octo 完整微调示例
import octo
from octo.model import OctoModel
from octo.data import make_single_dataset

# 1. 加载预训练模型
model = OctoModel.load_pretrained("octo-base")

# 2. 准备新机器人的数据集
dataset = make_single_dataset(
    dataset_name="my_robot_dataset",    # RLDS 格式
    obs_horizon=2,                       # 观测历史长度
    action_horizon=4,                    # Action Chunk 长度
    image_obs_keys=["primary", "wrist"], # 相机名
    state_obs_keys=["joint_positions"],  # 本体感觉
    action_key="actions",
)

# 3. 配置微调
# 选项 A: 只替换 Action Head (最快，保留预训练骨干)
model = model.replace_action_head(
    new_action_dim=7,       # 新机器人的动作维度
    new_chunk_size=4,       # 新的 chunk 大小
    head_type="diffusion",  # 或 "mlp"
)

# 选项 B: 全量微调 (更好但更慢)
# 不调用 replace_action_head，直接在新数据上训练

# 4. 微调循环
optimizer = optax.adamw(learning_rate=3e-5, weight_decay=0.01)

for step, batch in enumerate(dataset):
    loss, grad = jax.value_and_grad(model.loss)(model.params, batch)
    model.params = optimizer.apply(grad, model.params)
    
    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
    
    if step >= 50000:  # 通常 50K 步足够
        break

# 5. 保存微调后的模型
model.save_pretrained("my_fine_tuned_octo")
```

### 微调数据量 vs 性能

| 微调数据量 | WidowX 桌面任务 | Franka 装配任务 |
|-----------|---------------|---------------|
| 0 (零样本) | 18% | 5% |
| 50 条轨迹 | 48% | 32% |
| 200 条轨迹 | 68% | 55% |
| 1000 条轨迹 | 79% | 71% |
| 从零训练 (1000条) | 52% | 38% |

**关键发现**：预训练后仅用 200 条轨迹微调即可超过从零训练 1000 条轨迹的性能。这验证了跨具身体预训练的迁移学习效果。

## 跨具身体实验结果

| 目标机器人 | 零样本 | 微调 (200条) | 从零训练 (200条) | 从零训练 (全量) |
|-----------|-------|-------------|----------------|---------------|
| WidowX (Bridge) | 25% | 72% | 45% | 78% |
| Franka Panda | 8% | 63% | 30% | 68% |
| Google Robot | 42% | 81% | 58% | 85% |
| xArm | 12% | 55% | 28% | 60% |
| UR5 | 5% | 48% | 22% | 52% |

### 与其他方法的对比

| 方法 | 参数量 | 开源 | 跨机器人 | WidowX 微调 | Franka 微调 |
|------|-------|------|---------|------------|------------|
| RT-1-X | 35M | 否 | 部分 | 55% | 42% |
| RT-2-X | 55B | 否 | 部分 | 68% | 58% |
| **Octo-Base** | **93M** | **是** | **是** | **72%** | **63%** |
| Octo-Small | 27M | 是 | 是 | 65% | 55% |

## Octo 的局限性

1. **没有 VLM 骨干**：Octo 的 Transformer 仅 93M 参数，没有利用互联网规模的视觉-语言预训练知识。语义理解能力有限，无法处理 RT-2 式的涌现任务
2. **视觉编码器较弱**：使用标准 ViT-B/16 从头训练，没有 ImageNet 或 CLIP 预训练
3. **Diffusion Head 推理慢**：20 步去噪需要 ~50ms，限制了控制频率
4. **零样本能力有限**：跨机器人零样本成功率很低（5-42%），必须微调才能获得可用性能
5. **JAX 生态**：基于 JAX/Flax 实现，社区生态不如 PyTorch 丰富

## Octo 对后续工作的影响

| 设计理念 | Octo 的贡献 | 后续继承者 |
|---------|-----------|----------|
| Readout Token | 灵活的动作输出接口 | 被多个后续工作借鉴 |
| 跨具身体预训练 | 证明跨机器人迁移可行 | π₀, OpenVLA |
| 可更换 Action Head | MLP/Diffusion 灵活切换 | π₀ (Flow Matching) |
| 开源生态 | 第一个可用的开源机器人基座模型 | 推动了开源 VLA 运动 |
| 微调 API | 简洁的适配新机器人流程 | OpenVLA HuggingFace API |

## 小结

| 概念 | 要点 |
|------|------|
| 定位 | 开源可微调的机器人基座模型，93M 参数 |
| Readout Token | 可学习查询 Token，全局注意力聚合信息→输出动作 |
| 注意力模式 | 分组因果注意力，Readout 看所有、Obs 看 Task+自己 |
| Action Head | MLP（快速简单） vs Diffusion（多模态更好），可更换 |
| 预训练 | Open X-Embodiment 800K 轨迹, 22 种机器人, TPU v4 ~36h |
| 微调 | 200 条轨迹微调 > 1000 条从零训练，验证迁移学习 |
| 开源影响 | 第一个可用的开源机器人基座模型，推动后续开源 VLA |

---

> **下一篇**：[OpenVLA 详解](./05-openvla.md)

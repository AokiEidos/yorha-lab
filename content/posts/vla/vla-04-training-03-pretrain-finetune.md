---
title: "4.3 预训练与微调范式"
date: 2026-04-20T16:05:33.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 4.3 预训练与微调范式

> **难度**: ⚙️进阶 | **前置阅读**: [2.1 整体架构](../02-architecture/01-overview.md), [4.1 大规模机器人数据集](./01-datasets.md)

## 核心思想

VLA模型的训练遵循大模型时代的经典范式：**先预训练，再微调**。但与纯语言模型不同，VLA需要将视觉-语言预训练的知识迁移到机器人动作生成这一全新领域。本节详解这一两阶段范式及其关键技术。

---

## 两阶段训练流水线 🔰入门

```
阶段一：VLM预训练                     阶段二：机器人微调
┌─────────────────────┐           ┌─────────────────────┐
│  大规模图文数据       │           │  机器人操作数据       │
│  (LAION, WebLI等)   │           │  (OXE, DROID等)     │
│         ↓           │           │         ↓           │
│  视觉编码器 + LLM    │    →→→    │  VLM + 动作解码头    │
│  学习：看图说话      │   权重迁移  │  学习：看图做事      │
│  输出：文本token     │           │  输出：动作token     │
└─────────────────────┘           └─────────────────────┘
```

### 阶段一：VLM预训练

**视觉-语言模型预训练（VLM Pre-training）** 让模型在海量图文配对数据上学习视觉理解和语言推理能力。

常用的预训练基座模型：

| VLM基座 | 参数量 | 使用该基座的VLA |
|---------|--------|----------------|
| PaLI-X | 55B | RT-2, RT-2-X |
| LLaMA/Vicuna | 7B-13B | OpenVLA |
| InternVL | 2B-40B | 多个中国VLA项目 |
| Paligemma | 3B | OpenVLA (替代方案) |
| Qwen-VL | 7B | RoboVLM 等 |

### 阶段二：机器人微调

在VLM基座上添加**动作解码头（Action Head）**，使用机器人数据进行微调。

```python
# VLA两阶段训练伪代码
class VLAModel(nn.Module):
    def __init__(self, vlm_checkpoint):
        # 阶段一的产物：预训练VLM
        self.vision_encoder = load_pretrained_vit(vlm_checkpoint)
        self.llm = load_pretrained_llm(vlm_checkpoint)
        
        # 阶段二新增：动作解码头
        self.action_head = ActionDecoder(
            input_dim=self.llm.hidden_size,
            output_dim=7,  # (dx, dy, dz, droll, dpitch, dyaw, gripper)
        )
    
    def forward(self, image, instruction, action_gt=None):
        # 视觉编码
        visual_tokens = self.vision_encoder(image)
        # 语言编码
        text_tokens = self.llm.tokenize(instruction)
        # 多模态融合 → LLM推理
        hidden = self.llm(visual_tokens, text_tokens)
        # 动作预测
        action_pred = self.action_head(hidden)
        
        if action_gt is not None:
            loss = F.mse_loss(action_pred, action_gt)
            return loss
        return action_pred
```

---

## 动作表示方式 ⚙️进阶

微调阶段的关键设计选择之一是如何表示动作输出：

### 方式一：动作离散化为token

RT-2的做法是将连续动作**离散化（Discretization）** 为token序列，复用LLM的文本生成能力。

```python
# RT-2 动作离散化
def discretize_action(continuous_action, n_bins=256):
    """将连续动作值映射到离散bin"""
    # 每个维度独立离散化到 [0, 255]
    bins = np.linspace(action_min, action_max, n_bins)
    discrete = np.digitize(continuous_action, bins)
    # 转为token string: "128 64 192 100 88 200 1"
    return ' '.join(map(str, discrete))

# 训练时：指令 + 图像 → "128 64 192 100 88 200 1"
# 推理时：自回归生成动作token序列
```

### 方式二：连续动作回归

OpenVLA等模型直接回归连续动作值：

```python
# 连续动作回归
class ContinuousActionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim=7):
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    
    def forward(self, hidden_states):
        # 取最后一个token的隐状态
        return self.mlp(hidden_states[:, -1, :])
```

### 方式三：扩散动作头

如[3.x 扩散策略](../03-models-zoo/) 所述，使用**扩散过程（Diffusion Process）** 生成动作序列：

| 方式 | 代表模型 | 优点 | 缺点 |
|------|---------|------|------|
| 离散token | RT-2 | 复用LLM能力 | 精度损失、推理慢 |
| 连续回归 | OpenVLA | 简单高效 | 难建模多模态分布 |
| 扩散生成 | Octo, Pi0 | 多模态分布、高质量 | 推理成本高 |

---

## Co-fine-tuning 防遗忘 🔬深入

**协同微调（Co-fine-tuning）** 是RT-2提出的关键技术，解决机器人微调导致VLM原有能力退化的问题。

### 灾难性遗忘问题

```
纯机器人微调的问题：

微调前：VLM能识别"红色苹果"、理解"把它放在碗里"
         ↓ 在机器人数据上微调
微调后：动作预测能力 ↑，但视觉理解/语言推理 ↓↓
         "这是什么？" → 乱码输出（遗忘了！）
```

### Co-fine-tuning 策略

```python
# Co-fine-tuning 训练循环
def co_finetune_step(model, robot_batch, vlm_batch, robot_ratio=0.5):
    """每个训练步同时使用机器人数据和原始VLM数据"""
    
    # 机器人数据：学习动作预测
    robot_loss = model.compute_action_loss(
        images=robot_batch['images'],
        instructions=robot_batch['instructions'],
        actions=robot_batch['actions']
    )
    
    # VLM数据：保持视觉-语言能力
    vlm_loss = model.compute_vlm_loss(
        images=vlm_batch['images'],
        questions=vlm_batch['questions'],
        answers=vlm_batch['answers']
    )
    
    # 混合损失
    total_loss = robot_ratio * robot_loss + (1 - robot_ratio) * vlm_loss
    total_loss.backward()
```

### 效果验证

| 方法 | 机器人任务成功率 | VQA准确率 | 视觉推理 |
|------|----------------|----------|---------|
| 仅机器人微调 | 68% | 31% (-37) | 退化严重 |
| Co-fine-tuning | 73% | 62% (-6) | 轻微下降 |
| 原始VLM（无微调） | 0% | 68% | 完整 |

---

## 冻结策略 ⚙️进阶

**冻结策略（Freezing Strategy）** 决定微调时哪些参数更新、哪些保持不变，是控制训练效率和遗忘的核心手段。

### 常见冻结方案

```
方案A: 全量微调（Full Fine-tuning）
  Vision Encoder: ✅ 更新    LLM: ✅ 更新    Action Head: ✅ 更新
  → 最强适应力，但易遗忘，计算成本最高

方案B: 冻结视觉编码器
  Vision Encoder: ❄️ 冻结    LLM: ✅ 更新    Action Head: ✅ 更新
  → 保留视觉特征，LLM学习动作映射（OpenVLA默认）

方案C: 仅训练动作头
  Vision Encoder: ❄️ 冻结    LLM: ❄️ 冻结    Action Head: ✅ 更新
  → 最快收敛，但任务泛化能力有限

方案D: LoRA微调（参见 4.6）
  Vision Encoder: ❄️ 冻结    LLM: 🔧 LoRA   Action Head: ✅ 更新
  → 参数效率高，遗忘少，推荐的折中方案
```

### 冻结策略选择指南

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 大规模预训练 | 全量微调 | 数据足够覆盖遗忘风险 |
| 单任务部署 | 冻结VLM + 训练Action Head | 快速适配，资源节约 |
| 多任务泛化 | LoRA + Co-fine-tuning | 平衡效率与能力保持 |
| 数据极少 (<100条) | 冻结大部分 + 小学习率 | 防止过拟合 |

---

## 训练超参数参考 ⚙️进阶

| 超参数 | RT-2 (55B) | OpenVLA (7B) | Octo (93M) |
|--------|-----------|-------------|------------|
| 学习率 | 2e-5 | 2e-5 | 3e-4 |
| Batch Size | 2048 | 256 | 1024 |
| 训练步数 | 60,000 | 150,000 | 300,000 |
| 优化器 | Adafactor | AdamW | AdamW |
| 权重衰减 | 0.01 | 0.01 | 0.05 |
| 预热步数 | 2,000 | 1,000 | 5,000 |
| 动作离散化bins | 256 | 256 | N/A (连续) |

---

## 本节小结

| 要点 | 说明 |
|------|------|
| 两阶段范式 | VLM预训练 → 机器人微调是VLA的标准训练路径 |
| 动作表示 | 离散token / 连续回归 / 扩散生成各有优劣 |
| Co-fine-tuning | 混合VLM数据训练，有效缓解灾难性遗忘 |
| 冻结策略 | 根据数据规模和场景选择合适的参数冻结方案 |

---

> **下一节**: [4.4 跨具身体迁移](./04-cross-embodiment.md) - 不同机器人的数据如何互相帮助？

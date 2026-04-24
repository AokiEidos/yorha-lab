---
title: "OpenVLA 详解"
date: 2026-04-20T16:34:20.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# OpenVLA 详解

> ⚙️ 进阶 | 前置知识：[RT-2 详解](./foundation-models-03-rt2)，[Octo 详解](./4-octo)

## 开源 VLA 的基线

**OpenVLA**（Stanford & UC Berkeley, 2024，发表于 *CoRL 2024*）是第一个真正意义上的开源 VLA——基于 7B VLM，在 Open X-Embodiment 上预训练，完全开源权重和代码。它为社区提供了一个可复现、可微调的 VLA 基线。

与闭源的 RT-2 相比，OpenVLA 在保持相当性能的同时，将 VLA 的门槛从"需要 Google 级计算资源"降低到"一张 A100 即可微调"。截至 2024 年末，OpenVLA 在 HuggingFace 上的下载量超过 100K，成为社区最广泛使用的 VLA 基线。

## 完整架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                     OpenVLA Architecture (7.6B)                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │           Prismatic VLM 双视觉编码器融合                    │     │
│  │                                                         │     │
│  │  RGB 图像 (224×224)                                      │     │
│  │       │                                                 │     │
│  │       ├──────────────────┐                              │     │
│  │       │                  │                              │     │
│  │       ▼                  ▼                              │     │
│  │  ┌──────────┐      ┌──────────┐                        │     │
│  │  │ SigLIP   │      │ DINOv2   │                        │     │
│  │  │ ViT-SO   │      │ ViT-L/14 │                        │     │
│  │  │ 400M     │      │ 304M     │                        │     │
│  │  │          │      │          │                        │     │
│  │  │ 语义理解  │      │ 空间细节  │                        │     │
│  │  │ 图文对比  │      │ 自监督    │                        │     │
│  │  │ 训练     │      │ 训练     │                        │     │
│  │  └────┬─────┘      └────┬─────┘                        │     │
│  │       │ 256 token       │ 256 token                    │     │
│  │       │ (1024-d)        │ (1024-d)                     │     │
│  │       └────────┬────────┘                              │     │
│  │                ▼                                        │     │
│  │       [Concat → 512 token (2048-d)]                    │     │
│  │                │                                        │     │
│  │                ▼                                        │     │
│  │       [MLP 投影层: 2048-d → 4096-d]                     │     │
│  │       (2 层 MLP with GELU)                              │     │
│  │                │                                        │     │
│  │                ▼                                        │     │
│  │       512 个视觉 Token (4096-d, 匹配 Llama 维度)         │     │
│  └──────────┬──────────────────────────────────────────────┘     │
│             │                                                    │
│             ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Llama 2 7B Backbone                         │     │
│  │                                                         │     │
│  │  输入序列:                                                │     │
│  │  [视觉Token (512)] [文本Token (变长)] → 自回归生成          │     │
│  │                                                         │     │
│  │  参数: 32 层, d=4096, 32 头, 6.7B 参数                    │     │
│  │  上下文: 2048 Token                                       │     │
│  │                                                         │     │
│  │  输出: 7 个动作 Token (自回归, 每维 256 bin)               │     │
│  └──────────┬──────────────────────────────────────────────┘     │
│             │                                                    │
│             ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │           动作反量化 (De-tokenization)                    │     │
│  │                                                         │     │
│  │  7 个 Token ID → 7 个 bin index → 7 维连续动作            │     │
│  │  [32128, 32191, 32064, 32200, 32133, 32100, 32255]      │     │
│  │     ↓       ↓       ↓       ↓       ↓       ↓       ↓   │     │
│  │  [Δx,     Δy,     Δz,   Δroll, Δpitch, Δyaw, grip]     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  总参数: 7.6B (SigLIP 400M + DINOv2 304M + MLP ~8M + Llama 6.7B)│
└──────────────────────────────────────────────────────────────────┘
```

## Prismatic VLM 双编码器融合详解

OpenVLA 基于 **Prismatic VLM**（Karamcheti et al., 2024），其核心创新是使用两个互补的视觉编码器：

| 编码器 | 架构 | 参数量 | 预训练方式 | 擅长 | Token 维度 |
|--------|------|-------|-----------|------|-----------|
| **SigLIP ViT-SO400M** | ViT-SO/14 | 400M | 对比学习 (图文对) | 语义理解 ("这是杯子") | 256×1024 |
| **DINOv2 ViT-L/14** | ViT-L/14 | 304M | 自监督 (MAE+DINO) | 空间细节 (边界、深度) | 256×1024 |

### 为什么需要双编码器

单一视觉编码器无法同时兼顾语义和空间：

```
场景: 桌面上有一个红色方块和一个蓝色方块，相距 3cm

SigLIP (语义):  "红色方块在蓝色方块的左边"  ← 知道是什么
                但空间位置模糊 (±2cm 误差)

DINOv2 (空间):  精确的物体边界和相对位置     ← 知道在哪里
                但不知道"红色方块"这个概念
                
融合后:         知道目标是什么 + 精确位置    ← 才能执行抓取
```

### 融合方式

```python
# 双编码器融合伪代码
class PrismaticVisionEncoder(nn.Module):
    def __init__(self, llm_dim=4096):
        super().__init__()
        self.siglip = SigLIPViT(model="ViT-SO400M/14")  # 语义
        self.dinov2 = DINOv2ViT(model="ViT-L/14")        # 空间
        
        # 投影层: 将拼接的特征映射到 LLM 维度
        combined_dim = 1024 + 1024  # SigLIP + DINOv2
        self.projector = nn.Sequential(
            nn.Linear(combined_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    
    def forward(self, image):  # image: (B, 3, 224, 224)
        # 各自提取特征
        siglip_tokens = self.siglip(image)   # (B, 256, 1024)
        dinov2_tokens = self.dinov2(image)    # (B, 256, 1024)
        
        # 拼接: 对应位置的 token 拼在一起
        fused = torch.cat([siglip_tokens, dinov2_tokens], dim=-1)  # (B, 256, 2048)
        
        # 投影到 LLM 维度
        visual_tokens = self.projector(fused)  # (B, 256, 4096)
        
        # 注意: 论文中还会将 256 token 保留或进一步处理为 512
        # 实际实现中两个编码器的 token 先各自保留再拼接 → 512 token
        return visual_tokens
```

实际实现中，SigLIP 和 DINOv2 各产生 256 个 Token，**先拼接为 512 个 Token（每个 2048 维）**，再通过 MLP 投射到 Llama 的 4096 维空间。

## 动作 Token 化：Bin 计算公式

OpenVLA 使用 **256 bin 均匀量化**，但与 RT-2 的关键区别是：OpenVLA **创建了新的 Token ID** 加入 Llama 的词表。

### 量化公式

对于第 $d$ 维动作 $a_d \in [a_{\min}^d, a_{\max}^d]$：

$$\text{bin}_d = \text{clip}\left(\left\lfloor \frac{a_d - a_{\min}^d}{a_{\max}^d - a_{\min}^d} \times 255 \right\rfloor, 0, 255\right)$$

$$\text{Token ID}_d = \text{vocab\_size\_original} + \text{bin}_d = 32000 + \text{bin}_d$$

### 反量化公式

$$a_d = a_{\min}^d + \frac{\text{bin}_d + 0.5}{256} \times (a_{\max}^d - a_{\min}^d)$$

注意反量化时加 0.5（bin 中心），减少量化误差。

### 各维度的范围和精度

| 动作维度 | 范围 | 每 bin 精度 | 对抓取的足够性 |
|---------|------|-----------|-------------|
| $\Delta x$ | [-0.05, 0.05] m | 0.39 mm | 足够 |
| $\Delta y$ | [-0.05, 0.05] m | 0.39 mm | 足够 |
| $\Delta z$ | [-0.05, 0.05] m | 0.39 mm | 足够 |
| $\Delta \text{roll}$ | [-0.25, 0.25] rad | 1.96 mrad ≈ 0.11° | 足够 |
| $\Delta \text{pitch}$ | [-0.25, 0.25] rad | 1.96 mrad ≈ 0.11° | 足够 |
| $\Delta \text{yaw}$ | [-0.25, 0.25] rad | 1.96 mrad ≈ 0.11° | 足够 |
| gripper | [0, 1] | 0.004 | 足够 |

## 训练配方

### 预训练

| 超参数 | 值 |
|-------|-----|
| 基座模型 | Prismatic-7B (Llama 2 7B + SigLIP + DINOv2) |
| 预训练数据 | Open X-Embodiment 子集, ~970K 条轨迹 |
| 训练目标 | Next-token prediction (交叉熵损失) |
| Co-fine-tuning | 是（混合视觉问答数据） |
| 优化器 | AdamW |
| 学习率 | $2 \times 10^{-5}$ (cosine decay) |
| 预热步数 | 2000 |
| Batch size | 2048 |
| 训练步数 | ~100K |
| 精度 | bf16 混合精度 |
| 训练硬件 | 64× A100 80GB |
| 训练时间 | ~14 天 |
| 总 GPU 小时 | ~21,504 GPU-hours |

### 损失函数

标准自回归交叉熵损失，仅计算动作 Token 部分：

$$\mathcal{L} = -\sum_{d=1}^{7} \log p_\theta(\text{bin}_d^* | \text{image}, \text{instruction}, \text{bin}_{<d}^*)$$

其中 $\text{bin}_d^*$ 是第 $d$ 维动作的真实 bin index。

## 微调：LoRA vs 全量微调

OpenVLA 支持两种微调方式，论文做了详细对比：

### LoRA 微调

**LoRA（Low-Rank Adaptation）** 只训练低秩增量矩阵，冻结原始权重：

$$W_{\text{new}} = W_{\text{original}} + \Delta W = W_{\text{original}} + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$。

```python
# OpenVLA LoRA 微调配置
lora_config = {
    "r": 32,                    # LoRA 秩
    "lora_alpha": 32,           # 缩放因子
    "target_modules": [         # 应用 LoRA 的模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力
        "gate_proj", "up_proj", "down_proj",       # FFN
    ],
    "lora_dropout": 0.05,
    "trainable_params": "~23M (0.3% of 7.6B)",
}
```

### 全量微调

解冻所有参数，在新数据上端到端训练。

### 对比实验

| 维度 | LoRA (r=32) | 全量微调 |
|------|------------|---------|
| 可训练参数 | 23M (0.3%) | 7.6B (100%) |
| GPU 显存 | ~24 GB (单 A100) | ~65 GB (需 A100 80GB) |
| 训练速度 | ~2x 快 | 基线 |
| WidowX 简单任务 | 82% | 85% |
| WidowX 复杂任务 | 68% | 74% |
| Franka 桌面任务 | 61% | 67% |
| **数据量 100 条** | 72% | 65% |
| **数据量 1000 条** | 78% | 84% |
| 灾难性遗忘 | 低 (冻结大部分权重) | 高 (需要正则化) |

**关键发现**：
- 数据量少时（<200 条），LoRA 反而优于全量微调（更不容易过拟合）
- 数据量充足时（>500 条），全量微调效果更好
- LoRA 对单 GPU 用户是最佳选择（24 GB 显存即可）

## HuggingFace 推理代码

```python
# OpenVLA 完整推理示例 (使用 HuggingFace Transformers)
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np

# 1. 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",           # 自动分配到 GPU
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

# 2. 准备输入
image = Image.open("current_observation.jpg")  # 当前相机图像
instruction = "pick up the red block and place it on the plate"

# 3. 构造 prompt
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# 4. 推理
inputs = processor(prompt, image, return_tensors="pt").to(model.device)
with torch.no_grad():
    action_tokens = model.generate(
        **inputs,
        do_sample=False,           # 贪心解码
        max_new_tokens=7,          # 7 维动作 → 7 个 token
        temperature=1.0,
    )

# 5. 解码为连续动作
# action_tokens 是 token ID，需要映射回 bin index
action_token_ids = action_tokens[0, -7:]    # 最后 7 个 token
bin_indices = action_token_ids - 32000       # 减去原始词表大小
continuous_action = dequantize(bin_indices)   # 反量化

# 6. 执行
# continuous_action: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
robot.execute(continuous_action)
```

### LoRA 微调代码

```python
# OpenVLA LoRA 微调示例
from transformers import AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1. 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 23M || all params: 7.6B || 0.30%

# 3. 训练
training_args = TrainingArguments(
    output_dir="./openvla-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # 有效 batch = 32
    learning_rate=2e-5,
    num_train_epochs=50,
    bf16=True,
    logging_steps=100,
    save_steps=5000,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=my_robot_dataset,  # 自定义数据集
)
trainer.train()
```

## 基准评测结果

### Google Robot (真机)

| 方法 | 简单拾取 | 定向拾取 | 抽屉操作 | 移动物体 | 平均 |
|------|---------|---------|---------|---------|------|
| RT-1 | 89% | 73% | 61% | 68% | 72.8% |
| RT-2-X | 92% | 81% | 70% | 74% | 79.3% |
| Octo-Base | 78% | 62% | 48% | 55% | 60.8% |
| **OpenVLA** | **90%** | **78%** | **67%** | **71%** | **76.5%** |

### WidowX (真机, Bridge V2 任务)

| 方法 | 单物体拾放 | 多物体场景 | 语言条件 | 平均 |
|------|-----------|-----------|---------|------|
| Octo-Base | 62% | 45% | 38% | 48.3% |
| RT-2-X | 75% | 63% | 58% | 65.3% |
| **OpenVLA** | **72%** | **58%** | **53%** | **61.0%** |
| OpenVLA + LoRA 微调 | 85% | 72% | 68% | 75.0% |
| OpenVLA + 全量微调 | 88% | 76% | 71% | 78.3% |

### 跨机器人零样本对比

| 目标机器人 | OpenVLA | Octo-Base | RT-2-X |
|-----------|---------|----------|--------|
| Google Robot | 55% | 42% | 62% |
| WidowX | 38% | 25% | 48% |
| Franka | 22% | 8% | 35% |

## 局限性

1. **单步预测**：每次只预测 1 步动作（无 Action Chunking），控制频率受限于 ~5 Hz
2. **推理延迟**：7B 模型的自回归生成 7 个 Token 需要 ~200ms（使用 A100）
3. **离散化精度**：256 bin 的量化精度 (~0.4mm) 对精密装配任务不足
4. **单模态动作**：自回归生成是单峰分布，无法处理多模态动作
5. **固定分辨率**：输入图像固定为 224×224，丢失精细空间信息
6. **无本体感觉**：不接收关节角度等本体状态，仅依赖视觉

## 小结

| 概念 | 要点 |
|------|------|
| 定位 | 开源 7B VLA 基线，HuggingFace 下载 100K+ |
| 双编码器 | SigLIP (语义 400M) + DINOv2 (空间 304M)，MLP 融合为 512 Token |
| 动作表示 | 256 bin 离散 Token，新 Token ID 32000-32255 |
| 训练 | 970K 轨迹, 64×A100, 14 天, lr=2e-5, Co-fine-tuning |
| LoRA 微调 | 23M 参数 (0.3%)，单 A100 24GB，少数据时优于全量微调 |
| 基准性能 | Google Robot 76.5%，接近闭源 RT-2-X (79.3%) |
| 开源生态 | HuggingFace Transformers 原生支持，3 行代码加载 |

---

> **下一篇**：[π₀ 详解](./06-pi-zero)

---
title: "4.6 高效微调"
date: 2026-04-20T16:08:37.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 4.6 高效微调

> **难度**: ⚙️进阶 | **前置阅读**: [4.3 预训练与微调范式](./03-pretrain-finetune.md), [2.1 整体架构](../02-architecture/01-overview.md)

## 为什么需要高效微调

VLA模型通常基于数十亿参数的VLM构建。全量微调需要大量GPU显存和计算资源，而机器人实验室往往只有有限的硬件。**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 技术只更新少量参数即可适配新任务，大幅降低训练成本。

| 微调方式 | 可训练参数 | 显存需求 (7B模型) | 训练速度 |
|---------|-----------|------------------|---------|
| 全量微调 | 100% (~7B) | ~60 GB | 基准 |
| LoRA | ~0.5% (~35M) | ~20 GB | 2-3x快 |
| Adapter | ~2% (~140M) | ~25 GB | 1.5-2x快 |
| Prompt Tuning | <0.1% (~1M) | ~18 GB | 3-4x快 |

---

## LoRA 在VLA中的应用 ⚙️进阶

**LoRA（Low-Rank Adaptation）** 通过在Transformer的权重矩阵旁注入低秩分解矩阵来实现高效微调，是目前VLA领域最流行的PEFT方法。

### 原理

```
原始权重矩阵 W (d×d)          LoRA改造后
                              W + ΔW
     ┌─────┐                  ┌─────┐   ┌───┐ ┌───┐
x →  │  W  │ → y        x →  │  W  │ + │ A │×│ B │ → y
     └─────┘            (冻结) └─────┘   └───┘ └───┘
                                         d×r   r×d
                                         (r << d, 如 r=16)
可训练参数: d×d              可训练参数: 2×d×r
例: 4096×4096 = 16M          例: 2×4096×16 = 131K (减少99.2%)
```

### VLA中的LoRA配置

```python
from peft import LoraConfig, get_peft_model

# VLA LoRA 配置
lora_config = LoraConfig(
    r=32,                       # 秩，VLA常用16-64
    lora_alpha=64,              # 缩放因子，通常为 2×r
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj",     # 注意力层的Q和V矩阵
        "k_proj", "o_proj",     # 可选：K和输出矩阵
        "gate_proj", "up_proj", # 可选：FFN层
    ],
    modules_to_save=[
        "action_head",          # 动作解码头始终全量训练
    ],
)

# 应用到VLA模型
vla_model = load_pretrained_vla("openvla-7b")
peft_model = get_peft_model(vla_model, lora_config)

# 查看可训练参数
peft_model.print_trainable_parameters()
# 输出: trainable params: 35,127,296 (0.47% of 7,478,000,000)
```

### LoRA的VLA适配要点

| 设计选择 | 推荐 | 原因 |
|---------|------|------|
| 目标层 | Q/V矩阵 + FFN | 覆盖注意力和前馈，性价比最高 |
| 秩 r | 32-64 | 机器人任务比NLP复杂，需较高秩 |
| 视觉编码器 | 冻结或极低秩LoRA | 视觉特征通用性高，改动少 |
| 动作头 | 全量训练 | 参数少、任务特异性强 |
| LLM层 | LoRA | 核心融合与推理层 |

---

## Adapter 方法 ⚙️进阶

**Adapter** 在Transformer每一层中插入小型瓶颈网络，只训练这些插入的模块。

### 结构

```python
class AdapterLayer(nn.Module):
    """Transformer层内的Adapter模块"""
    def __init__(self, hidden_dim, bottleneck_dim=64):
        self.down = nn.Linear(hidden_dim, bottleneck_dim)  # 降维
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)    # 升维
        self.scale = 0.1  # 残差缩放
    
    def forward(self, x):
        residual = x
        x = self.up(self.activation(self.down(x)))
        return residual + self.scale * x

class TransformerWithAdapter(nn.Module):
    """在原始Transformer层后插入Adapter"""
    def __init__(self, original_layer, adapter_dim=64):
        self.original = original_layer  # 冻结
        self.adapter = AdapterLayer(
            original_layer.hidden_size, adapter_dim
        )
    
    def forward(self, x):
        x = self.original(x)    # 原始层（冻结）
        x = self.adapter(x)     # Adapter（可训练）
        return x
```

### Adapter在VLA中的特殊用途

Adapter除了用于高效微调，还被用于[4.4 跨具身体迁移](./04-cross-embodiment.md)中的具身体特定适配：

```python
# 多具身体Adapter方案
class MultiEmbodimentVLA(nn.Module):
    def __init__(self, base_vla, embodiment_list):
        self.base = base_vla        # 冻结共享主干
        self.adapters = nn.ModuleDict({
            name: AdapterLayer(base_vla.hidden_dim, bottleneck_dim=128)
            for name in embodiment_list
        })
    
    def forward(self, image, instruction, embodiment):
        features = self.base.encode(image, instruction)  # 共享
        adapted = self.adapters[embodiment](features)     # 具身体特定
        return self.base.action_head(adapted)
```

---

## Prompt Tuning ⚙️进阶

**Prompt Tuning（提示调优）** 在输入序列前添加可学习的虚拟token，不修改模型任何参数。

```python
class PromptTunedVLA(nn.Module):
    """Prompt Tuning for VLA"""
    def __init__(self, base_vla, n_prompt_tokens=20):
        self.base = base_vla  # 完全冻结
        
        # 可学习的软提示（唯一可训练参数）
        self.soft_prompts = nn.Parameter(
            torch.randn(n_prompt_tokens, base_vla.hidden_dim)
        )
        # 参数量: 20 × 4096 = 81,920 (极少)
    
    def forward(self, image, instruction):
        visual_tokens = self.base.vision_encoder(image)
        text_tokens = self.base.tokenize(instruction)
        
        # 在序列前拼接软提示
        input_sequence = torch.cat([
            self.soft_prompts.expand(batch_size, -1, -1),
            visual_tokens,
            text_tokens,
        ], dim=1)
        
        hidden = self.base.llm(input_sequence)
        return self.base.action_head(hidden)
```

### Prompt Tuning的局限性

| 方面 | 表现 |
|------|------|
| 参数效率 | 最优（<0.1%参数） |
| 单任务适配 | 效果良好 |
| 多任务泛化 | 较弱 |
| 新具身体适配 | 动作空间变化时不够灵活 |
| 适用场景 | 同一机器人的任务切换（如不同抓取目标） |

---

## 方法对比与选择 🔬深入

### 性能对比（基于OpenVLA-7B在Bridge V2上的实验）

| 方法 | 可训练参数 | 成功率 | 训练时间 | GPU显存 |
|------|-----------|--------|---------|---------|
| 全量微调 | 7.0B (100%) | 73% | 48h (8×A100) | 480 GB |
| LoRA (r=32) | 35M (0.5%) | 70% | 12h (2×A100) | 40 GB |
| LoRA (r=64) | 70M (1.0%) | 72% | 16h (2×A100) | 45 GB |
| Adapter (d=128) | 140M (2.0%) | 69% | 18h (2×A100) | 50 GB |
| Prompt Tuning | 0.8M (0.01%) | 61% | 6h (1×A100) | 18 GB |
| 仅Action Head | 2M (0.03%) | 55% | 4h (1×A100) | 16 GB |

### 选择决策树

```
开始
 ├─ GPU资源充足（8+ A100）？
 │    ├─ 是 → 数据量大（>50K轨迹）？
 │    │        ├─ 是 → 全量微调
 │    │        └─ 否 → LoRA (r=64) + Co-fine-tuning
 │    └─ 否 → 继续
 │
 ├─ GPU资源有限（1-2 GPU）？
 │    ├─ 需要多任务/多具身体？
 │    │    ├─ 是 → LoRA (r=32) 或 Adapter
 │    │    └─ 否 → LoRA (r=16)
 │    └─ 继续
 │
 └─ 极有限资源（单卡消费级GPU）？
      ├─ 同一机器人换任务 → Prompt Tuning
      └─ 新机器人适配 → LoRA (r=8) + 量化 (QLoRA)
```

---

## QLoRA：更极致的效率 🔬深入

**QLoRA（Quantized LoRA）** 将基础模型量化为4-bit，在此基础上应用LoRA，进一步降低显存需求。

```python
from transformers import BitsAndBytesConfig

# QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4量化
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # 双重量化
)

# 加载4-bit量化模型
model = AutoModelForCausalLM.from_pretrained(
    "openvla-7b",
    quantization_config=bnb_config,
)

# 在量化模型上应用LoRA
peft_model = get_peft_model(model, lora_config)
# 显存: ~12 GB (单张RTX 4090即可微调7B VLA)
```

| 方法 | 模型精度 | 显存 (7B) | 性能损失 |
|------|---------|----------|---------|
| 全量 FP32 | 32-bit | ~120 GB | - |
| LoRA BF16 | 16-bit | ~20 GB | -1~3% |
| QLoRA 4-bit | 4-bit (base) + 16-bit (LoRA) | ~12 GB | -3~5% |

---

## 本节小结

| 要点 | 说明 |
|------|------|
| LoRA | VLA高效微调的首选方法，0.5%参数达到接近全量微调的性能 |
| Adapter | 适合多具身体场景，每种机器人一个轻量级Adapter |
| Prompt Tuning | 参数最少但表现受限，适合简单任务切换 |
| QLoRA | 4-bit量化 + LoRA，单张消费级GPU即可微调7B模型 |
| 选择策略 | 根据GPU资源、数据规模、任务复杂度综合选择 |

---

> **下一节**: [4.7 开源工具链](./07-toolchains.md) - 有哪些现成的训练工具可以使用？

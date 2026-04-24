---
title: "适配器与微调方法"
date: 2026-04-20T17:27:58.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 适配器与微调方法

> ⚙️ 进阶 | 前置知识：[Latent Diffusion 与 Stable Diffusion](../02-models-zoo/04-latent-diffusion)

## 概述

除了 ControlNet 和 IP-Adapter 外，还有一系列轻量级适配和微调方法，让用户可以教模型学习新概念（如特定人物、特定风格）或注入额外控制信号。这些方法的核心取舍是：**用多少参数、多少数据、多少时间来换取什么程度的个性化效果**。

## T2I-Adapter

**T2I-Adapter（Text-to-Image Adapter）**（Mou et al., 2023）是一种比 ControlNet 更轻量的控制方案。

### 与 ControlNet 的架构对比

```
ControlNet:                          T2I-Adapter:
                                     
条件图 → [完整 U-Net 编码器副本]     条件图 → [小型 4 层卷积编码器]
          ~361M 参数                          ~77M 参数
             ↓ (Zero Conv)                       ↓ (直接加法)
          加到主 U-Net 各层             加到主 U-Net 的编码器特征

核心区别: ControlNet 复制编码器保持特征对齐
         T2I-Adapter 用小网络直接提取控制特征
```

| 方面 | ControlNet | T2I-Adapter |
|------|-----------|-------------|
| 额外参数 | ~361M | ~77M |
| 连接方式 | 零卷积（可学习缩放） | 直接特征加法（固定） |
| 控制精度 | 最高 | 略低（~90%） |
| 训练 GPU 时间 | 3-5 天 (8xA100) | ~1 天 (8xA100) |
| 推理额外开销 | +40% | +15% |
| 适用场景 | 需要精确控制 | 轻量控制足够 |

## LoRA（Low-Rank Adaptation）

**LoRA（低秩适配，Low-Rank Adaptation）**（Hu et al., 2021）最初为大语言模型设计，后被广泛用于 Stable Diffusion 的微调。它是当前 SD 社区最流行的个性化方法。

### 数学原理

将模型权重的更新分解为低秩矩阵：

$$W' = W + \Delta W = W + BA$$

其中：
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是原始权重（冻结）
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$，$A \in \mathbb{R}^{r \times d_{\text{in}}}$，$r \ll \min(d_{\text{in}}, d_{\text{out}})$
- $r$ 是秩（rank），通常 $r = 4 \sim 64$

**为什么低秩有效？** 🔬 研究表明（Aghajanyan et al., 2020），预训练模型在微调时的权重变化 $\Delta W$ 具有**低本征维度（Low Intrinsic Dimensionality）**——即 $\Delta W$ 可以被低秩矩阵良好近似。直觉：微调只是在预训练知识的基础上做"微调"，所需的信息量远小于原始参数量。学习一个特定画师风格可能只需要修改几千个方向，而非数亿个独立参数。

### Rank 选择分析

| Rank | 可训练参数 (SD 1.5) | 文件大小 | 表达力 | 过拟合风险 | 适用场景 |
|------|---------------------|---------|--------|-----------|---------|
| 1 | ~0.5M | ~2 MB | 很低 | 极低 | 极简风格调整 |
| 4 | ~2M | ~8 MB | 低 | 低 | **简单风格** |
| 8 | ~4M | ~16 MB | 中 | 中低 | **通用推荐** |
| 16 | ~8M | ~32 MB | 中高 | 中 | 复杂风格 |
| 32 | ~16M | ~64 MB | 高 | 中高 | 复杂角色 |
| 64 | ~32M | ~128 MB | 很高 | 高 | 接近全量微调 |
| 128 | ~64M | ~256 MB | 最高 | 很高 | 特殊需求 |

**经验法则**：
- 风格 LoRA：$r = 4 \sim 16$（风格是"整体偏移"，低秩足够）
- 角色 LoRA：$r = 8 \sim 32$（角色需要更多细节）
- 概念 LoRA：$r = 4 \sim 8$（单个概念信息量不大）

### 缩放因子 Alpha

实际实现中使用缩放因子 $\alpha$：

$$W' = W + \frac{\alpha}{r} \cdot BA$$

$\alpha$ 控制 LoRA 的全局影响强度。常见设置 $\alpha = r$（即缩放因子为 1）或 $\alpha = 1$（较弱影响）。

### 在 SD 中的应用层

```python
class LoRALayer(nn.Module):
    """LoRA 层实现"""
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_layer  # 冻结
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        self.lora_A = nn.Linear(d_in, rank, bias=False)    # 降维
        self.lora_B = nn.Linear(rank, d_out, bias=False)   # 升维
        
        nn.init.kaiming_uniform_(self.lora_A.weight)       # A: 正常初始化
        nn.init.zeros_(self.lora_B.weight)                  # B: 零初始化
        # → 初始 ΔW = BA = 0，不改变原始模型行为
        
        self.scale = alpha / rank
    
    def forward(self, x):
        original_out = self.original(x)                     # 冻结路径
        lora_out = self.lora_B(self.lora_A(x)) * self.scale # LoRA 路径
        return original_out + lora_out
    
    def merge(self):
        """推理时合并到原始权重，消除额外计算"""
        self.original.weight.data += (
            self.lora_B.weight @ self.lora_A.weight * self.scale
        )
```

通常对 U-Net 的交叉注意力层（Q、K、V、Out 投影）应用 LoRA：

| 目标层 | SD 1.5 参数 | LoRA (rank=8) | 说明 |
|--------|------------|---------------|------|
| Cross-Attn Q | ~320K/层 × 16 | ~5K/层 × 16 | 影响查询 |
| Cross-Attn K | ~320K/层 × 16 | ~5K/层 × 16 | 影响键 |
| Cross-Attn V | ~320K/层 × 16 | ~5K/层 × 16 | 影响值 |
| Cross-Attn Out | ~320K/层 × 16 | ~5K/层 × 16 | 影响输出 |
| Self-Attn (可选) | 更多 | ~5K/层 | 影响全局特征 |
| **总计** | **~860M** | **~4M** | **<0.5%** |

### SDXL LoRA 特殊性

SDXL（SD XL）的 U-Net 更大（~2.6B 参数），LoRA 的应用有一些差异：

| 方面 | SD 1.5 LoRA | SDXL LoRA |
|------|------------|-----------|
| 基座参数 | ~860M | ~2.6B |
| 推荐 rank | 4-16 | 8-32 |
| 可训练参数 | ~4M (rank=8) | ~12M (rank=8) |
| 文件大小 | ~16 MB | ~50 MB |
| 训练时间 | ~1 小时 (A100) | ~2 小时 (A100) |
| 训练分辨率 | 512x512 | 1024x1024 |
| 额外要求 | — | 需要两个文本编码器 |

## Textual Inversion

**Textual Inversion（文本反转）**（Gal et al., 2022）不修改模型权重，而是学习新的**文本嵌入向量**来代表新概念。

### 优化过程

```python
def textual_inversion_training(sd_model, concept_images, placeholder="<my-concept>",
                                 num_steps=3000, lr=5e-4):
    """
    Textual Inversion 训练
    
    核心：冻结所有模型参数，只优化一个嵌入向量
    """
    # 初始化新的嵌入向量（从相似词初始化效果更好）
    token_id = tokenizer.add_tokens(placeholder)
    initial_embedding = text_encoder.get_input_embeddings().weight[
        tokenizer.encode("dog")[1]  # 如果概念是狗，用 "dog" 初始化
    ].clone()
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.get_input_embeddings().weight.data[token_id] = initial_embedding
    
    # 只有这一个向量是可训练的
    trainable_params = [text_encoder.get_input_embeddings().weight[token_id]]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    templates = [
        "a photo of {}",
        "a rendering of {}",
        "a cropped photo of {}",
        "the photo of {}",
        "a good photo of {}",
    ]
    
    for step in range(num_steps):
        # 随机选择模板和图像
        template = random.choice(templates)
        prompt = template.format(placeholder)
        x_0 = random.choice(concept_images)
        
        # 标准扩散训练损失
        t = torch.randint(0, T, (1,))
        eps = torch.randn_like(x_0)
        z_t = noise_schedule.add_noise(x_0, eps, t)
        text_emb = text_encoder(tokenizer(prompt))
        eps_pred = sd_model.unet(z_t, t, text_emb)
        
        loss = F.mse_loss(eps_pred, eps)
        loss.backward()
        
        # 只更新 placeholder 的嵌入
        optimizer.step()
        optimizer.zero_grad()
```

**训练后效果**：
```
训练前: "a photo of <my-cat>" → 无意义输出
训练后: "a photo of <my-cat>" → 你的猫
        "<my-cat> wearing a hat" → 你的猫戴帽子
        "<my-cat> in Van Gogh style" → 梵高风格的你的猫
```

## DreamBooth

**DreamBooth**（Ruiz et al., 2023）通过微调模型来学习特定概念，效果比 Textual Inversion 好得多。

### 先验保持损失的推导

DreamBooth 的关键创新是**先验保持损失（Prior Preservation Loss）**：

$$L = \underbrace{\mathbb{E}_{t,\epsilon}[\| \epsilon - \epsilon_\theta(z_t, t, c_{\text{[V] class}}) \|^2]}_{\text{重建损失：学习目标概念}} + \lambda \cdot \underbrace{\mathbb{E}_{t,\epsilon}[\| \epsilon - \epsilon_\theta(z_t^{pr}, t, c_{\text{class}}) \|^2]}_{\text{先验保持损失：防止遗忘类别知识}}$$

**为什么需要先验保持？**

不加先验保持时：
```
训练数据: 5 张特定狗的图片，prompt = "a [V] dog"
问题: 模型把 "dog" 的概念也改成了你的狗
      → "a dog" 现在只能生成你的狗
      → "a cat playing with a dog" 中的狗也变成你的狗
```

先验保持损失的做法：
```
1. 用原始 SD 生成 ~200 张 "a dog" 的图像（先验样本）
2. 训练时同时优化:
   - 让 "a [V] dog" 生成你的狗
   - 让 "a dog" 仍然能生成各种狗
3. 两个损失加权组合，λ 通常 = 1.0
```

### DreamBooth 训练配方

```python
def dreambooth_training(sd_model, concept_images, class_name="dog", 
                         identifier="sks", lr=5e-6, num_steps=800):
    """DreamBooth 训练"""
    # 1. 生成先验保持样本（用原始模型生成 ~200 张同类图像）
    prior_images = [sd_model.generate(f"a {class_name}") for _ in range(200)]
    
    for step in range(num_steps):
        # 2. 重建损失：让 "a sks dog" 生成你的狗
        loss_concept = diffusion_loss(sd_model, random.choice(concept_images),
                                       f"a {identifier} {class_name}")
        # 3. 先验保持：让 "a dog" 仍能生成各种狗
        loss_prior = diffusion_loss(sd_model, random.choice(prior_images),
                                     f"a {class_name}")
        # 4. 加权组合
        loss = loss_concept + 1.0 * loss_prior
        loss.backward(); optimizer.step()
```

### 训练超参数（全量 vs LoRA）

| 超参数 | DreamBooth 全量 | DreamBooth + LoRA |
|--------|----------------|-------------------|
| 学习率 | 5e-6 | 1e-4 |
| 训练步数 | 800-1200 | 500-800 |
| Batch size | 1-2 | 1-2 |
| 先验样本数 | 200 | 200 |
| 训练时间 (A100) | ~30 分钟 | ~15 分钟 |
| 输出文件大小 | ~4 GB | ~50-100 MB |
| 训练数据 | 3-5 张 | 3-5 张 |
| 过拟合风险 | 中（需要早停） | 低 |

## 方法选择决策树

| 需求 | 首选方法 | 备选方法 |
|------|---------|---------|
| 控制空间结构（精确） | ControlNet (~361M) | T2I-Adapter (~77M) |
| 迁移参考图风格（通用） | IP-Adapter (~22M) | LoRA per-style |
| 生成特定人物/物品（高质量） | DreamBooth + LoRA (~4M) | DreamBooth 全量 |
| 生成特定概念（极轻量） | Textual Inversion (~1K) | — |
| 多维度联合控制 | CFG + ControlNet + IP-Adapter + LoRA | 全兼容 |

更详细的选型对比参见 [条件生成方法对比](./07-comparison)。

## 综合对比

| 方法 | 训练数据 | 可训练参数 | 训练时间 | 个性化质量 | 文件大小 |
|------|---------|-----------|---------|-----------|---------|
| Textual Inversion | 3-5 张 | ~1K | ~30 分钟 | 中等 | ~几 KB |
| LoRA (rank=8) | 10-50 张 | ~4M | ~1 小时 | 好 | ~16 MB |
| DreamBooth | 3-5 张 | ~860M | ~30 分钟 | 最好 | ~4 GB |
| DreamBooth + LoRA | 3-5 张 | ~4M | ~15 分钟 | 很好 | ~50-100 MB |
| T2I-Adapter | 100K+ | ~77M | ~1 天 | — | ~300 MB |
| ControlNet | 100K-3M | ~361M | 3-5 天 | — | ~1.4 GB |

## 小结

| 方法 | 核心思路 | 适用场景 | 关键超参 |
|------|---------|---------|---------|
| T2I-Adapter | 轻量特征注入 | 结构控制（轻量替代） | ~77M 参数 |
| LoRA | $W' = W + BA$，低秩更新 | 风格/角色/画风 | rank=4~64, alpha |
| Textual Inversion | 学习新词嵌入 | 轻量个性化 | lr=5e-4, ~3000 步 |
| DreamBooth | 微调模型 + 先验保持损失 | 高质量个性化 | lr=5e-6, ~800 步 |
| DreamBooth + LoRA | 兼得两者优点 | **社区最流行方案** | lr=1e-4, ~500 步 |

---

> **下一篇**：[条件生成方法对比](./07-comparison)

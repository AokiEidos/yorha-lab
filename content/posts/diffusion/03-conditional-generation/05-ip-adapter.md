---
title: "IP-Adapter"
date: 2026-04-20T17:22:34.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# IP-Adapter

> ⚙️ 进阶 | 前置知识：[ControlNet](./04-controlnet)

## 核心问题

ControlNet 擅长空间结构控制，但不擅长风格和内容参考——"按照这张图的风格生成"或"生成一个和这张图相似的角色"。**IP-Adapter（Image Prompt Adapter，图像提示适配器）**（Ye et al., 2023）让用户可以用参考图像作为"图像提示"来引导生成。

IP-Adapter 解决了一个长期痛点：文本难以精确描述视觉风格（"这种光影感觉""那种色调氛围"），但用一张参考图就能轻松传达。

## 设计思路：解耦交叉注意力

### 为什么不能直接拼接？

在 SD 中，交叉注意力层已经被文本条件占用。如果直接把图像特征和文本特征拼接在一起喂给交叉注意力：

```
朴素方案（效果差）:
Q(图像) × K(concat[文本, 图像]) → Attention → V(concat[文本, 图像])

问题: 文本和图像的嵌入空间不同（CLIP text vs CLIP image），
      拼接后注意力分数不均衡，文本信号被图像信号淹没（或反之）
```

### 解耦交叉注意力架构

**解耦交叉注意力（Decoupled Cross-Attention）**：为图像条件添加**独立的** K、V 投影层，与文本的交叉注意力并行运行。

```
┌─────────────────────────────────────────────────────────────────┐
│               解耦交叉注意力 (Decoupled Cross-Attention)          │
│                                                                  │
│  U-Net 中间特征 Z ──→ Q = W_Q · Z         [B, HW, 320]         │
│                                                                  │
│  ┌─── 文本分支（冻结）─────────────────────────────────────────┐ │
│  │ 文本嵌入 c_text [B, 77, 768]                                │ │
│  │ K_text = W_K · c_text     [B, 77, 320]                     │ │
│  │ V_text = W_V · c_text     [B, 77, 320]                     │ │
│  │ Attn_text = softmax(Q · K_text^T / √d) · V_text            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── 图像分支（可训练）───────────────────────────────────────┐ │
│  │ 图像嵌入 c_img [B, N_img, 1024]  (N_img=1 或 257)          │ │
│  │ K_img = W_K' · c_img      [B, N_img, 320]   ← 独立投影层   │ │
│  │ V_img = W_V' · c_img      [B, N_img, 320]   ← 独立投影层   │ │
│  │ Attn_img = softmax(Q · K_img^T / √d) · V_img               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  最终输出 = Attn_text + λ · Attn_img                             │
│             ↑ 冻结       ↑ scale 参数控制图像条件强度              │
└─────────────────────────────────────────────────────────────────┘
```

### 完整前向传播代码

```python
class DecoupledCrossAttention(nn.Module):
    """IP-Adapter 解耦交叉注意力层"""
    def __init__(self, dim=320, text_dim=768, image_dim=1024, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        
        # 文本分支（从预训练 SD 加载，冻结）
        self.to_q = nn.Linear(dim, dim, bias=False)         # 共享 Q
        self.to_k_text = nn.Linear(text_dim, dim, bias=False)
        self.to_v_text = nn.Linear(text_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # 图像分支（新增，可训练）
        self.to_k_ip = nn.Linear(image_dim, dim, bias=False)   # 独立 K'
        self.to_v_ip = nn.Linear(image_dim, dim, bias=False)   # 独立 V'
    
    def forward(self, x, text_emb, image_emb, scale=1.0):
        """
        x: U-Net 特征 [B, HW, dim]
        text_emb: CLIP 文本嵌入 [B, 77, text_dim]
        image_emb: CLIP 图像嵌入 [B, N_img, image_dim]
        scale: 图像条件缩放因子 λ
        """
        q = self.to_q(x)  # [B, HW, dim]
        
        # === 文本交叉注意力（冻结权重）===
        k_text = self.to_k_text(text_emb)   # [B, 77, dim]
        v_text = self.to_v_text(text_emb)   # [B, 77, dim]
        out_text = self._attention(q, k_text, v_text)
        
        # === 图像交叉注意力（可训练权重）===
        k_img = self.to_k_ip(image_emb)     # [B, N_img, dim]
        v_img = self.to_v_ip(image_emb)     # [B, N_img, dim]
        out_img = self._attention(q, k_img, v_img)
        
        # === 合并 ===
        out = self.to_out(out_text + scale * out_img)
        return out
    
    def _attention(self, q, k, v):
        """标准多头注意力"""
        B, N, _ = q.shape
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return out
```

## 图像编码器

IP-Adapter 使用 CLIP 图像编码器提取参考图像的特征。根据使用的特征层级，分为两个版本。

### IP-Adapter（基础版）

使用 CLIP ViT 的 **CLS token**（全局特征）：

```
参考图像 224×224 → CLIP ViT-H/14 → CLS token [1, 1024]
                                      ↓
                               Projection MLP
                                      ↓
                               图像嵌入 [1, N_tokens, image_dim]
                               (N_tokens = 4, 通过线性投影展开)
```

- **优点**：紧凑（仅 1 个 token），计算开销小
- **缺点**：丢失空间信息，只能传递整体风格/语义
- 适用：风格迁移、氛围参考

### IP-Adapter Plus（增强版）

使用 CLIP ViT 倒数第二层的 **patch tokens**（局部特征）：

```
参考图像 224×224 → CLIP ViT-H/14 → 倒数第二层输出 [257, 1280]
                                      ↓                (1 CLS + 256 patch)
                               Perceiver Resampler
                               (16 个 learnable queries)
                                      ↓
                               图像嵌入 [16, image_dim]
```

- **优点**：保留空间细节，内容保真度更高
- **缺点**：更多 token，计算稍高
- 适用：角色一致性、内容参考

### 对比

| 版本 | 图像 Token 数 | 特征来源 | 内容保真度 | 风格迁移 | 额外参数 |
|------|-------------|---------|-----------|---------|---------|
| IP-Adapter | 4 | CLS token | 中等 | 好 | ~22M |
| IP-Adapter Plus | 16 | Patch tokens + Resampler | 好 | 很好 | ~32M |
| IP-Adapter Full | 257 | 全部 token | 最好 | 最好 | ~46M |

## 训练详情

### 训练数据

- **数据集**：LAION-2B 的子集（~10M 高质量图文对）
- 每个样本：(图像, 文本描述)
  - 图像同时作为生成目标和条件参考
  - 训练时对参考图像做数据增强（裁剪、颜色抖动），避免模型学到"直接复制"

### 训练配置

| 超参数 | 值 | 说明 |
|--------|------|------|
| 基座模型 | SD 1.5 / SDXL | 冻结 |
| 可训练参数 | ~22M (基础版) | 仅图像分支的 K'/V' |
| 学习率 | 1e-4 | 使用余弦退火 |
| Batch size | 8 per GPU | |
| 训练 GPU | 8x V100 32GB | ~1 周 |
| 训练步数 | 1M steps | |
| 图像编码器 | CLIP ViT-H/14 | 冻结 |

### 训练损失

与标准扩散训练相同的 MSE 损失，唯一区别是额外输入图像嵌入：

$$L = \mathbb{E}_{t, \epsilon}\left[\| \epsilon - \epsilon_\theta(z_t, t, c_{\text{text}}, c_{\text{image}}) \|^2\right]$$

## Scale 参数 $\lambda$ 的调优

### 效果对照

| $\lambda$ | 文本控制力 | 图像相似度 | 典型用途 |
|----------|----------|-----------|---------|
| 0.0 | 100%（纯文本） | 无 | 基线对照 |
| 0.3-0.5 | 高 | 轻微风格影响 | 轻度风格参考 |
| **0.6-0.8** | **平衡** | **明显参考** | **推荐默认** |
| 1.0 | 中等 | 强参考 | 角色一致性 |
| 1.2-1.5 | 弱 | 非常强 | 近似复制 |

**实践建议**：
- 风格迁移：$\lambda = 0.5 \sim 0.7$（保留文本的内容控制力）
- 角色参考：$\lambda = 0.8 \sim 1.0$（更忠于参考图像的外观）
- 与 ControlNet 组合时：$\lambda = 0.5 \sim 0.7$（给 ControlNet 留空间）

## IP-Adapter FaceID 变体

**IP-Adapter-FaceID**（2024）专门用于人脸一致性生成：

```
参考人脸 → InsightFace (ArcFace) → 人脸 ID 嵌入 [1, 512]
                                         ↓
                                   MLP Projection
                                         ↓
                                   [1, N, dim] → 解耦交叉注意力
```

与标准 IP-Adapter 的区别：
- 使用 **InsightFace** 而非 CLIP 编码人脸 ID
- InsightFace 的嵌入更关注**身份特征**（面部结构）而非视觉相似度
- 可以在不同姿态、光照下保持人脸一致性

| 方法 | 身份保持 | 姿态灵活性 | 额外训练 |
|------|---------|-----------|---------|
| IP-Adapter (CLIP) | 中等 | 高 | 通用训练 |
| IP-Adapter FaceID | 高 | 高 | 需人脸数据 |
| DreamBooth (人脸) | 最高 | 中等 | 需要 per-ID 微调 |

## 与 ControlNet 的组合架构

IP-Adapter 和 ControlNet 解决不同维度的控制问题，**可以同时使用**：

```
┌────────────────────────────────────────────────────────────┐
│              IP-Adapter + ControlNet 组合架构                │
│                                                            │
│  参考图像 → CLIP Image Encoder → 图像嵌入                   │
│                                      ↓                     │
│  条件图 → ControlNet 控制分支 → 控制特征                     │
│                ↓ (特征加法)            ↓ (解耦注意力)        │
│  噪声 z_t →  U-Net Encoder  ←──────  Cross-Attention ←──  │
│                   ↓                                        │
│              U-Net Decoder → z_0 预测                       │
│                                                            │
│  文本 → CLIP Text → Cross-Attention (原始)                  │
│                                                            │
│  三路控制:                                                   │
│  1. 文本 → 语义内容 (cross-attention, 原始)                  │
│  2. ControlNet → 空间结构 (特征加法, 零卷积)                 │
│  3. IP-Adapter → 视觉风格 (解耦 cross-attention)            │
└────────────────────────────────────────────────────────────┘
```

**组合效果对比**：

| 组合方式 | 内容控制 | 结构控制 | 风格控制 | 总推理开销 |
|---------|---------|---------|---------|-----------|
| 纯文本 (CFG) | 好 | 差 | 差 | 2x（CFG） |
| + ControlNet | 好 | 很好 | 差 | 2.4x |
| + IP-Adapter | 好 | 差 | 好 | 2.2x |
| + ControlNet + IP-Adapter | 好 | 很好 | 好 | 2.6x |

## 与其他风格迁移方法的对比

| 方法 | 训练需求 | 推理时间 | 风格保真度 | 内容自由度 | 参数量 |
|------|---------|---------|-----------|-----------|--------|
| LoRA 风格微调 | 需要 per-style 训练 (~1h) | 无额外开销 | 很好 | 好 | ~4M/style |
| Textual Inversion | 需要 per-concept 优化 (~30min) | 无额外开销 | 中等 | 好 | ~1K/concept |
| **IP-Adapter** | **一次训练通用** | +10% 开销 | **好** | **好** | **22M (共享)** |
| Style Transfer (NST) | 不需要 | 较慢 | 好 | 差（强耦合） | — |
| InstructPix2Pix | 一次训练通用 | 无额外 | 中等 | 好 | — |
| DreamBooth | 需要 per-concept 微调 | 无额外开销 | 最好 | 中等 | ~860M/concept |

**IP-Adapter 的核心优势**：一次训练，任意参考图即用——不需要为每个新风格重新训练。

## 小结

| 概念 | 要点 |
|------|------|
| 解耦交叉注意力 | 图像和文本各自独立的 K/V 投影，并行计算 |
| 图像编码器 | CLIP ViT-H/14，CLS（基础版）或 patch tokens（Plus） |
| 训练开销 | 只训练 K'/V' 投影，~22M 参数，8xV100 ~1 周 |
| Scale $\lambda$ | 控制图像参考强度，推荐 0.6-0.8 |
| FaceID 变体 | 使用 InsightFace 编码人脸 ID，身份保持更好 |
| 与 ControlNet 互补 | ControlNet 控结构 + IP-Adapter 控风格 = 全方位控制 |
| vs LoRA/DreamBooth | 无需 per-style 训练，一次训练通用 |

---

> **下一篇**：[适配器与微调方法](./06-adapters)

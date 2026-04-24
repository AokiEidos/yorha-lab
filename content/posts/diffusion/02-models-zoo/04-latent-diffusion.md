---
title: "Latent Diffusion 与 Stable Diffusion"
date: 2026-04-17T17:49:24.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Latent Diffusion 与 Stable Diffusion

> ⚙️ 进阶 | 前置知识：[DDPM 详解](./01-ddpm)，了解 VAE 和 Transformer 注意力机制

## 从像素空间到潜空间

在像素空间做扩散面临严峻的计算挑战：一张 512×512×3 的 RGB 图像有 786,432 个维度。在如此高维的空间中运行 U-Net 上千步——训练和推理都极其昂贵。

**潜空间扩散模型（Latent Diffusion Model, LDM）**（Rombach et al., 2022）的核心思想：先用一个预训练的 VAE 将图像压缩到低维**潜空间（Latent Space）**，然后在潜空间中做扩散。

```
像素空间方案:   512×512×3  →  U-Net 扩散  →  512×512×3
                786K 维度     计算极大

潜空间方案:     512×512×3  → [VAE 编码] → 64×64×4  →  U-Net 扩散  →  64×64×4  → [VAE 解码] → 512×512×3
                                          16K 维度     计算 ~48× 节省
```

### 为什么这样做可行

Rombach et al. 的关键观察：图像中大部分信息是**感知上不可见的高频细节**（如纹理的精确像素值），而人眼关注的**语义信息**可以用更紧凑的表示捕获。

VAE 编码器充当"语义压缩器"——丢弃感知上不重要的细节，保留语义。扩散模型只需要在这个语义空间中学习生成。

### 压缩比的选择

| 下采样因子 $f$ | 潜空间尺寸（512 输入） | 压缩比 | 重建质量 |
|-------------|-------------------|--------|---------|
| $f=4$ | 128×128 | 16× | 极高 |
| $f=8$ | 64×64 | 64× | 高（常用） |
| $f=16$ | 32×32 | 256× | 中等 |
| $f=32$ | 16×16 | 1024× | 较低 |

$f=8$ 是最常用的选择（Stable Diffusion 使用），在压缩率和重建质量之间取得了好的平衡。

## LDM 架构

### 完整推理流水线

```
文本 "a cat sitting on a beach"
  │
  ▼
[文本编码器 (CLIP/T5)]
  │
  ▼ 文本嵌入 (77×768)
  │
  ▼ Cross-Attention
  │
[U-Net (潜空间扩散)]  ←── 噪声 z_T ~ N(0,I), shape: 64×64×4
  │
  ▼ 去噪后的潜变量 z_0, shape: 64×64×4
  │
[VAE 解码器]
  │
  ▼
生成图像 512×512×3
```

### 组件 1：VAE（自编码器）

LDM 使用的 VAE 独立于扩散模型训练，包含：

- **编码器** $\mathcal{E}$：将图像 $x \in \mathbb{R}^{H \times W \times 3}$ 编码为 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$
- **解码器** $\mathcal{D}$：将潜变量重建为图像 $\tilde{x} = \mathcal{D}(z) \approx x$

VAE 训练使用感知损失（Perceptual Loss）+ 对抗损失（GAN Loss）+ KL 正则化，确保潜空间分布接近标准正态分布。

### 组件 2：条件 U-Net

潜空间中的 U-Net 与 DDPM 的 U-Net 结构类似，但增加了**交叉注意力（Cross-Attention）** 层来注入条件信息：

```python
class CrossAttentionBlock(nn.Module):
    def forward(self, x, context):
        # x: 潜空间特征 [B, HW, C]
        # context: 文本嵌入 [B, L, D]（L=77 tokens, D=768）
        
        Q = self.to_q(x)          # Query 来自图像特征
        K = self.to_k(context)    # Key 来自文本
        V = self.to_v(context)    # Value 来自文本
        
        attn = softmax(Q @ K.T / sqrt(d_k))
        out = attn @ V            # 文本信息注入图像特征
        return out
```

每个 U-Net 块的结构：ResBlock → Self-Attention → Cross-Attention

- **Self-Attention**：图像特征内部的全局关系
- **Cross-Attention**：将文本信息注入图像特征——这是文本控制生成的关键机制

### 组件 3：文本编码器

将文本 prompt 编码为嵌入向量序列，供交叉注意力使用。

## Stable Diffusion 家族演化

### Stable Diffusion 1.x（2022）

| 组件 | 选择 |
|------|------|
| VAE | $f=8$，4 通道潜空间 |
| 文本编码器 | CLIP ViT-L/14（768 维） |
| U-Net | ~860M 参数 |
| 训练数据 | LAION-5B（50 亿图文对） |
| 分辨率 | 512×512 |
| 参数化 | $\epsilon$-预测 |

SD 1.x 是第一个开源的大规模文生图模型，引爆了社区生态。

### Stable Diffusion 2.x（2022）

主要改进：
- 文本编码器升级为 OpenCLIP ViT-H（1024 维），更强的文本理解
- 支持 768×768 分辨率
- 切换到 $v$-预测参数化
- 去除了 NSFW 内容的训练

但社区反馈 SD 2.x 在某些风格上不如 1.x——可能因为训练数据过滤过于激进。

### Stable Diffusion XL（SDXL, 2023）

重大架构升级：

- **双文本编码器**：CLIP ViT-L + OpenCLIP ViT-bigG，拼接后更丰富的文本表示
- **更大的 U-Net**：~2.6B 参数（3× SD 1.x）
- **更高分辨率**：1024×1024
- **Refiner 模型**：两阶段生成——Base 模型生成整体结构，Refiner 模型精炼细节
- **Micro-conditioning**：将原始分辨率和裁切坐标作为额外条件输入

### Stable Diffusion 3（SD3, 2024）

革命性架构变化：

- **MM-DiT（Multimodal Diffusion Transformer）**：用 Transformer 替代 U-Net
  - 图像和文本 token 在同一 Transformer 中交互
  - 使用联合注意力（Joint Attention）而非交叉注意力
- **三文本编码器**：CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL
- **Flow Matching 训练**：替代传统的 DDPM 训练框架
- **Rectified Flow**：更直的采样路径，更少的采样步数

```
SD 1.x/2.x 架构:  文本 → CLIP → Cross-Attn → U-Net → VAE
SDXL 架构:        文本 → 双 CLIP → Cross-Attn → 大 U-Net → VAE → Refiner
SD3 架构:         文本 → 三编码器 → Joint-Attn → MM-DiT → VAE
```

### 演化趋势

| 趋势 | 方向 |
|------|------|
| 骨干网络 | U-Net → DiT / MM-DiT |
| 文本编码 | 单 CLIP → 多编码器 + T5 |
| 条件注入 | Cross-Attention → Joint Attention |
| 训练框架 | DDPM → Flow Matching |
| 模型规模 | ~1B → ~8B 参数 |

## 小结

| 概念 | 要点 |
|------|------|
| 潜空间扩散 | VAE 压缩 → 潜空间扩散 → VAE 解码，~48× 节省计算 |
| 交叉注意力 | Q 来自图像，K/V 来自文本，实现文本控制 |
| SD 1.x | 开源里程碑，CLIP + U-Net + VAE |
| SDXL | 更大模型 + 双编码器 + 1024 分辨率 |
| SD3 | MM-DiT + 三编码器 + Flow Matching，架构革新 |

---

> **下一篇**：[Consistency Models](./05-consistency-models)

---
title: "文生视频技术"
date: 2026-04-20T17:19:00.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 文生视频技术

> ⚙️ 进阶 | 前置知识：[DiT 架构深度解析](./02-dit)，[采样算法](../01-fundamentals/04-sampling-algorithms)

## 核心挑战

**文生视频（Text-to-Video, T2V）** 在文生图基础上增加了**时间维度**，从生成一张静态图像扩展到生成一段时间连贯的视频序列。这个看似简单的维度扩展带来了质的挑战：

| 挑战 | 描述 | 技术难点 |
|------|------|---------|
| **时间一致性（Temporal Consistency）** | 相邻帧内容、颜色、光照必须连贯 | 避免时间闪烁（Temporal Flickering），人物外观漂移 |
| **运动建模** | 物体运动需要物理合理 | 重力、惯性、碰撞、流体等物理规律 |
| **计算量爆炸** | 视频比图像多一个维度 | 16 帧 × 512² ≈ 单帧 16 倍计算量；1 秒 24fps 1080p ≈ 5000 万像素/秒 |
| **训练数据** | 高质量视频-文本对稀缺 | 视频标注成本远高于图像；视频数据存储和加载开销巨大 |
| **长时生成** | 维持数十秒到数分钟的一致性 | 内存不足以一次性处理所有帧；质量随长度衰减 |

---

## 技术路线一：从图像模型扩展（插入时间层）

最直觉的方法——在预训练的图像扩散模型基础上**插入时间注意力层（Temporal Attention）**，将 2D 架构扩展为处理 3D 时空数据。

### 3D U-Net 架构图

```
  文本编码器                              3D U-Net 去噪骨干
  ┌────────┐                    ┌──────────────────────────────────────┐
  │ T5-XXL │──text_emb─────────▶│                                      │
  └────────┘                    │  ┌─────────────────┐   ┌─────────┐  │
                                │  │ Down Block 1    │   │ Up Block│  │
  视频潜变量 z_t                 │  │ ResBlock3D      │──▶│  3      │  │
  [B,T,4,64,64]                │  │ SpatialAttn     │   │ +Skip   │  │
       │                        │  │ TemporalAttn    │   └────┬────┘  │
       ▼                        │  │ CrossAttn(text) │        │       │
  ┌──────────┐                  │  │ Downsample      │        ▲       │
  │ 3D Conv  │──────────────────▶  └────────┬────────┘        │       │
  │ 入口     │                  │           ▼                  │       │
  └──────────┘                  │  ┌─────────────────┐   ┌────┴────┐  │
                                │  │ Down Block 2    │   │ Up Block│  │
                                │  │ (同样结构)       │──▶│  2      │  │
                                │  └────────┬────────┘   └────┬────┘  │
                                │           ▼                  ▲       │
                                │  ┌─────────────────┐   ┌────┴────┐  │
                                │  │ Middle Block    │──▶│ Up Block│  │
                                │  │ SpatialAttn     │   │  1      │  │
                                │  │ TemporalAttn    │   └─────────┘  │
                                │  │ CrossAttn(text) │                │
                                │  └─────────────────┘                │
                                └──────────────────────────────────────┘
```

### 时空注意力完整伪代码 🔬

**时空分离注意力（Factorized Spatial-Temporal Attention）** 是当前视频生成中最常用的设计。核心思想：将全 3D 注意力分解为独立的空间注意力和时间注意力，大幅降低计算量。

```python
from einops import rearrange

class SpatioTemporalAttentionBlock(nn.Module):
    """时空分离注意力块——视频扩散模型的核心组件"""
    
    def __init__(self, dim, num_heads, num_frames):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # 时间注意力初始化为零（保留预训练图像模型能力）
        nn.init.zeros_(self.temporal_attn.out_proj.weight)
        nn.init.zeros_(self.temporal_attn.out_proj.bias)
    
    def forward(self, x, text_emb):
        """
        x:        [B, T, C, H, W]  — T 帧视频特征
        text_emb: [B, L, C]        — 文本嵌入
        """
        B, T, C, H, W = x.shape
        
        # ── 空间注意力：每帧独立，帧内所有空间位置交互 ──
        # 将 B 和 T 合并，每帧独立做空间自注意力
        x_spatial = rearrange(x, 'b t c h w -> (b t) (h w) c')
        x_spatial = self.spatial_norm(x_spatial)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x = x + rearrange(attn_out, '(b t) (h w) c -> b t c h w', b=B, t=T, h=H, w=W)
        
        # ── 时间注意力：同一空间位置跨帧交互 ──
        # 将 B, H, W 合并，每个空间位置独立做时间自注意力
        x_temporal = rearrange(x, 'b t c h w -> (b h w) t c')
        x_temporal = self.temporal_norm(x_temporal)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x = x + rearrange(attn_out, '(b h w) t c -> b t c h w', b=B, h=H, w=W)
        
        # ── 交叉注意力：文本条件注入 ──
        x_cross = rearrange(x, 'b t c h w -> (b t) (h w) c')
        x_cross = self.cross_norm(x_cross)
        text_expanded = text_emb.repeat(T, 1, 1)  # 每帧共享文本条件
        attn_out, _ = self.cross_attn(x_cross, text_expanded, text_expanded)
        x = x + rearrange(attn_out, '(b t) (h w) c -> b t c h w', b=B, t=T, h=H, w=W)
        
        return x
```

**为什么时间注意力初始化为零？** 这个技巧被称为 **零初始化膨胀（Zero-Init Inflation）**。预训练的图像模型已经学会了优秀的空间生成能力。插入新的时间注意力层时，如果初始化为零，那么整个模块在训练初期就是恒等映射——视频模型退化为"逐帧独立的图像模型"。训练过程中逐渐学会利用时间注意力建模帧间关系。这避免了破坏已学到的空间生成能力。

### 计算量分析

| 注意力方式 | 复杂度 | 16 帧 32×32 (token 数) | 实际 FLOPs |
|-----------|--------|----------------------|-----------|
| 全 3D 注意力 | $O((T \cdot H \cdot W)^2)$ | $(16384)^2$ | ~268M tokens 交互 |
| 分离时空 | $O(T \cdot (HW)^2 + HW \cdot T^2)$ | $16 \times 1024^2 + 1024 \times 16^2$ | ~17M，约 16x 节省 |

---

## 技术路线二：Spacetime DiT（Sora 路线）🔬

将视频视为 3D 数据，用 **3D Patch 化** 处理——这是 Sora 开创的路线。

### 3D Patchification 原理

```
视频潜变量: [B, C, T, H, W] = [B, 4, 16, 64, 64]
    │
    ▼ 3D Patch 化 (p_t × p_h × p_w = 2 × 2 × 2)
    │
    T' = T/p_t = 8,  H' = H/p_h = 32,  W' = W/p_w = 32
    每个 patch: C × p_t × p_h × p_w = 4 × 2 × 2 × 2 = 32 维
    │
    ▼ 线性投影 32 → D
    │
Spacetime tokens: [B, 8×32×32, D] = [B, 8192, D]
    │
    ▼ + 3D 位置编码 (时间+空间)
    │
    ▼ Full Transformer Attention (所有 token 间交互)
    │
    ▼ Unpatchify → [B, 4, 16, 64, 64]
```

### Sora 的关键设计分析

基于 OpenAI 技术报告（2024.02）的公开信息，Sora 的核心创新包括：

| 设计选择 | 具体做法 | 为什么这样做 |
|---------|---------|-------------|
| **可变分辨率和时长** | 不固定 patch 数量，直接在原始宽高比上训练 | 避免裁剪/填充导致的构图失真；支持任意输出尺寸 |
| **Spacetime DiT** | 全注意力而非分离时空 | 时间和空间信息自然融合，无需手动设计交互方式 |
| **视频-图像联合训练** | 图像视为"单帧视频" | 利用海量图像数据增强视觉质量 |
| **渐进式训练** | 从低分辨率短视频逐步扩展 | 降低初始训练成本，稳定训练 |
| **重描述（Recaptioning）** | 用 LLM 为训练视频生成详细描述 | 提升 prompt 遵循度（类似 DALL-E 3 的 caption 改进策略） |

---

## 技术路线三：自回归 + 扩散混合

逐帧或逐段生成，每段以前面的帧为条件：

```
初始帧:  Frame 1-4 ← 扩散生成（文本条件）
延续段1: Frame 5-8 ← 扩散生成（条件于 Frame 1-4 + 文本）
延续段2: Frame 9-12 ← 扩散生成（条件于 Frame 5-8 + 文本）
  ...可以无限延续
```

**优势**：理论上可以生成任意长度的视频
**劣势**：长期一致性依赖条件传递，容易累积漂移（drift）；分段边界处可能出现不连续

---

## 视频 VAE：时空压缩 ⚙️

视频 VAE 需要同时压缩**空间**和**时间**维度：

```
原始视频:      [B, 3, 24, 512, 512]     (24 帧, 512×512)
                    │
空间压缩 (f=8):     │  512/8 = 64
时间压缩 (f_t=4):   │  24/4 = 6
通道扩展:           │  3 → 4 (或 16)
                    ▼
潜空间:        [B, 4, 6, 64, 64]        压缩比 ≈ 384x
```

| 视频 VAE 设计 | 空间压缩 $f$ | 时间压缩 $f_t$ | 潜通道 | 压缩比 | 使用模型 |
|--------------|-------------|---------------|--------|--------|---------|
| SVD (Stability) | 8 | 1 (不压缩) | 4 | 64× | Stable Video Diffusion |
| CogVideoX VAE | 8 | 4 | 16 | 512× | CogVideoX |
| Sora VAE (推测) | 8 | ~4 | ~16 | ~512× | Sora |
| Open-Sora VAE | 8 | 4 | 4 | 128× | Open-Sora |

**为什么要压缩时间维度？** 相邻帧之间高度相似（视频的时间冗余远大于空间冗余）。不压缩时间的 SVD 方案导致序列长度线性增长，计算量巨大。CogVideoX 等方案通过 3D 卷积 VAE 同时压缩时空，使 4 秒 24fps 视频（96 帧）压缩为仅 24 个时间步的潜变量。

---

## 运动表示与控制方法

| 方法 | 输入形式 | 控制精度 | 代表工作 |
|------|---------|---------|---------|
| **光流条件（Optical Flow）** | 密集位移场 [T, 2, H, W] | 像素级精确 | VideoComposer, DragNUWA |
| **轨迹条件（Trajectory）** | 关键点坐标序列 [(x,y,t)] | 物体级 | DragAnything, MotionCtrl |
| **摄像机参数** | 旋转/平移矩阵 [T, 3, 4] | 全局运动 | CameraCtrl, MotionCtrl |
| **运动向量（Motion Score）** | 标量/低维向量 | 粗粒度（速度快慢） | AnimateDiff Motion LoRA |
| **参考视频** | 另一段视频的运动 | 运动迁移 | VMC, MotionClone |

---

## 代表性模型对比 🔬

| 模型 | 机构 | 年份 | 架构 | 最大时长 | 最大分辨率 | 开源 | 关键特点 |
|------|------|------|------|---------|-----------|------|---------|
| Sora | OpenAI | 2024.02 | Spacetime DiT | ~60s | 1080p | 否 | 可变分辨率/时长训练 |
| Runway Gen-3 Alpha | Runway | 2024.06 | DiT | 10s | 1080p | 否 | 商业化成熟 |
| Kling 1.5 | 快手 | 2024.06 | 3D DiT | 2min | 1080p | 否 | 长视频+运动控制 |
| CogVideoX | 智谱 | 2024.08 | 3D DiT | 6s | 720p | 是 | Expert Transformer |
| Veo 2 | Google | 2024.12 | DiT (级联) | ~60s | 4K | 否 | 物理一致性强 |
| HunyuanVideo | 腾讯 | 2024.12 | DiT | 5s | 720p | 是 | 双流 Transformer |
| Open-Sora | HPC-AI | 2024 | STDiT | ~16s | 720p | 是 | 开源复现 Sora 路线 |
| Wan2.1 | 阿里 | 2025 | DiT | 5s | 720p | 是 | 图生视频质量高 |

---

## 训练挑战与评估

### 训练资源需求

| 规模 | GPU 数量 | 训练时长 | 显存需求 | 数据规模 |
|------|---------|---------|---------|---------|
| 研究级 (CogVideoX) | ~256 A100 | 数周 | 80GB/卡 | ~1000 万视频 |
| 产业级 (Sora 级) | 数千 H100 | 数月 | 80GB/卡 | 数亿视频 |

### 评估指标

| 指标 | 全称 | 衡量内容 | 说明 |
|------|------|---------|------|
| **FVD↓** | Frechet Video Distance | 视频整体质量 | 类似 FID 但在时空特征空间 |
| **FID↓** | Frechet Inception Distance | 单帧画质 | 抽帧后用图像 FID |
| **CLIPSIM↑** | CLIP Similarity | 文本-视频一致性 | 抽帧 CLIP 评分平均 |
| **时间一致性↑** | Temporal Consistency | 帧间连贯性 | 相邻帧 CLIP/LPIPS 差异 |
| **运动质量** | Motion Quality | 运动自然度 | 光流平滑度 + 人类评测 |

### 视频编辑扩展

视频扩散模型可以自然扩展到视频编辑任务：

- **视频风格迁移**：SDEdit 策略直接推广到视频（逐帧/全局加噪后去噪）
- **视频 Inpainting**：在时空维度上做局部修复（如移除物体的所有帧）
- **视频超分辨率**：条件扩散将低分辨率视频逐帧/联合提升分辨率
- **图像动画化（Image-to-Video）**：以单帧图像为条件生成视频（SVD, I2VGen-XL）

---

## 小结

| 概念 | 要点 |
|------|------|
| 时间一致性 | T2V 的核心挑战，通过时间注意力/3D 全注意力/自回归条件传递解决 |
| 分离时空注意力 | 空间 $O(HW)^2$ + 时间 $O(T)^2$，比全 3D 注意力节省约 16x 计算 |
| Spacetime DiT | 3D patch → 全序列注意力，Sora 路线，时空自然融合但序列极长 |
| 视频 VAE | 同时压缩空间（8x）和时间（4x），实现 100-500x 总压缩比 |
| 运动控制 | 光流/轨迹/相机参数等多种粒度的运动条件注入 |
| 训练挑战 | 数据稀缺 + 计算量巨大 + 评估指标不完善 |

---

> **下一篇**：[图像编辑](./04-image-editing) — Diffusion 模型如何实现精细的图像编辑。

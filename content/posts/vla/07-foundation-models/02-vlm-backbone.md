---
title: "VLM 作为感知骨干"
date: 2026-04-20T16:29:54.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# VLM 作为感知骨干

> ⚙️ 进阶 | 前置知识：[VLM 回顾](../01-foundations/01-vlm-review)，[多模态输入融合](../02-architecture/04-multimodal-fusion)

## 视觉编码器：VLA 的"眼睛"

视觉编码器是 VLA 的感知基础——它决定了模型能"看到"什么、"看清"到什么程度。选择错误的编码器，即使后续的 LLM 和动作头再强也无济于事。本文深入分析各编码器在机器人任务上的特性差异和选型策略。

## 主流视觉编码器对比

### CLIP ViT 系列

**CLIP ViT**（OpenAI, 2021）通过图文对比学习训练，在**语义对齐（Semantic Alignment）** 上极强——它知道图像中"是什么"，也知道如何用语言描述。

```
训练目标: 让"一张猫的照片"的图像特征 ≈ "a photo of a cat"的文本特征
结果: 编码器学会了将视觉概念映射到语言空间
```

在 VLA 中的价值：理解任务指令中提到的物体（"pick up the red cup"→ 识别"red cup"）。

局限：对比学习优化全局匹配，可能牺牲**空间细节**——知道"画面中有一个杯子"但不够精确知道"杯子把手朝哪边"。

### SigLIP

**SigLIP（Sigmoid Loss for Language-Image Pre-training）**（Zhai et al., 2023）是 CLIP 的改进版：

| 改进点 | CLIP | SigLIP |
|--------|------|--------|
| 损失函数 | Softmax（全局归一化） | Sigmoid（逐对独立） |
| Batch Size 依赖 | 强（需要大 batch） | 弱（小 batch 也行） |
| 训练效率 | 需要全局负样本 | 逐对计算，更高效 |
| 最终性能 | 好 | 略好于 CLIP |

SigLIP 的 Sigmoid 损失让每个图文对的匹配判断**独立于 batch 中其他样本**，避免了 Softmax 的全局竞争导致的信息丢失。

**在 VLA 中的采用**：OpenVLA 和 π₀ 都选择了 SigLIP 作为主视觉编码器。

### DINOv2

**DINOv2**（Oquab et al., 2023, Meta）通过**自监督学习（Self-supervised Learning）** 训练——不需要文本配对，只用图像本身。

```
训练目标: 让同一图像的不同裁剪/变换具有相似的特征
         (自蒸馏: 教师网络 → 学生网络)
结果: 编码器学会了图像的空间结构和局部细节
```

DINOv2 的特征在**空间感知**上极强：

| 能力 | CLIP/SigLIP | DINOv2 |
|------|------------|--------|
| 物体边界 | 模糊 | **清晰** |
| 深度线索 | 有限 | **丰富** |
| 纹理细节 | 中等 | **详细** |
| 语义分割 | 中等 | **优秀** |
| 语言对齐 | **优秀** | 无（未用文本训练） |

### EVA-CLIP

**EVA-CLIP**（Fang et al., 2023）是大规模训练的增强版 CLIP：
- 使用 EVA 预训练的 ViT 初始化（更好的视觉特征起点）
- 更大的训练数据和更长的训练
- ViT-G（1B 参数），目前 CLIP 系列最强

### InternViT

**InternViT**（Chen et al., 2024）由上海 AI Lab 训练，专门为中文多模态场景优化，是 InternVL 系列的视觉骨干。

## 语义 vs 空间：核心权衡

这是 VLA 视觉编码器选型的**核心矛盾**：

```
语义理解强 (CLIP/SigLIP)              空间感知强 (DINOv2)
│                                      │
│  "画面中有一个红色杯子"              │  "杯子在坐标 (0.3, 0.2)，
│  → 理解指令中的物体                  │   把手朝右，高 12cm"
│  → 支持开放词汇                      │  → 精确定位和抓取
│  → 零样本泛化                        │  → 深度估计
│                                      │
│  但：杯子具体在哪？朝向？             │  但：这是杯子还是花瓶？
│  → 精度不够 ❌                       │  → 语义理解弱 ❌
```

### 为什么 OpenVLA 用了双编码器

OpenVLA 的解决方案——**双编码器互补（Dual Encoder Complementarity）**：

```
图像 (224×224)
  │
  ├─→ [SigLIP ViT-SO400M/14] → 256 个语义 Token
  │     "这是一个红色马克杯"
  │
  └─→ [DINOv2 ViT-L/14]      → 256 个空间 Token
        "杯子在 (0.32, 0.18)，高 11cm，把手朝右 23°"
        │
        ▼
  [MLP 投影 + 拼接] → 512 个融合 Token → Llama 2 7B
```

实验验证（OpenVLA 论文 Table 3）：

| 视觉编码器 | Google Robot 成功率 | WidowX 成功率 |
|-----------|-------------------|--------------|
| SigLIP only | 71.2% | 42.8% |
| DINOv2 only | 58.4% | 38.5% |
| **SigLIP + DINOv2** | **76.8%** | **48.3%** |

双编码器比任何单一编码器都好——语义+空间的互补性得到了实验验证。

### 为什么 π₀ 只用了 SigLIP

π₀ 选择只用 SigLIP（通过 PaliGemma VLM）：

推测原因：
1. π₀ 依赖 Flow Matching 动作头的强表达力来弥补视觉细节
2. PaliGemma 的 SigLIP 分辨率较高（224→448 可调），部分弥补空间细节
3. 本体感觉（关节角度）提供了另一维度的精确空间信息
4. 架构简洁性考虑——单编码器降低复杂度和推理延迟

## 分辨率与效率的权衡

### 分辨率的影响

| 分辨率 | Patch 数 (patch=14) | Token 数 | 理论 Attention 计算 | 适合场景 |
|--------|---------------------|----------|-------------------|---------|
| 224×224 | 16×16 | 256 | 基线 (1×) | 一般操作 |
| 336×336 | 24×24 | 576 | ~5× | 中等精度 |
| 448×448 | 32×32 | 1024 | ~16× | 精细操作 |
| 672×672 | 48×48 | 2304 | ~81× | 极精细操作 |

关键观察：Attention 计算量与 Token 数的**平方**成正比。分辨率翻倍→Token 数 4 倍→计算量 16 倍。

### 什么时候需要高分辨率

| 任务 | 需要的空间精度 | 推荐分辨率 |
|------|-------------|-----------|
| 粗抓取（大物体） | ~1cm | 224×224 |
| 精抓取（小物体） | ~2mm | 336-448 |
| 精密装配 | ~0.5mm | 448-672 |
| 灵巧手操作 | ~1mm | 336-448 + 手眼相机 |

### Token 压缩策略

为了在高分辨率下控制计算量，有几种 Token 压缩方案：

| 方法 | 原理 | 压缩比 | 信息损失 |
|------|------|--------|---------|
| **TokenLearner** (RT-1) | 可学习的 Token 选择 | 8-16× | 中等 |
| **Perceiver Resampler** (Flamingo) | 可学习查询 Token 交叉注意力 | 可控 | 低 |
| **平均池化** | 相邻 Token 平均 | 4× | 较高 |
| **动态分辨率** (LLaVA-NeXT) | 将高分辨率图像切成子图分别编码 | 无压缩 | 无 |

```python
# TokenLearner 伪代码
class TokenLearner(nn.Module):
    def __init__(self, num_tokens=8):
        super().__init__()
        # 为每个输出 Token 学习一个空间注意力图
        self.attention_maps = nn.Sequential(
            nn.Conv2d(C, num_tokens, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W] (视觉特征图)
        maps = self.attention_maps(x)  # [B, num_tokens, H, W]
        # 每个注意力图加权池化得到一个 Token
        tokens = torch.einsum('bchw,bnhw->bnc', x, maps)  # [B, num_tokens, C]
        return tokens  # 从 H×W 个 Token 压缩到 num_tokens 个
```

## 多视角处理策略

机器人通常有 1-3 个相机（第三人称 + 手眼 + 可选的侧视角）。处理策略：

### 方案 1：共享编码器 + Token 拼接

```
相机 1 图像 → [共享 ViT] → Token₁ (256个)
相机 2 图像 → [共享 ViT] → Token₂ (256个)
                                    │
                              拼接 → Token (512个) → LLM
```

优势：参数共享，高效
劣势：不同视角的特征可能需要不同的处理

### 方案 2：视角特定编码器

```
第三人称 → [ViT_global] → Token₁ (全局场景理解)
手眼相机 → [ViT_local]  → Token₂ (局部精细信息)
                                    │
                              拼接 → LLM
```

较少使用，但理论上更合理——全局和局部视角的最优编码策略可能不同。

### 方案 3：多图拼接后统一编码

```
[图像1 | 图像2 | 图像3] → 拼成大图 → [ViT] → Token
```

简单但浪费——大部分 patch 不含有用信息（拼接边界区域）。

## 视觉编码器选型决策树

```
需要语言指令控制？
  │
  ├─ 是 → 需要精细空间控制？
  │         ├─ 是 → SigLIP + DINOv2 双编码器 (OpenVLA 方案)
  │         └─ 否 → SigLIP 单编码器 (π₀/PaliGemma 方案)
  │
  └─ 否 (目标图像/固定任务) → 需要 3D 信息？
            ├─ 是 → DINOv2 + 深度编码器 (3D-VLA 方案)
            └─ 否 → DINOv2 或 ResNet (Diffusion Policy 方案)
```

## 前沿方向

### 视频编码器

当前大多数 VLA 只处理单帧或少量帧堆叠。视频编码器（如 VideoMAE、InternVideo）可以更好地捕捉时间动态。GR 系列已探索了这个方向。

### 3D 感知视觉编码器

3D-VLA 等工作探索将点云或 3D 特征直接编入视觉 Token。对于需要精确 3D 推理的任务（如从特定角度插入），这可能比 2D+深度更高效。

### 自适应分辨率

根据任务阶段动态调整分辨率：接近目标时提高分辨率（精细定位），远离目标时降低分辨率（全局规划）。类似人眼的注视机制。

## 小结

| 概念 | 要点 |
|------|------|
| CLIP/SigLIP | 语义强，语言对齐好，空间细节中等 |
| DINOv2 | 空间强，边界清晰，无语言对齐 |
| 双编码器 | SigLIP+DINOv2 互补，OpenVLA 验证有效 |
| 分辨率权衡 | 224(粗)→448(精细)，计算量增长平方 |
| Token 压缩 | TokenLearner/Perceiver Resampler 控制序列长度 |
| 选型建议 | 语言控制→SigLIP(+DINOv2)；固定任务→DINOv2/ResNet |

---

> **下一篇**：[Diffusion 作为动作生成器](./03-diffusion-action) — Diffusion Policy 和 Flow Matching 在 VLA 中的应用。

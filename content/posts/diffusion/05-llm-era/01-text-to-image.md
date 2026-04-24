---
title: "文生图技术栈"
date: 2026-04-20T17:27:00.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 文生图技术栈

> ⚙️ 进阶 | 前置知识：[Latent Diffusion 与 Stable Diffusion](../02-models-zoo/04-latent-diffusion)，[CFG](../03-conditional-generation/03-cfg)，[Flow Matching](../02-models-zoo/06-flow-matching)

## 文生图全景

**文生图（Text-to-Image, T2I）** 是 Diffusion 模型最成功的商业应用，也是当前 AIGC 产业的核心能力。从 2022 年 DALL-E 2 / Imagen / Stable Diffusion 掀起的"AI 绘画"浪潮，到 2024 年 SD3 / FLUX 的架构革新，T2I 技术栈经历了三次范式迁移。本文系统梳理完整技术栈和主流路线，帮助读者建立从组件到系统的全景认知。

## 完整 T2I 管线

一个现代 T2I 系统由三大核心组件构成，下图展示了从用户 prompt 到最终图像的完整数据流：

```
                         ┌──────────────────────────────────────────────────────┐
  用户 prompt            │              Text-to-Image Pipeline                  │
  "a corgi on           │                                                      │
   the beach,           │  ┌──────────────┐    ┌─────────────────────────────┐ │
   sunset"              │  │ Text Encoder │    │     Denoising Backbone      │ │
       │                │  │              │    │                             │ │
       ▼                │  │ CLIP-L ──┐   │    │   z_T ~ N(0,I)            │ │
  ┌─────────┐           │  │ OpenCLIP ─┼──▶│c  │     │  [64×64×4]          │ │
  │ Tokenize│──────────▶│  │ T5-XXL ──┘   │──▶│     ▼                     │ │
  └─────────┘           │  │              │    │  ┌──────────┐             │ │
                        │  │ Pool + Seq   │    │  │ U-Net 或 │──step 1──▶z │ │
                        │  └──────────────┘    │  │   DiT    │──step 2──▶z │ │
                        │                      │  │(+CFG/条件)│──...────▶z_0│ │
                        │                      │  └──────────┘             │ │
                        │                      └──────────┬───────────────┘ │
                        │                                 │ z_0 [64×64×4]   │
                        │                                 ▼                  │
                        │                      ┌─────────────────┐          │
                        │                      │   VAE Decoder   │          │
                        │                      │ 64×64×4→512×512 │          │
                        │                      └────────┬────────┘          │
                        └───────────────────────────────┼───────────────────┘
                                                        ▼
                                                  最终图像 [512×512×3]
```

下面逐一深入每个组件的技术细节。

---

## 一、文本编码器：从自然语言到条件向量 🔬

文本编码器负责将自然语言 prompt 转换为去噪网络可以理解的**条件嵌入向量（Conditioning Embedding）**。这个组件直接决定了模型"理解 prompt"的能力上限。

### 编码器详细对比

| 编码器 | 类型 | 嵌入维度 | 最大 Token | 参数量 | 训练方式 | 使用模型 |
|--------|------|---------|-----------|--------|---------|---------|
| CLIP ViT-L/14 | 图文对比 | 768 | 77 | 428M | 对比学习（400M 图文对） | SD 1.x |
| OpenCLIP ViT-H/14 | 图文对比 | 1024 | 77 | 986M | 对比学习（LAION-2B） | SD 2.x |
| OpenCLIP ViT-bigG/14 | 图文对比 | 1280 | 77 | 2.5B | 对比学习（LAION-2B） | SDXL（配合 CLIP-L） |
| T5-XXL | 纯语言 | 4096 | 256 | 4.7B | Span corruption（C4 语料） | Imagen, SD3, FLUX |
| CLIP-L + OpenCLIP-G | 双编码拼接 | 768+1280=2048 | 77 | ~3B | 各自独立预训练 | SDXL |
| CLIP-L + G + T5-XXL | 三编码融合 | 768+1280+4096 | 77+256 | ~8B | 各自独立预训练 | SD3 |

### 为什么需要多编码器？—— 设计决策深度分析

**CLIP 系编码器的局限**：CLIP 通过图文对比学习训练，擅长捕获"图像与文本的整体相关性"，但对复杂语义的理解有限。具体表现：

- **属性绑定弱**："a red car and a blue house" 容易出现颜色错配
- **计数能力差**："three cats" 常生成数量不对的猫
- **空间关系理解有限**："A on top of B" 经常出错
- **77 token 限制**：长 prompt 被截断，丢失关键信息

**T5 类编码器的优势**：T5 是在大规模纯文本语料上训练的编码器-解码器模型，具备更强的语义理解能力：

- 256 token 长度支持复杂详细的描述
- 对属性绑定、计数、空间关系理解更好（因为是语言模型，理解句法结构）
- 4096 维嵌入提供更丰富的语义表示

**多编码器融合策略**（以 SD3 为例）：

```python
# SD3 三编码器融合示意
clip_l_emb = clip_l_encoder(tokens)          # [B, 77, 768]
clip_g_emb = openclip_g_encoder(tokens)      # [B, 77, 1280]
t5_emb = t5_xxl_encoder(tokens)              # [B, 256, 4096]

# 池化嵌入拼接 → 用于 timestep conditioning
pool_emb = concat(clip_l_pool, clip_g_pool)  # [B, 2048] → adaLN

# 序列嵌入 → 用于交叉注意力 / 联合注意力
# CLIP 序列 padding 到 256，与 T5 序列拼接
clip_seq = pad(concat(clip_l_emb, clip_g_emb), max_len=256)  # [B, 256, 2048]
clip_seq = linear_proj(clip_seq)             # [B, 256, 4096]  投影到 T5 维度
text_seq = concat(clip_seq, t5_emb, dim=1)   # [B, 512, 4096]  总文本序列
```

**趋势**：从单编码器（SD 1.x）到双编码器（SDXL）再到三编码器（SD3），模型对文本的理解力持续增强。编码器参数占整个模型的比例也从 ~30% 提升到 ~50% 以上。

---

## 二、去噪骨干网络：U-Net vs DiT ⚙️

骨干网络是 T2I 系统的计算核心，负责在条件引导下逐步去除噪声。两大架构路线——U-Net 和 DiT——各有技术特色。

### 架构对比

```
    U-Net 架构                              DiT 架构
    ─────────                               ─────────
    编码器 (下采样)                          Patchify
      │                                       │
      ├─ ResBlock + SpatialAttn              Linear Proj + PosEmb
      ├─ CrossAttn (文本条件)                   │
      ├─ Downsample                          ┌──────────────┐
      │                                      │  DiT Block   │ × N
    中间层                                    │ adaLN-Zero   │
      │                                      │ Self-Attn    │
    解码器 (上采样)                            │ (或 Joint)   │
      ├─ ResBlock + SpatialAttn              │ MLP          │
      ├─ CrossAttn (文本条件)                 └──────────────┘
      ├─ Skip Connection                         │
      └─ Upsample                            Unpatchify → 输出
```

| 维度 | U-Net | DiT / MM-DiT |
|------|-------|-------------|
| 代表模型 | SD 1.5 (860M), SDXL (2.6B) | SD3 (2B/8B), FLUX (12B) |
| 条件注入 | Cross-Attention（文本→图像单向） | adaLN-Zero / Joint Attention（双向） |
| 归纳偏置 | 强（多尺度、局部性、skip connection） | 弱（依赖数据和规模） |
| Scaling 表现 | 存在瓶颈（~3B 后收益递减） | 遵循 Scaling Laws（越大越好） |
| 分辨率灵活性 | 需要固定分辨率训练 | Patch 化支持可变分辨率 |
| 内存效率 | Skip connection 占显存 | 全注意力，序列长时显存大 |
| 推理延迟 | 较低（架构高效） | 较高（全序列注意力 O(n^2)） |
| 社区生态 | 极成熟（LoRA/ControlNet/IP-Adapter） | 快速成长中 |

### 为什么 DiT 正在取代 U-Net？

核心原因是 **Scaling Laws（缩放定律）**。LLM 领域的核心发现是"模型越大、数据越多、效果越好"的幂律关系。U-Net 由于其多尺度编码器-解码器的固有结构，参数规模难以持续扩展（超过 3B 后架构设计变得困难）。而 DiT 使用标准 Transformer，可以像 LLM 一样简单地堆叠层数来扩大规模，天然适配 Scaling Laws。详见 [DiT 架构深度解析](./02-dit)。

---

## 三、图像解码器（VAE）

**变分自编码器（Variational Autoencoder, VAE）** 负责像素空间和潜空间之间的转换。详细原理参见 [Latent Diffusion](../02-models-zoo/04-latent-diffusion)。

| VAE 版本 | 下采样倍率 $f$ | 潜空间通道 | 潜空间尺寸（512 输入） | 使用模型 |
|----------|-------------|-----------|---------------------|---------|
| SD 1.x VAE | 8 | 4 | 64×64×4 | SD 1.5, SD 2.x |
| SDXL VAE | 8 | 4 | 128×128×4（1024 输入） | SDXL |
| SD3 VAE | 8 | 16 | 64×64×16（512 输入） | SD3, FLUX |

SD3 将潜空间通道从 4 增加到 16，代价是计算量增加，但换来更高质量的图像重建（VAE 重建 PSNR 提升约 1.5 dB）。

---

## 四、分辨率提升策略

### 策略对比

| 策略 | 原理 | 代表 | 优势 | 劣势 |
|------|------|------|------|------|
| 级联生成 | 64²→256²→1024²，多阶段独立模型 | Imagen, DALL-E 2 | 每阶段计算量小，高分辨率细节好 | 需训练多个模型，错误级联，总延迟高 |
| 直接高分辨率 | 在目标分辨率的潜空间直接训练 | SDXL, SD3 | 单模型端到端，架构简洁 | 训练成本极高，需要高效潜空间 |
| 潜空间上采样 | 潜空间低分辨率→高分辨率 | SD Upscaler | 复用预训练模型，灵活 | 两阶段推理，一致性挑战 |
| 多分辨率训练 | 混合多种分辨率训练，推理时选择 | SDXL (bucket), FLUX | 灵活适配不同宽高比 | 训练复杂度增加 |

### 级联生成详细流程（Imagen）

```
Stage 1: Text → 64×64                    Stage 2: 64×64 → 256×256               Stage 3: 256×256 → 1024×1024
┌──────────────────────┐                 ┌──────────────────────┐                ┌──────────────────────┐
│ T5-XXL 编码          │                 │ 条件：Stage 1 输出    │                │ 条件：Stage 2 输出    │
│ Base Diffusion Model │                 │ + T5 文本嵌入         │                │ + T5 文本嵌入         │
│ 2B params            │                 │ SR Diffusion (600M)  │                │ SR Diffusion (400M)  │
│ DDPM, 1000 steps     │                 │ + noise augmentation │                │ + noise augmentation │
└──────────────────────┘                 └──────────────────────┘                └──────────────────────┘
```

Imagen 在级联中使用了**噪声增强（Noise Augmentation）**——对上阶段输出添加少量噪声再输入下阶段，使下阶段模型对上阶段的瑕疵更鲁棒。

---

## 五、主流模型全景对比 🔬

| 模型 | 年份 | 骨干架构 | 文本编码器 | 训练框架 | 最大分辨率 | 参数量 | 开源 | FID↓ (COCO-30K) | CLIP Score↑ |
|------|------|---------|-----------|---------|-----------|--------|------|-----------------|-------------|
| DALL-E 2 | 2022.04 | CLIP Prior + UNet 解码 | CLIP ViT-L | Diffusion | 1024² | ~3.5B | 否 | 10.39 | — |
| Imagen | 2022.05 | 级联 U-Net (3 stage) | T5-XXL (4.7B) | DDPM | 1024² | ~4.6B | 否 | 7.27 | 0.27 |
| SD 1.5 | 2022.10 | U-Net (LDM) | CLIP-L | DDPM(epsilon) | 512² | 860M | 是 | ~12 | ~0.31 |
| SDXL | 2023.07 | 大 U-Net (LDM) | CLIP-L + OpenCLIP-G | DDPM(epsilon) | 1024² | 2.6B | 是 | ~9.5 | ~0.33 |
| DALL-E 3 | 2023.10 | U-Net (级联) | T5 + CLIP | DDPM | 1024² | ~4B | 否 | — | — |
| SD3 Medium | 2024.06 | MM-DiT | CLIP-L + G + T5-XXL | Rectified Flow | 1024² | 2B | 是 | — | — |
| SD3.5 Large | 2024.10 | MM-DiT | CLIP-L + G + T5-XXL | Rectified Flow | 1024² | 8B | 是 | — | — |
| FLUX.1 [dev] | 2024.08 | DiT (单流+双流) | CLIP-L + T5-XXL | Rectified Flow | 1024²+ | 12B | 部分 | — | — |
| Imagen 3 | 2024 | DiT (级联) | T5 | — | 1024²+ | 未公开 | 否 | — | — |

> **注**：FID（Frechet Inception Distance，越低越好）和 CLIP Score（越高越好）是最常用的自动评测指标。但这些指标与人类偏好的相关性有限，2024 年后业界越来越倾向于使用人类偏好评测（ELO rating）和 GenAI-Bench 等综合基准。

### 训练数据格局

| 数据集 | 规模 | 来源 | 使用模型 | 状态 |
|--------|------|------|---------|------|
| LAION-5B | 58 亿图文对 | 网络爬取（Common Crawl） | SD 1.x, SDXL | 因版权争议部分下线 |
| LAION-2B (EN) | 23 亿英文图文对 | LAION-5B 子集 | OpenCLIP 训练 | 同上 |
| 内部数据集（各公司） | 数亿到数十亿 | 授权/合成/内部标注 | DALL-E 3, Imagen 3 | 不公开 |
| 合成数据 | 规模不等 | 现有模型生成 + 过滤 | SD3, FLUX | 趋势增长 |

2023 年后，高质量合成数据（synthetic data）和精细的数据过滤成为核心竞争力。DALL-E 3 的一个关键创新是用 LLM 为训练图像重新生成高质量描述（caption improvement），显著提升了 prompt 遵循度。

### 训练框架演进

| 框架 | 损失目标 | 采样路径 | 代表模型 | 特点 |
|------|---------|---------|---------|------|
| DDPM (epsilon) | 噪声预测 $\|\epsilon - \epsilon_\theta\|^2$ | 马尔可夫链 | SD 1.x, SDXL | 简单稳定，1000 步训练 |
| DDPM (v-prediction) | 速度预测 $\|v - v_\theta\|^2$ | 马尔可夫链 | Imagen, SD 2.x | 高 SNR 范围更稳定 |
| Flow Matching | 速度场 $\|u_t - v_\theta\|^2$ | ODE 直线路径 | SD3, FLUX | 理论优雅，采样效率高 |
| Rectified Flow | 整流路径 $\|(x_1-x_0) - v_\theta\|^2$ | 直线插值 | SD3 | Flow Matching 特例，路径最直 |

从 DDPM 到 Flow Matching 的迁移带来了显著的采样效率提升——SD3 在 28 步采样下即可达到 SD 1.5 在 50 步下的质量，详见 [Flow Matching](../02-models-zoo/06-flow-matching)。

---

## 六、评估指标体系

| 指标 | 全称 | 衡量内容 | 公式/方法 | 局限 |
|------|------|---------|----------|------|
| FID↓ | Frechet Inception Distance | 生成分布与真实分布的距离 | Inception V3 特征空间的 Frechet 距离 | 对多样性和质量混合衡量 |
| CLIP Score↑ | CLIP Similarity | 图文一致性 | CLIP 图像嵌入与文本嵌入的余弦相似度 | 对复杂 prompt 不敏感 |
| IS↑ | Inception Score | 生成质量和多样性 | Inception V3 预测分布的 KL 散度 | 不衡量真实性 |
| HPSv2↑ | Human Preference Score | 人类偏好 | 基于人类偏好数据训练的评分模型 | 可能有偏差 |

---

## 小结

| 概念 | 要点 |
|------|------|
| 文本编码器演进 | CLIP→多编码器→T5 融合，语义理解持续增强 |
| 骨干网络迁移 | U-Net→DiT，遵循 Scaling Laws |
| 分辨率策略 | 级联→直接高分辨率→多分辨率训练 |
| 训练范式 | DDPM→Flow Matching（Rectified Flow） |
| 评估体系 | FID/CLIP Score→人类偏好评测 |
| 数据趋势 | 网络爬取→合成数据 + 高质量 caption |

文生图技术栈从 CLIP+U-Net+DDPM（SD 1.x 时代）演化为多编码器+DiT+Flow Matching（SD3/FLUX 时代），三大组件都经历了根本性升级。理解这些变迁背后的技术逻辑——更强的语义理解、更好的扩展性、更优的采样效率——是理解整个 LLM 时代 Diffusion 技术发展的关键。

---

> **下一篇**：[DiT 架构深度解析](./02-dit) — 深入 Transformer 如何取代 U-Net 成为扩散模型的新骨干。

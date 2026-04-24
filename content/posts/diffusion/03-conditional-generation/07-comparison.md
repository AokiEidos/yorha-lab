---
title: "条件生成方法对比"
date: 2026-04-20T17:25:27.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 条件生成方法对比

> 🔰 入门 | 本文提供快速选型参考
> 前置知识：建议先阅读各方法的详细文档，本文为综合对比总结

## 综合对比表

### 核心维度对比

| 方法 | 额外训练 | 条件类型 | 对预训练模型侵入 | 推理额外开销 | 可训练参数 |
|------|---------|---------|----------------|-------------|-----------|
| Classifier Guidance | 需要训练噪声分类器 | 类别标签 | 无 | +分类器梯度（~3x） | 分类器参数 |
| **CFG** | 无（训练时丢弃条件） | 任意 | 无 | 2x 前向传播 | 0 |
| **ControlNet** | 需要训练控制分支 | 空间结构 | 无（主网冻结） | +40% | ~361M |
| **IP-Adapter** | 需要训练 K'/V' | 参考图像 | 无（主网冻结） | +10% | ~22M |
| T2I-Adapter | 需要训练适配器 | 空间结构 | 无（主网冻结） | +15% | ~77M |
| **LoRA** | 需要微调 | 特定概念/风格 | 低秩更新 | **无**（可合并） | ~4M |
| Textual Inversion | 需要优化嵌入 | 特定概念 | 无 | **无** | ~1K |
| DreamBooth | 需要微调全量 | 特定概念 | 全量微调 | **无** | ~860M |
| DreamBooth+LoRA | 需要微调 LoRA | 特定概念 | 低秩更新 | **无**（可合并） | ~4M |

### 训练成本对比

| 方法 | 训练数据量 | 训练时间 (A100) | GPU 显存需求 | 输出模型大小 |
|------|-----------|----------------|-------------|-------------|
| Classifier Guidance | ImageNet 全集 | 数天 | ~32GB | ~200M |
| CFG | 与主模型一起 | 0（无额外训练） | — | — |
| ControlNet | 100K-3M | 3-5 天 (8xA100) | ~40GB | ~1.4GB |
| IP-Adapter | ~10M | ~1 周 (8xV100) | ~32GB | ~100MB |
| T2I-Adapter | 100K+ | ~1 天 (8xA100) | ~24GB | ~300MB |
| LoRA | 10-50 张 | ~1 小时 (1xA100) | ~16GB | ~16MB |
| Textual Inversion | 3-5 张 | ~30 分钟 (1xA100) | ~12GB | ~几KB |
| DreamBooth | 3-5 张 | ~30 分钟 (1xA100) | ~24GB | ~4GB |
| DreamBooth+LoRA | 3-5 张 | ~15 分钟 (1xA100) | ~16GB | ~50-100MB |

### 质量与控制精度对比

| 方法 | 文本匹配度 | 结构控制精度 | 风格迁移能力 | 个性化质量 | 多样性 |
|------|-----------|------------|------------|-----------|--------|
| Classifier Guidance | — | — | — | — | 中（受 $s$ 控制） |
| CFG | 很好 | — | — | — | 好（受 $w$ 控制） |
| ControlNet | 好 | **最好** | — | — | 好 |
| IP-Adapter | 好 | — | **很好** | — | 好 |
| T2I-Adapter | 好 | 好 | — | — | 好 |
| LoRA | 好 | — | 好 | 好 | 中 |
| Textual Inversion | 好 | — | 中 | 中 | 好 |
| DreamBooth | 好 | — | — | **最好** | 中 |

## 方法组合兼容性矩阵

不同条件生成方法可以自由组合。以下矩阵标注了每种组合的兼容性和注意事项：

| 组合 | 兼容性 | 效果 | 注意事项 |
|------|--------|------|---------|
| CFG + ControlNet | 完全兼容 | 文本+结构双控 | 标准组合，无冲突 |
| CFG + IP-Adapter | 完全兼容 | 文本+图像风格双控 | 适当降低 IP-Adapter scale |
| CFG + LoRA | 完全兼容 | 文本+学习的风格 | LoRA 合并后零开销 |
| CFG + DreamBooth | 完全兼容 | 文本+学习的概念 | 注意 $w$ 不宜太高 |
| CFG + Textual Inversion | 完全兼容 | 文本+新概念词 | 最轻量组合 |
| ControlNet + IP-Adapter | 完全兼容 | 结构+风格双控 | **推荐组合**，互补性最好 |
| ControlNet + LoRA | 完全兼容 | 结构+学习的风格 | LoRA 影响生成风格 |
| ControlNet + ControlNet | 完全兼容 | 多条件联合控制 | 权重和建议 <1.5 |
| IP-Adapter + LoRA | 完全兼容 | 图像参考+风格微调 | 注意风格可能冲突 |
| IP-Adapter + DreamBooth | 兼容 | 图像参考+概念 | 可能出现语义冲突 |
| LoRA + LoRA | 兼容（加法合并） | 多风格叠加 | 权重需调整避免冲突 |
| LoRA + Textual Inversion | 完全兼容 | 风格+概念词 | 互不干扰 |
| DreamBooth + Textual Inv | 兼容 | 概念微调+新词 | 一般不需要同时用 |

### 推荐组合方案

```
场景 1: 高质量可控文生图（商业级）
  CFG (w=7.5) + ControlNet-Depth (w=0.6) + IP-Adapter (λ=0.5)
  → 文本控内容 + 深度控结构 + 参考图控风格

场景 2: 特定角色多姿态生成
  CFG (w=7.5) + DreamBooth+LoRA (角色) + ControlNet-Pose (w=0.8)
  → 文本控场景 + LoRA 锁角色 + 骨骼控姿态

场景 3: 风格化创作
  CFG (w=5.0) + LoRA (画师风格) + IP-Adapter (参考图, λ=0.3)
  → 文本控主题 + LoRA 定大风格 + IP-Adapter 微调氛围

场景 4: 精确结构到图（建筑/产品设计）
  CFG (w=7.5) + ControlNet-Canny (w=0.8) + ControlNet-Depth (w=0.4)
  → 文本控材质 + 边缘控轮廓 + 深度控空间
```

## 具体部署场景推荐

### 场景 A: 在线文生图服务（高并发、低延迟）

**推荐方案**: CFG + 预加载 LoRA 热切换

| 要求 | 选择 | 理由 |
|------|------|------|
| 基础能力 | CFG (w=7.5) | 所有请求必需 |
| 风格选项 | 预训练 LoRA 库 | 合并后零额外开销，热切换快 |
| 结构控制 | ControlNet（可选） | 按需加载，牺牲 40% 延迟换精度 |
| IP-Adapter | 不推荐 | 需要额外编码图像，增加延迟 |

**关键指标**: 单图延迟 <2s (A100, SD 1.5, 50 步)

### 场景 B: 创作工具（Photoshop 插件、Figma 插件）

**推荐方案**: CFG + ControlNet + IP-Adapter + LoRA 全组合

- 用户可以手绘草图（ControlNet-Scribble）
- 上传参考图（IP-Adapter）
- 选择风格预设（LoRA）
- 输入文字描述（CFG）
- 延迟要求宽松（5-10s 可接受）

### 场景 C: 角色一致性短视频（AI 角色动画）

**推荐方案**: DreamBooth+LoRA (角色) + ControlNet-Pose (动作) + CFG

- DreamBooth+LoRA 锁定角色外观
- ControlNet-Pose 逐帧控制动作
- CFG 控制场景描述
- 需要帧间一致性，考虑 IP-Adapter-FaceID 辅助

### 场景 D: 数据增强（训练数据生成）

**推荐方案**: CFG + ControlNet (结构保持) + 多种 LoRA (多样性)

- 用 ControlNet 保持原始图像的结构信息
- 用不同 LoRA 生成不同风格变体
- CFG scale 适当降低 (w=3-5) 增加多样性

## 条件生成方法演进时间线

```
2020 ─── DDPM (Ho et al.) ─ 无条件扩散模型奠基
  │
2021 ─── Classifier Guidance (Dhariwal & Nichol) ─ 首次超越 GAN
  │       ADM: FID 4.59 on ImageNet 256
  │       里程碑: 证明扩散模型可以生成高质量图像
  │       局限: 需要额外训练噪声分类器
  │
  ├───── LoRA (Hu et al.) ─ 提出低秩适配（原为 NLP）
  │
2022 ─── CFG (Ho & Salimans) ─ 去掉分类器，一招鲜吃遍天
  │       关键创新: 随机丢弃条件 10-20%
  │       影响: 成为所有后续文生图模型的标配
  │
  ├───── LDM / Stable Diffusion (Rombach et al.) ─ 潜在空间 + Cross-Attention
  │       开源引爆社区，CLIP 文本编码器 + CFG
  │
  ├───── Textual Inversion (Gal et al.) ─ 学习新词嵌入
  │       极轻量（~1K 参数），但表达力有限
  │
  ├───── DALL-E 2 (Ramesh et al.) ─ CLIP + Diffusion Prior
  │
2023 ─── DreamBooth (Ruiz et al.) ─ 先验保持损失 + 全量/LoRA 微调
  │       3-5 张图即可学习新概念
  │
  ├───── ControlNet (Zhang et al.) ─ 零卷积 + 编码器副本
  │       空间结构精确控制，社区最大生态
  │
  ├───── IP-Adapter (Ye et al.) ─ 解耦交叉注意力
  │       图像提示，风格/内容参考
  │
  ├───── T2I-Adapter (Mou et al.) ─ 轻量版 ControlNet
  │
  ├───── GLIGEN (Li et al.) ─ 布局控制
  │
  ├───── SDXL ─ 更大模型，LoRA 适配更流行
  │
2024 ─── SD3 / FLUX ─ Flow Matching 时代
  │       CFG scale 大幅降低（4-5 vs 7-8），模型本身条件能力更强
  │
  ├───── ControlNet-XS ─ 轻量变体（减少 85% 参数）
  │
  ├───── IP-Adapter-FaceID ─ 人脸一致性
  │
  ├───── CFG++ ─ 在 x₀ 空间做引导，减少过度饱和
  │
2025 ─── 多条件组合成为标准工作流
          CFG + ControlNet + IP-Adapter + LoRA 自由组合
          方法间边界模糊化，走向统一控制框架
```

## 选型速查卡

### 按控制维度选择

| 控制维度 | 首选方法 | 备选方法 | 说明 |
|---------|---------|---------|------|
| 文本 → 内容 | CFG | — | 所有模型标配 |
| 边缘 → 轮廓 | ControlNet-Canny | T2I-Adapter | ControlNet 更精确 |
| 深度 → 空间 | ControlNet-Depth | ControlNet-Normal | 深度最通用 |
| 骨骼 → 姿态 | ControlNet-Pose | — | OpenPose 预处理 |
| 参考图 → 风格 | IP-Adapter | LoRA (per-style) | IP-Adapter 更通用 |
| 参考图 → 人脸 | IP-Adapter-FaceID | DreamBooth | FaceID 不需要 per-ID 训练 |
| 训练数据 → 概念 | DreamBooth+LoRA | Textual Inversion | DreamBooth+LoRA 性价比最高 |
| 画师风格 → 全局 | LoRA (rank=8-16) | IP-Adapter | LoRA 推理零开销 |

### 按资源预算选择

| 预算 | 推荐方案 | 效果 |
|------|---------|------|
| 零额外成本 | CFG + Textual Inversion | 基础文本控制 + 轻量个性化 |
| 低（1xA100, 1 小时） | CFG + LoRA | 文本 + 风格/角色 |
| 中（1xA100, 1 天） | CFG + ControlNet (社区预训练) + LoRA | 文本 + 结构 + 风格 |
| 高（8xA100, 1 周） | 自训练 ControlNet + IP-Adapter + LoRA | 全方位定制控制 |

## 小结

| 维度 | 最佳方法 | 理由 |
|------|---------|------|
| 文本控制 | **CFG** | 标配，无额外成本 |
| 空间结构控制 | **ControlNet** | 精度最高，生态最丰富 |
| 风格/内容参考 | **IP-Adapter** | 通用，无需 per-style 训练 |
| 个性化（质量优先） | **DreamBooth + LoRA** | 质量好，成本适中 |
| 个性化（成本优先） | **Textual Inversion** | 极轻量，几 KB |
| 多维度联合控制 | **CFG + ControlNet + IP-Adapter + LoRA** | 全兼容，自由组合 |
| 生产部署（低延迟） | **CFG + LoRA**（预合并） | LoRA 合并后零额外开销 |

---

> **下一篇**：[采样加速概述](../04-acceleration-deployment/01-overview) -- 进入模块四。

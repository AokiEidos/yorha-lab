---
title: "条件生成概述"
date: 2026-04-20T17:26:19.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 条件生成概述

> 🔰 入门 | 前置知识：[扩散模型核心直觉](../01-fundamentals/01-core-intuition)

## 为什么需要条件生成

无条件的扩散模型只能从数据分布中随机采样——生成什么完全不可控。**条件生成（Conditional Generation）** 让我们可以通过文本、图像、语义图等条件信号来控制生成内容。

这是 Stable Diffusion 等应用能"按照文字描述画图"的核心能力。没有条件生成，扩散模型只是一个"随机图像生成器"；有了条件生成，它才变成了"可控创作工具"。

## 条件生成的数学本质

从概率论角度，条件生成是从**条件分布** $p(x|c)$ 中采样，其中 $c$ 是条件信号（文本、类别、图像等）。

利用贝叶斯定理：

$$p(x|c) = \frac{p(c|x) \cdot p(x)}{p(c)}$$

对应到得分函数：

$$\nabla_x \log p(x|c) = \underbrace{\nabla_x \log p(x)}_{\text{无条件得分}} + \underbrace{\nabla_x \log p(c|x)}_{\text{条件信号}}$$

这个分解决定了两大范式的分野：
- **引导式**：在采样时加上 $\nabla_x \log p(c|x)$ 项
- **注入式**：直接训练模型学习 $\nabla_x \log p(x|c)$

## 两大范式

### 完整分类体系

```
条件生成方法
├── 引导式（Guidance-based）── 不修改模型，采样时引导
│   ├── Classifier Guidance (2021) ── 外部分类器梯度
│   ├── Classifier-Free Guidance (2022) ── 条件/无条件差值
│   └── CLIP Guidance ── CLIP 梯度引导（社区方法）
│
├── 注入式（Injection-based）── 模型内部添加条件模块
│   ├── 交叉注意力 (Cross-Attention) ── LDM/SD 文本注入
│   ├── ControlNet (2023) ── 旁路编码器 + 零卷积
│   ├── IP-Adapter (2023) ── 解耦交叉注意力
│   ├── T2I-Adapter (2023) ── 轻量特征加法
│   └── GLIGEN (2023) ── Gated Self-Attention 注入布局
│
└── 微调式（Fine-tuning-based）── 修改模型权重
    ├── Textual Inversion (2022) ── 学习新词嵌入
    ├── DreamBooth (2023) ── 全量/LoRA 微调
    ├── LoRA (2021→SD 2023) ── 低秩权重更新
    └── Custom Diffusion (2023) ── 交叉注意力 KV 微调
```

### 引导式（Guidance-based）

**不修改模型结构**，在采样时通过梯度信号引导去噪方向。

```
采样过程: x_T → 去噪 → x_{T-1} → 去噪 → ... → x_0
                 ↑ 引导梯度          ↑ 引导梯度
                 条件信号             条件信号
```

代表方法：
- **Classifier Guidance**（Dhariwal & Nichol, 2021）：用外部分类器在噪声图像上计算梯度，引导生成特定类别。首次让扩散模型在 ImageNet 上超越 GAN（FID: 4.59 vs BigGAN 6.95）。
- **Classifier-Free Guidance (CFG)**（Ho & Salimans, 2022）：训练时随机丢弃条件（10-20%），推理时用条件/无条件预测的差作为引导。当前所有主流文生图模型的标配。

优势：不侵入模型架构，理论优雅，可复用预训练模型
劣势：每步采样需要额外计算（CFG 需要两次前向传播）

### 注入式（Injection-based）

在模型内部添加额外模块来注入条件信息。

```
条件信号 → [条件编码器] → 条件特征
                              ↓ 注入（加法/注意力/拼接）
x_T → [U-Net/DiT + 条件模块] → x_0
```

代表方法及其特点：
- **交叉注意力（Cross-Attention）**：LDM/SD 中文本条件注入。文本经 CLIP 编码后通过 cross-attention 与 U-Net 特征交互，是最基础的注入方式。
- **ControlNet**（Zhang et al., 2023）：复制 U-Net 编码器作为控制分支，通过零卷积连接。支持边缘/深度/姿态等空间条件，社区生态丰富。~361M 额外参数。
- **IP-Adapter**（Ye et al., 2023）：解耦交叉注意力注入图像条件，为图像特征添加独立 KV 投影。~22M 额外参数。
- **T2I-Adapter**（Mou et al., 2023）：轻量级特征加法注入，~77M 参数。比 ControlNet 小 5 倍但控制精度略低。
- **GLIGEN**（Li et al., 2023）：通过 Gated Self-Attention 注入布局信息（边界框 + 文本），支持精确的空间布局控制。

优势：控制精细，可学习复杂条件
劣势：需要额外训练或微调

### 微调式（Fine-tuning-based）

修改模型权重以学习新概念。

代表方法：
- **Textual Inversion**：学习新词嵌入向量，不改模型权重。仅 ~1K 参数，极轻量但表达力有限。
- **DreamBooth**：用 3-5 张图片微调模型学习特定概念。质量最好但成本高（全量微调~860M 参数）。
- **LoRA**：低秩权重更新，~4M 可训练参数。DreamBooth + LoRA 是当前社区最流行的个性化方案。

## 条件注入的技术实现方式

### 加法注入（Additive Injection）

最简单的方式：将条件特征直接加到 U-Net 的中间特征上。

```python
# 加法注入示意
h = unet_block(x_t, t_emb)           # U-Net 中间特征 [B, C, H, W]
c = condition_encoder(condition)       # 条件特征 [B, C, H, W]
h = h + c                             # 直接相加
```

代表方法：T2I-Adapter、ControlNet（零卷积后的加法）

**优点**：实现简单，计算开销小
**缺点**：条件和图像特征必须维度对齐，交互能力有限

### 交叉注意力注入（Cross-Attention Injection）

条件作为 Key/Value，U-Net 特征作为 Query 做注意力：

```python
# 交叉注意力注入（LDM/SD 的核心文本注入方式）
q = W_q(h)                # 图像特征 → Query   [B, HW, dim]
k = W_k(text_emb)         # 文本嵌入 → Key     [B, 77, dim]
v = W_v(text_emb)         # 文本嵌入 → Value   [B, 77, dim]
attn = softmax(q @ k.T / sqrt(d)) @ v   # 注意力输出
h = h + attn               # 残差连接
```

代表方法：LDM/SD 文本条件、IP-Adapter 图像条件

**优点**：条件维度灵活（不需要空间对齐），语义交互能力强
**缺点**：计算开销较大（$O(N \cdot M)$，$N$ 为图像 token 数，$M$ 为条件 token 数）

### 拼接注入（Concatenation Injection）

将条件直接拼接到输入通道：

```python
# 拼接注入示意
x_input = torch.cat([x_t, condition_map], dim=1)  # 通道拼接
# U-Net 输入通道从 4 变为 4+C
eps_pred = unet(x_input, t, text_emb)
```

代表方法：Inpainting（拼接 mask + masked image）、图像编辑

**优点**：信息保留最完整，空间对齐自然
**缺点**：改变输入维度，需要修改模型架构

### 自适应归一化注入（Adaptive Normalization）

通过调制 GroupNorm/LayerNorm 的 scale 和 shift 参数注入条件：

```python
# AdaGN / AdaLN（DiT 使用此方式注入时间步和类别）
gamma, beta = condition_mlp(condition_emb)   # 从条件生成调制参数
h = GroupNorm(h)
h = gamma * h + beta   # 自适应调制
```

代表方法：ADM（时间步+类别）、DiT（AdaLN-Zero）

**优点**：不增加序列长度，开销极小
**缺点**：只能注入全局条件（标量/向量），无法注入空间条件

## 条件信号类型详解

| 条件类型 | 信号形式 | 典型方法 | 具体模型 | 应用场景 |
|---------|---------|---------|---------|---------|
| 文本 | Token 序列 | CFG + Cross-Attn | SD, DALL-E, Imagen, FLUX | 文生图/文生视频 |
| 类别标签 | One-hot / 嵌入 | Classifier Guidance | ADM, DiT | ImageNet 条件生成 |
| 图像（参考） | CLIP 图像特征 | IP-Adapter | IP-Adapter, IP-Adapter-FaceID | 风格迁移、角色一致性 |
| 边缘图 | Canny / HED 边缘 | ControlNet | ControlNet-Canny | 保持物体轮廓 |
| 深度图 | MiDaS 深度估计 | ControlNet | ControlNet-Depth | 3D 感知生成 |
| 人体姿态 | OpenPose 骨骼 | ControlNet | ControlNet-Pose | 姿态控制 |
| 语义分割图 | 像素级标签 | ControlNet / SPADE | ControlNet-Seg, OASIS | 语义控制 |
| 低分辨率图像 | 下采样图 | 条件扩散 | SR3, StableSR | 超分辨率 |
| 遮罩 | 二值遮罩 | Inpainting | SD-Inpaint, PowerPaint | 局部编辑 |
| 布局/边界框 | 坐标 + 标签 | GLIGEN | GLIGEN | 空间布局控制 |
| 动作序列 | 连续向量 | FM/Diffusion Head | $\pi_0$, Diffusion Policy | 机器人动作生成 |

**跨领域应用注意**：条件生成不仅用于图像——在 VLA（视觉-语言-动作模型）领域，扩散/Flow Matching 模型以视觉和语言为条件生成机器人动作序列。参见 [VLA: Diffusion 作为动作生成器](../../../..//posts/vla/07-foundation-models/03-diffusion-action)。

## 训练阶段 vs 推理阶段的条件注入

不同方法在训练和推理阶段的行为差异很大：

| 方法 | 训练阶段 | 推理阶段 | 额外模型 |
|------|---------|---------|---------|
| Classifier Guidance | 训练去噪网络（无条件）+ 单独训练噪声分类器 | 采样时加分类器梯度 | 需要噪声分类器 |
| CFG | 训练去噪网络（随机丢弃条件） | 两次前向传播取差值 | 不需要 |
| ControlNet | 冻结主网络，训练控制分支 | 控制分支特征加到主网络 | 控制分支 |
| IP-Adapter | 冻结主网络，训练 K'/V' 投影 | CLIP 编码 + 解耦注意力 | 图像编码器 + 投影层 |
| LoRA | 冻结主网络，训练低秩矩阵 | 合并到主网络（零额外开销） | 不需要 |
| DreamBooth | 微调整个网络 | 正常推理 | 不需要 |

**关键观察**：LoRA 是唯一在推理时**零额外开销**的方法——训练好的低秩矩阵可以直接合并到原始权重中。

## 条件生成方法的历史演进

```
时间线:
2021 ──── Classifier Guidance (Dhariwal & Nichol)
  │        首次让扩散超越 GAN，但需要额外分类器
  │
2022 ──── CFG (Ho & Salimans)                    Textual Inversion (Gal et al.)
  │        去掉分类器，成为主流                      学习新词嵌入
  │
  │─────── LDM/Stable Diffusion (Rombach et al.)
  │        Cross-Attention 文本注入，潜在空间扩散
  │
2023 ──── ControlNet (Zhang)     IP-Adapter (Ye)     DreamBooth (Ruiz)
  │        空间结构控制           图像风格控制          个性化微调
  │
  │─────── T2I-Adapter (Mou)     GLIGEN (Li)         LoRA for SD
  │        轻量控制              布局控制              低秩适配，社区标配
  │
2024 ──── SD3/FLUX              ControlNet-XS        IP-Adapter-FaceID
  │        Flow Matching 时代    更轻量的控制          人脸一致性
  │
2025 ──── 多条件组合成为标准工作流: CFG + ControlNet + IP-Adapter + LoRA
```

## 本模块路线图

| 文档 | 内容 | 核心问题 | 难度 |
|------|------|---------|------|
| [Classifier Guidance](./02-classifier-guidance) | 分类器梯度引导 | 如何用外部分类器控制生成 | ⚙️ |
| [CFG](./03-cfg) | 无分类器引导 | 如何不需要额外分类器就能引导 | ⚙️ |
| [ControlNet](./04-controlnet) | 结构控制网络 | 如何精确控制空间结构 | ⚙️ |
| [IP-Adapter](./05-ip-adapter) | 图像提示适配 | 如何用参考图像控制风格 | ⚙️ |
| [适配器与微调](./06-adapters) | LoRA、DreamBooth 等 | 如何让模型学习新概念 | ⚙️ |
| [方法对比](./07-comparison) | 综合对比表 | 如何选择合适的方法 | 🔰 |

---

> **下一篇**：[Classifier Guidance](./02-classifier-guidance)

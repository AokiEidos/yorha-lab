---
title: VLA 从入门到精通（二）：多模态基础 — CLIP、ALIGN 与视觉-语言对齐
hidden: True
date: 2026-04-15 17:02:00+08:00
draft: False
tags: ['VLA', '视觉-语言模型', '多模态', '深度学习']
toc: True
---
# VLA 从入门到精通（二）：多模态基础 — CLIP、ALIGN 与视觉-语言对齐

> **前置知识**：本系列假设读者了解深度学习基础（MLP、反向传播、Transformer）。本文解释 VLA 的视觉编码器如何工作，以及"视觉-语言对齐"这个核心概念。术语首次出现时会有 🧠 注。

---

## 0. 核心主旨 (Core Gist/Theme)

VLA 的视觉编码器不是从零训练的——它直接复用 **CLIP** 或 **SigLIP** 这类大规模视觉-语言对比学习模型。这些模型在数十亿张图像-文本对上学会了"这张图在说什么"，这种能力是 VLA 零样本泛化的基础。

本文梳理视觉-语言对齐的数学本质：**对比学习（Contrastive Learning）** 如何让模型同时理解图像和文字，以及 CLIP/SigLIP 的架构差异。最后讨论这种预训练方式对 VLA 的意义与局限。

---

## 1. 什么是"视觉-语言对齐"？

### 1.1 问题定义

**视觉-语言对齐（Vision-Language Alignment）** 指的是让同一个模型能够理解"图像说了什么"和"文字写了什么"，并且知道它们之间的关系。

举例：给定一张柯基犬在草地上奔跑的照片 + 文字"a corgi running on grass"，对齐后的模型应该知道：
- 图像的视觉特征和文字的语义特征**指向同一个概念**
- 这个概念和"a cat sitting on a couch"是不同的

### 1.2 为什么这对 VLA 很重要？

VLA 的输入是"相机图像 + 语言指令"，输出是"动作序列"。视觉编码器的职责是把相机图像转换成模型能理解的**向量表示**——这个表示的质量直接决定了后续的推理质量。

CLIP/SigLIP 提供了两件事：
1. **语义丰富的视觉特征**：从海量互联网数据中学到的物体、场景、动作概念
2. **零样本泛化能力**：见过"柯基"的照片后，能识别没见过的"柴犬"，因为两者都是"狗"这个概念的实例

> **🧠 零样本识别 vs 少样本识别**
>
> - **零样本（Zero-shot）**：完全没训练过的类别，模型靠语言描述识别（如"长腿、短腿、身体低矮"→ 腊肠犬）
> - **少样本（Few-shot）**：每个类别给 1-5 张示例图片，模型从少量样本中快速学习
>
> CLIP 的零样本能力来自它学会了"狗"这个概念的语言描述，所以看到新的狗品种时，能通过语言描述匹配到正确的视觉特征。

---

## 2. 对比学习：CLIP 的数学框架

### 2.1 InfoNCE 损失函数

CLIP 的训练目标是**对比学习**，核心是 **InfoNCE（Noise Contrastive Estimation）** 损失函数。

设一个 batch 有 $N$ 个图像-文本配对 $(I_i, T_i)$。图像编码器输出特征 $f_I(I_i)$，文本编码器输出特征 $f_T(T_i)$。两个特征向量先做 L2 归一化，然后计算余弦相似度矩阵 $S_{ij} = f_I(I_i) \cdot f_T(T_j)$。

**对称交叉熵损失（Symmetric Cross-Entropy Loss）：**

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2} \left( \sum_i \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} + \sum_j \frac{e^{S_{jj}}}{\sum_i e^{S_{ij}}} \right)$$

其中：
- 第一项：图像 $i$ 的正确配对文本 $i$ 在相似度矩阵第 $i$ 行的概率
- 第二项：文本 $j$ 的正确配对图像 $j$ 在相似度矩阵第 $j$ 列的概率

> **🧠 直观理解 InfoNCE**
>
> 把 $N$ 个配对的 $(图像_i, 文本_i)$ 想象成 $N$ 对舞伴。模型的任务是：给每个图像找到它的"正确舞伴"文本。
>
> - $S_{ij}$ 是模型认为"图像 $i$ 和文本 $j$ 是一对"的置信度
> - 损失鼓励：对角线上的配对（正确配对）的相似度要高，对角线外的相似度要低
> - 类似于"有 N-1 个负样本参与对比"，这就是 Noise Contrastive Estimation（NCE）的含义

### 2.2 双塔架构

CLIP 采用**双塔（Dual-Encoder）架构**：

```
图像 → 视觉编码器（ViT） → 归一化特征 f_I
文本 → 文本编码器（Transformer） → 归一化特征 f_T
                      ↓
            余弦相似度矩阵 S
```

- **ViT（Vision Transformer）**：将图像切成 16×16 patches，每个 patch 作为一个 token 输入 Transformer
- **文本 Transformer**：标准自注意力编码器，处理文本 token 序列

### 2.3 温度系数 $\tau$

相似度矩阵前会除以一个**温度系数** $\tau$（temperature）：

$$S_{ij} = \frac{f_I(I_i) \cdot f_T(T_j)}{\tau}$$

$\tau$ 控制相似度分布的"尖锐程度"：
- **$\tau$ 大**（如 1.0）：相似度分布较平缓，模型对错误配对的惩罚较轻
- **$\tau$ 小**（如 0.07）：相似度分布更尖锐，正确配对和错误配对的差距放大，训练更有效但容易不稳定

CLIP 训练时通常用 $\tau \approx 0.07$，这是通过网格搜索找到的最优值。

---

## 3. CLIP vs SigLIP：两种视觉-语言对比学习

### 3.1 CLIP（2021）

**论文**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，Radford et al.，OpenAI，2021

CLIP 是第一个大规模视觉-语言对比学习模型：
- 训练数据：**4亿对** 图像-文本对（从互联网上收集）
- 视觉编码器：**ViT-B/16** 或 ViT-L/16
- 核心贡献：证明了自然语言监督可以学习高质量的视觉表示

**CLIP 的零样本分类能力**：

在 ImageNet（1.2M 图像，1000 类）上，CLIP 不用训练直接做 zero-shot 分类：

$$\text{预测类别} = \arg\max_i \frac{e^{f_I(I) \cdot f_T(\text{类别}_i) / \tau}}{\sum_j e^{f_I(I) \cdot f_T(\text{类别}_j) / \tau}}$$

这就是把每个类别名称（如"柯基"、"柴犬"）直接作为文本输入，然后用图像特征和文本特征对比——不需要任何训练数据。

### 3.2 SigLIP（2024）

**论文**：[SigLIP: Simple Scaled Linguistic Image Pretraining](https://arxiv.org/abs/2309.16634)，Zhai et al.，Google，2024

SigLIP 是 CLIP 的升级版，核心改进是**更简单的目标函数**和**更大的训练规模**。

**SigLIP vs CLIP 的主要区别：**

| | CLIP | SigLIP |
|---|---|---|
| **损失函数** | 对称交叉熵（对称正负样本） | **Sigmoid 损失**（每对单独计算） |
| **负样本处理** | 对称（双向）交叉熵 | **成对 sigmoid**（单向） |
| **训练规模** | 4 亿对 | **9 亿对** |
| **ImageNet Zero-shot** | 76.2% | **82.4%** |

**Sigmoid 损失的定义：**

$$\mathcal{L}_{\text{SigLIP}} = -\sum_{i,j} y_{ij} \log \sigma(S_{ij}) + (1 - y_{ij}) \log (1 - \sigma(S_{ij}))$$

其中 $y_{ij} = 1$ 如果 $(i,j)$ 是正确配对，$y_{ij} = 0$ 否则；$\sigma$ 是 sigmoid 函数。

> **🧠 为什么 SigLIP 用 Sigmoid 而不是 Softmax？**
>
> CLIP 的对称损失需要计算整个 batch 的 softmax归一化，这涉及全局信息——每次更新都需要知道所有样本的相似度，batch 内的所有负样本都参与。
>
> SigLIP 的 sigmoid 损失是**成对独立计算**的：每对 $(i,j)$ 的损失只依赖 $S_{ij}$，不依赖其他样本。这允许 SigLIP 用更大的 batch size（因为不需要全局归一化），从而使用更多负样本训练——这就是它能达到 9 亿对训练数据和更高性能的原因。

---

## 4. 对 VLA 架构的具体影响

### 4.1 视觉编码器的选择

VLA 使用哪种视觉编码器，决定了 VLA 的视觉理解上限。

**OpenVLA（Stanford）**：使用 **SigLIP-L**（ViT-L/16）作为视觉编码器
- SigLIP-L 有 4.06 亿参数，在 90 亿对数据上训练
- 在 94 种机器人技能上预训练后，零样本泛化能力最强

**π₀（Physical Intelligence）**：使用 **SigLIP-SO400M**
- SO400M 是 SigLIP 的超大规模变体（4亿参数），在更大规模数据上训练
- 视觉理解能力最强，但计算成本也最高

**RT-2（Google）**：使用 **PaLM-E** 内置的视觉编码器
- PaLM-E 是多模态 LLM，视觉编码器是单独的 ViT
- 与纯视觉-语言预训练模型相比，视觉理解稍弱，但与语言模型的集成更紧密

### 4.2 视觉编码器与动作输出的解耦

CLIP/SigLIP 是**双塔架构**：视觉编码器和文本编码器是分开的。

在 VLA 中，这种解耦是有意为之的：

```
视觉编码器（SigLIP/SigLIP）→ 视觉特征 f_I
语言编码器（独立或共享）     → 语言特征 f_T
          ↓
     Fusion Transformer（跨模态注意力）
          ↓
     动作输出头
```

跨模态融合发生在 Fusion Transformer 阶段——视觉特征和语言特征通过**交叉注意力（Cross-Attention）**机制交互，模型学习"当指令是'拿起红色积木'时，应该关注图像中的哪个区域"。

---

## 5. 视觉-语言对齐的局限性

### 5.1 空间关系理解弱

CLIP/SigLIP 在"物体识别"上很强，但在**空间关系**上较弱：

- "a person on a horse" vs "a horse on a person" → CLIP 对这两个句子的相似度判断经常出错
- 对于 VLA 来说，这意味着模型可能混淆"把积木放到碗里"和"把碗放到积木上"

> **🧠 为什么 CLIP 空间关系弱？**
>
> CLIP 的训练数据是**图像-标题对**（image-caption pairs），标题通常只描述图像的主要内容，不会精确描述空间关系（如"左边"、"上面"、"前面"）。所以 CLIP 学到的是"主要物体+场景"的语义，而不是"物体之间的精确空间关系"。
>
> 解决方向：使用定位数据（如框选标注、深度图）或更强的视觉编码器（如 SAM）。

### 5.2 动作语义缺失

CLIP/SigLIP 的文本端是**静态描述**（caption），不包含动作意图（如"拿起"、"推动"、"旋转"）。

这对 VLA 来说是一个根本性问题：**预训练的视觉-语言对齐没有学过"动作语义"**。

例如，"push the red block to the left" 和 "push the blue block to the left" 在 CLIP 看来，视觉特征差异可能很小（都是"积木"），但动作意图完全不同（不同颜色的积木）。

**解决思路（当前研究前沿）：**
1. 在 VLA 训练阶段，专门加入**动作描述数据**（action描述的图像对）
2. 用预训练的 LLM（如 GPT-4V）生成更详细的过程描述（chain-of-thought）
3. 引入额外的动作编码器，直接接收机器人关节状态作为输入

### 5.3 连续动作 vs 离散表示的根本矛盾

这是 CLIP 到 VLA 的核心技术跳跃：

- **CLIP 学习的表示**：图像 → 语义特征（"是什么"）
- **VLA 需要的动作**：语义特征 → 具体动作（"怎么做"）

这两者之间有巨大的语义鸿沟。VLA 的 Fusion Transformer + 动作头的任务就是**弥合这个鸿沟**——从"理解场景语义"跨越到"执行精确动作"。

---

## 6. 关键术语解释

| 术语 | 解释 |
|------|------|
| **视觉-语言对齐** | 让视觉编码器和文本编码器输出的特征在同一个语义空间中 |
| **对比学习（Contrastive Learning）** | 通过拉近正样本对、推远负样本对来学习表示 |
| **InfoNCE** | Noise Contrastive Estimation，用于对比学习的损失函数 |
| **双塔架构（Dual-Encoder）** | 视觉和文本分别编码，最后通过对比损失联合训练 |
| **ViT（Vision Transformer）** | 将图像切成 patches 作为 token 的 Transformer |
| **温度系数（Temperature）** | 控制相似度分布尖锐程度的超参数 |
| **Sigmoid 损失** | 成对独立计算的二分类交叉熵，允许大批量负样本 |
| **Zero-shot 分类** | 在完全未训练过的类别上做分类 |
| **交叉注意力（Cross-Attention）** | 让视觉特征和语言特征交互的注意力机制 |
| **语义鸿沟** | 从"理解场景"到"执行动作"之间的巨大语义跨越 |

---

## 7. 下一步预告

有了视觉-语言对齐的基础，下一篇我们将分析 **π₀（Pi-Zero）** 的完整架构，重点是：
- Flow Matching 如何用扩散模型输出连续动作
- 为什么连续动作比 RT-2 的离散 token 更适合精细操作
- π₀ 的跨机器人泛化是如何实现的

---

*参考文献：*
1. Radford et al., **Learning Transferable Visual Models From Natural Language Supervision** (CLIP, 2021)
2. Zhai et al., **SigLIP: Simple Scaled Linguistic Image Pretraining** (2024)
3. Zhai et al., **Sigmoid Loss for Language Image Pre-Training** (SigLIP v2, 2024)
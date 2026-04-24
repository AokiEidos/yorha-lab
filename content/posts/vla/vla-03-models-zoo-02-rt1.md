---
title: "RT-1 详解"
date: 2026-04-20T16:29:34.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# RT-1 详解

> ⚙️ 进阶 | 前置知识：[SayCan](./01-saycan.md)

## 历史地位

**RT-1（Robotics Transformer 1）**（Google, 2022，发表于 *RSS 2023*）是第一个用 Transformer 在大规模真机数据上训练的端到端机器人策略。它的核心论点颠覆了当时社区对模型架构的执念：**数据规模比模型架构更重要**——用足够多的真机演示数据，一个相对简单的 Transformer 就能学到高度泛化的操作策略。

RT-1 拥有 **35M 参数**（对比同期的 Gato 1.2B），但在真机任务上的性能远超更大的模型。这一发现直接推动了后续的大规模数据收集运动（Open X-Embodiment 等）。

## 完整架构图

```
输入层:
  RGB 图像 (t-5, t-4, ..., t) ─→ 6 帧历史图像 (300×300×3 each)
  自然语言指令 ─────────────────→ "pick up the red block"

                    ┌──────────────────────────────────────────────┐
                    │          Image Tokenization Pipeline         │
                    │                                              │
                    │  各帧图像 (300×300×3)                          │
                    │       │                                      │
                    │       ▼                                      │
                    │  [EfficientNet-B3] (ImageNet 预训练)          │
                    │       │                                      │
                    │       ▼                                      │
                    │  特征图 (9×9×512) per frame                   │
                    │       │                                      │
                    │       ▼                                      │
                    │  [FiLM 条件层] ← 语言嵌入 (USE 512-d)         │
                    │  (Feature-wise Linear Modulation)            │
                    │       │                                      │
                    │       ▼                                      │
                    │  条件化特征图 (9×9×512)                        │
                    │       │                                      │
                    │       ▼                                      │
                    │  [TokenLearner] → 8 个视觉 Token per frame   │
                    │       │                                      │
                    │  6 帧 × 8 Token = 48 个视觉 Token             │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │         Transformer Decoder                  │
                    │                                              │
                    │  输入: 48 个视觉 Token                        │
                    │  架构: 8 层 self-attention                    │
                    │        8 头, d_model=512, d_ff=2048           │
                    │        因果 mask (自回归)                      │
                    │                                              │
                    │  输出: 离散动作 Token (自回归生成)              │
                    └──────────────────┬───────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │         Action Tokenization                  │
                    │                                              │
                    │  11 个动作维度, 每维 256 bin:                  │
                    │  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw,          │
                    │   gripper_open, terminate,                   │
                    │   base_Δx, base_Δy, base_Δyaw]               │
                    │                                              │
                    │  自回归: dim1 → dim2 → ... → dim11            │
                    └──────────────────────────────────────────────┘
```

## 关键组件深度解析

### EfficientNet-B3 视觉编码器

RT-1 选择 **EfficientNet-B3**（12M 参数）而非当时更常见的 ResNet-50，原因是：
- 参数效率更高：EfficientNet-B3 在 ImageNet 上 top-1 accuracy 81.6%（vs ResNet-50 的 76.1%），但参数量相当
- 从 ImageNet 预训练初始化，在机器人数据上端到端微调
- 输入分辨率 300x300，输出 9x9 特征图

### FiLM 条件层

**FiLM（Feature-wise Linear Modulation）** 是将语言信息注入视觉特征的轻量机制：

$$\text{FiLM}(F_{ij}) = \gamma(e) \cdot F_{ij} + \beta(e)$$

其中 $F_{ij}$ 是特征图的空间位置 $(i,j)$ 处的向量，$\gamma(e)$ 和 $\beta(e)$ 是从语言嵌入 $e$ 通过线性层生成的缩放和偏移参数。

**为什么用 FiLM 而不是 cross-attention？** FiLM 的计算开销极低（仅两个线性层），而 cross-attention 需要完整的注意力计算。在 RT-1 的架构中，语言条件化发生在 Transformer 之前的每一帧特征提取中，FiLM 的轻量特性在此更合适。

### TokenLearner 机制详解

**TokenLearner**（Ryoo et al., 2021）是 RT-1 中最关键的效率组件。它将 9x9=81 个空间位置的特征图压缩为仅 **8 个** Token：

```
特征图 (9×9×512)
       │
       ▼
 [空间注意力网络] → 8 组空间注意力权重 α_k (9×9), k=1..8
       │
       ▼
 Token_k = Σ_{ij} α_k(i,j) · F(i,j)   (加权空间池化)
       │
       ▼
 8 个 Token (512-d each)
```

```python
# TokenLearner 伪代码
class TokenLearner(nn.Module):
    def __init__(self, num_tokens=8, feature_dim=512):
        super().__init__()
        self.num_tokens = num_tokens
        # 每个 token 有独立的空间注意力网络
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, 1),
                nn.GELU(),
                nn.Conv2d(feature_dim // 4, 1, 1),  # 输出 1-channel 空间权重
                nn.Sigmoid()
            ) for _ in range(num_tokens)
        ])
    
    def forward(self, x):  # x: (B, C, H, W)
        tokens = []
        for attn_layer in self.attention_layers:
            alpha = attn_layer(x)        # (B, 1, H, W) 空间注意力权重
            token = (x * alpha).mean(dim=[2, 3])  # (B, C) 加权池化
            tokens.append(token)
        return torch.stack(tokens, dim=1)  # (B, num_tokens, C)
```

**效率增益**：6 帧 × 81 位置 = 486 个 Token → 6 帧 × 8 Token = 48 个 Token。Transformer 的自注意力计算量从 $O(486^2) \approx 236K$ 降为 $O(48^2) \approx 2.3K$，减少了 **~100 倍**。

### 语言编码器

RT-1 使用 Google 的 **Universal Sentence Encoder (USE)**，将自然语言指令编码为 512 维向量。USE 的选择基于效率考虑——它是一个紧凑的句子编码器，推理速度远快于 BERT/T5 等模型。

## 动作空间与 Token 化

| 动作维度 | 范围 | 含义 | Bin 数 |
|---------|------|------|-------|
| $\Delta x$ | [-0.05, 0.05] m | 末端位移 x | 256 |
| $\Delta y$ | [-0.05, 0.05] m | 末端位移 y | 256 |
| $\Delta z$ | [-0.05, 0.05] m | 末端位移 z | 256 |
| $\Delta \text{roll}$ | [-0.25, 0.25] rad | 姿态 roll | 256 |
| $\Delta \text{pitch}$ | [-0.25, 0.25] rad | 姿态 pitch | 256 |
| $\Delta \text{yaw}$ | [-0.25, 0.25] rad | 姿态 yaw | 256 |
| gripper | {open, close} | 夹爪状态 | 2 |
| terminate | {continue, terminate} | 终止标志 | 2 |
| base $\Delta x$ | [-0.05, 0.05] m | 底盘位移 x | 256 |
| base $\Delta y$ | [-0.05, 0.05] m | 底盘位移 y | 256 |
| base $\Delta \text{yaw}$ | [-0.25, 0.25] rad | 底盘旋转 | 256 |

控制频率：**3 Hz**（每秒发送 3 条动作指令）。

## 训练超参数

| 超参数 | 值 | 说明 |
|-------|-----|------|
| 优化器 | Adafactor | Google 常用的内存高效优化器 |
| 学习率 | $10^{-4}$ | 恒定学习率 |
| Batch size | 4096 | 大 batch 提高数据利用率 |
| 训练步数 | 200K | ~2 天训练（TPU v3 Pod） |
| 损失函数 | 交叉熵 | 每个动作维度独立的分类损失 |
| 数据增强 | 随机裁剪、颜色抖动 | 提升视觉泛化 |
| 帧采样 | 最近 6 帧 @3Hz | 历史窗口 ~2 秒 |
| 权重初始化 | EfficientNet: ImageNet; 其他: 随机 | 视觉编码器预训练 |
| 总参数量 | **35M** | 远小于同期的 Gato (1.2B) |
| 训练硬件 | TPU v3 Pod (128 cores) | ~2 天完成训练 |

训练损失函数为标准交叉熵，对 11 个动作维度求和：

$$\mathcal{L} = \sum_{d=1}^{11} \text{CE}(\hat{a}_d, a_d^*) = -\sum_{d=1}^{11} \log p_\theta(\text{bin}(a_d^*) | o, l, a_{<d})$$

## 数据规模与分布分析

**130K 条真机演示轨迹**——在当时是前所未有的规模：

| 数据维度 | 规格 |
|---------|------|
| 总轨迹数 | 130,000+ |
| 机器人数量 | 13 台 Everyday Robots |
| 收集周期 | 17 个月 |
| 任务种类 | 700+ |
| 数据收集方式 | 远程遥操作（人类操作员） |
| 平均轨迹长度 | ~30 步 (@3Hz, ~10s) |
| 总步数 | ~4M |

### 700+ 任务的分布

```
任务类别                    占比      示例
─────────────────────────────────────────────────────
拾取类 (pick)              ~35%     "pick up the red block"
放置类 (place)             ~20%     "place the can on the shelf"
导航+操作                   ~15%     "go to the counter and grab sponge"
开/关类                     ~10%     "open the drawer"
整理/清洁                   ~8%      "wipe the table"
移动物体                    ~7%      "push the block to the left"
其他                        ~5%      "knock over the bottle"
```

### 数据多样性的关键维度

- **物体多样性**：训练包含 ~200 种不同物体（食品、工具、容器、玩具等）
- **场景多样性**：厨房台面、桌面、架子等多种场景
- **光照多样性**：不同时间段收集的数据
- **指令多样性**：同一任务的多种自然语言表述

## 核心实验结果

### 主要评估（Google Everyday Robot, 真机）

| 方法 | 参数量 | 已见任务成功率 | 未见任务成功率 |
|------|-------|-------------|-------------|
| BC-Z (Lynch et al.) | ~5M | 57% | 24% |
| Gato (Reed et al.) | 1.2B | 64% | 29% |
| **RT-1** | **35M** | **97%** | **76%** |
| RT-1 (无 TokenLearner) | 35M | 90% | 65% |
| RT-1 (单任务训练) | 35M | 85% | N/A |

### 泛化性实验

| 泛化场景 | RT-1 成功率 | Gato 成功率 | BC-Z 成功率 |
|---------|-----------|-----------|-----------|
| 新物体（训练中未见） | 53% | 26% | 19% |
| 新背景/场景 | 70% | 38% | 28% |
| 干扰物存在 | 83% | 57% | 41% |
| 不同光照 | 91% | 62% | 50% |

### 数据规模消融实验

这是 RT-1 最重要的发现之一：

```
训练数据量 vs 成功率:

100%  │                                          ★ RT-1 (130K)
      │                                    ╱
 80%  │                              ╱────╱
      │                        ╱────╱
 60%  │                  ╱────╱
      │            ╱────╱
 40%  │      ╱────╱
      │╱────╱
 20%  │
      │
  0%  └──────────────────────────────────────────
      0    10K   20K   40K   60K   80K  100K  130K
                        训练轨迹数
```

性能随数据量近似**对数增长**——从 10K 到 130K 轨迹，成功率从 ~40% 提升到 ~97%。

## 与 BC-Z 和 Gato 的详细对比

| 维度 | RT-1 | BC-Z | Gato |
|------|------|------|------|
| **发布** | Google 2022 | Google 2022 | DeepMind 2022 |
| **参数量** | 35M | ~5M | 1.2B |
| **视觉编码器** | EfficientNet-B3 | ResNet-18 | ViT 自训练 |
| **语言编码器** | USE (固定) | 任务 ID | 自训练 |
| **训练数据** | 130K 真机 | 100K 真机 | 多任务混合 |
| **多任务** | 700+ 机器人任务 | ~100 机器人任务 | 游戏+对话+机器人 |
| **动作表示** | 256 bin 离散 | 256 bin 离散 | 1024 bin 离散 |
| **帧数** | 6 帧历史 | 2 帧 | 不固定 |
| **TokenLearner** | 有 (8 token) | 无 | 无 |
| **控制频率** | 3 Hz | 5 Hz | ~3 Hz |

**关键差异分析**：

- **RT-1 vs Gato**：Gato 试图做"通才"模型（游戏、对话、机器人共同训练），参数量 34 倍于 RT-1，但在机器人任务上反而大幅落后。原因是 Gato 的训练数据中机器人数据占比极低（<5%），且不同任务之间的负迁移影响了性能。RT-1 证明了专注的大规模机器人数据比泛化的多模态训练更有效。

- **RT-1 vs BC-Z**：BC-Z 的架构与 RT-1 类似但更简单（无 TokenLearner、更小的视觉编码器），数据规模也稍小。RT-1 的优势主要来自数据规模和 TokenLearner 带来的效率提升。

## RT-1 的局限性

1. **没有利用预训练 VLM**：视觉编码器仅从 ImageNet 初始化，语言编码器是固定的 USE。没有利用互联网规模的视觉-语言知识，限制了语义泛化能力
2. **单步预测**：每次只预测 1 步动作，无 Action Chunking。这限制了控制频率（3 Hz）和时间一致性
3. **仅限 Google 机器人**：所有数据来自 Everyday Robots，无跨机器人泛化验证
4. **数据收集成本**：13 台机器人 × 17 个月的遥操作数据收集，成本极高
5. **闭源**：模型权重和训练数据未公开

## RT-1 的历史意义

RT-1 为后续 VLA 奠定了三个关键基础：

1. **证明 Transformer + 大规模数据的路线可行**：简单架构 + 海量数据 > 复杂架构 + 少量数据
2. **证明动作 Token 化 + 自回归生成是有效的动作表示方式**：为 RT-2 的"动作即语言"范式铺路
3. **建立了数据规模的重要性认知**：推动社区投入大规模数据收集（最终催生 Open X-Embodiment）

但 RT-1 没有使用预训练的 VLM——它的视觉和语言编码器都相对小型。RT-2 将在此基础上引入大规模 VLM，开创真正的 VLA 范式。

## 小结

| 概念 | 要点 |
|------|------|
| 核心论点 | 数据规模 >> 模型架构，35M 参数胜过 1.2B |
| TokenLearner | 将 81 个空间特征压缩为 8 Token，自注意力计算减少 ~100x |
| 数据规模 | 130K 真机轨迹，13 台机器人，17 个月，700+ 任务 |
| 关键结果 | 已见任务 97%，未见任务 76%，远超 Gato 和 BC-Z |
| 训练 | Adafactor, lr=1e-4, batch=4096, 200K steps, TPU v3 Pod |
| 历史意义 | 验证 Transformer + 大规模真机数据路线，奠定 VLA 基础 |

---

> **下一篇**：[RT-2 详解](./03-rt2.md)

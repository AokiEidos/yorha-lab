---
title: "DiT 架构深度解析"
date: 2026-04-20T17:27:43.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# DiT 架构深度解析

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[文生图技术栈](./01-text-to-image)，了解 Vision Transformer (ViT) 基础，[Score Matching](../01-fundamentals/06-score-matching)

## 从 U-Net 到 Transformer

**DiT（Diffusion Transformer）**（Peebles & Xie, 2023）用标准 Transformer 替代 U-Net 作为扩散模型的去噪骨干网络。这不仅仅是"换了个架构"——它开启了扩散模型的 **Scaling Laws（缩放定律）** 时代，使得扩散模型可以像 LLM 一样通过简单地增大模型规模来持续提升性能。

### 为什么要替代 U-Net？

U-Net 在扩散模型中服役了三年多（2020-2023），是一个非常成功的架构，但面临几个根本性瓶颈：

| 维度 | U-Net 的问题 | Transformer 的优势 |
|------|-------------|-------------------|
| 扩展性 | 多尺度编码器-解码器结构复杂，参数扩展需要手动设计每层宽度 | 简单堆叠层数即可扩展 |
| 归纳偏置 | 强归纳偏置（局部性、多尺度）在小数据时有利，但限制了大数据时的表现上限 | 弱归纳偏置，完全依赖数据学习，上限更高 |
| 硬件友好 | 非均匀计算（不同分辨率层计算量不同），GPU 利用率不均 | 均匀计算，每层结构相同，GPU 利用率高 |
| 生态复用 | 独立于 NLP，无法复用 LLM 的优化技术 | 与 LLM 架构统一，可复用 FlashAttention、模型并行等技术 |

---

## DiT 完整架构 🔬

### 1. Patch 化（Patchification）

将潜空间特征图切成固定大小的 patch，映射为 token 序列——这与 ViT 处理图像的方式完全一致。

**数学描述**：

给定潜变量 $z \in \mathbb{R}^{C \times H \times W}$（例如 $4 \times 64 \times 64$），patch 大小为 $p$：

$$N = \frac{H}{p} \times \frac{W}{p} = \frac{64}{2} \times \frac{64}{2} = 1024 \text{ (patch 数量)}$$

每个 patch 的原始维度为 $C \times p \times p = 4 \times 2 \times 2 = 16$，通过线性投影映射到模型维度 $D$。

```
潜变量 z: [B, 4, 64, 64]
    │
    ▼ 重排为 patch: 每个 patch 是 4×2×2 = 16 维
    │
Patch 序列: [B, 1024, 16]
    │
    ▼ 线性投影: 16 → D (如 D=1152 for DiT-XL)
    │
Token 序列: [B, 1024, 1152]
    │
    ▼ 加入位置编码 (2D sinusoidal positional encoding)
    │
最终输入: [B, 1024, 1152]
```

**位置编码**：DiT 使用 2D 正弦位置编码——对每个 patch 的 $(row, col)$ 坐标分别编码（各占 $D/2$ 维），拼接后得到完整的位置嵌入。这使模型感知空间结构，也支持通过插值位置编码来处理可变分辨率。

### 2. DiT Block 与 adaLN-Zero 🔬

每个 DiT Block 包含两个子层——**自注意力（Self-Attention）** 和 **前馈网络（MLP）**——均通过 adaLN-Zero 注入条件信息。数据流如下：

```
输入 x → adaLN(γ₁,β₁) → Self-Attn → ×α₁(gate) → +残差 → adaLN(γ₂,β₂) → MLP → ×α₂(gate) → +残差 → 输出
              ↑                                                ↑
              └──── Conditioning MLP: t_emb+c → 6 参数 ────────┘
                    (γ₁, β₁, α₁, γ₂, β₂, α₂)
```

**自适应层归一化（Adaptive Layer Norm, adaLN）** 是 DiT 的核心条件注入机制。与 U-Net 使用 Cross-Attention 注入文本条件不同，DiT 通过调制 LayerNorm 的参数来注入条件。

adaLN-Zero 的 "Zero" 指的是 gate 参数 $\alpha$ 初始化为零，确保训练初期每个 DiT Block 是**恒等映射（Identity Function）**——输出等于输入。这极大地稳定了深层 Transformer 的训练。

```python
class DiTBlock(nn.Module):
    """完整的 DiT Block 实现，包含 adaLN-Zero 条件注入"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # 两个 LayerNorm，关闭可学习的 affine 参数（由 adaLN 提供）
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        
        # adaLN-Zero 调制网络：输出 6 个调制参数
        # gamma1, beta1, alpha1 (用于 Attention 分支)
        # gamma2, beta2, alpha2 (用于 MLP 分支)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # 关键：零初始化最后一层，使 alpha 初始化为 0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, c):
        """
        x: [B, N, D] — patch token 序列
        c: [B, D]    — 条件向量（时间步嵌入 + 类别/文本嵌入）
        """
        # 生成 6 个调制参数
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = (
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        )
        # 参数含义：
        #   gamma: scale（缩放 LayerNorm 输出）
        #   beta:  shift（平移 LayerNorm 输出）
        #   alpha: gate （门控残差连接，初始化为 0）
        
        # ── Attention 分支 ──
        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1            # adaLN 调制
        h, _ = self.attn(h, h, h)               # Self-Attention
        x = x + alpha1 * h                      # 门控残差（alpha1 初始为 0）
        
        # ── MLP 分支 ──
        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2            # adaLN 调制
        h = self.mlp(h)                         # FFN
        x = x + alpha2 * h                      # 门控残差（alpha2 初始为 0）
        
        return x
```

**为什么零初始化如此重要？** 在训练开始时，$\alpha_1 = \alpha_2 = 0$，因此每个 DiT Block 的输出就是其输入（恒等映射）。这意味着无论网络有多深，梯度都能顺畅传播——等价于"所有层不存在"的情况。随着训练推进，模型逐渐学会利用每一层。这个技巧来源于 ResNet 的零初始化残差分析，DiT 将其推广到了条件生成场景。

---

## DiT 缩放实验结果

DiT 论文在 ImageNet 256×256 类条件生成上的实验结果证明了 Scaling Laws 在扩散模型中同样成立：

| 模型 | 层数 | 隐藏维度 D | 注意力头 | 参数量 | GFLOPs | FID↓ (250 步 DDPM) | FID↓ (cfg=1.5) |
|------|------|-----------|---------|--------|--------|--------------------|--------------------|
| DiT-S/2 | 12 | 384 | 6 | 33M | 6.1 | 68.4 | 9.62 |
| DiT-B/2 | 12 | 768 | 12 | 130M | 23.0 | 43.5 | 5.57 |
| DiT-L/2 | 24 | 1024 | 16 | 458M | 80.7 | 9.62 | 3.22 |
| DiT-XL/2 | 28 | 1152 | 16 | 675M | 118.6 | **2.27** | **2.27** |

**关键观察**：
- 从 33M 到 675M（~20x 参数），FID 从 68.4 降到 2.27（~30x 提升）
- DiT-XL/2 的 FID=2.27 在当时（2023 年初）刷新了 ImageNet 256 的 SOTA
- patch 大小对性能影响显著：/2（patch=2）远优于 /4（patch=4）和 /8（patch=8），因为更小的 patch 保留更多空间细节，但序列更长

### GFLOPs vs FID 的 Scaling 曲线

```
FID (ImageNet 256×256, cfg=1.5)
 │
 70├── S/8 ●
   │
 50├────────── S/4 ●
   │
 30├──────── B/8 ●
   │         B/4 ●
 10├─ S/2 ●──── B/2 ●
   │                    L/4 ●
  5├──────────────── L/2 ●
   │
  2├──────────────────────── XL/2 ●
   │
   └──────┬──────┬──────┬──────┬──────── GFLOPs
          1     10     50    100    150
```

无论 patch 大小如何，更大的模型总是更好——这就是 Scaling Laws 的力量。

---

## MM-DiT：Stable Diffusion 3 的联合注意力 🔬

SD3 提出的 **MM-DiT（Multimodal Diffusion Transformer）** 在 DiT 基础上做了关键创新——**联合注意力（Joint Attention）**，让图像和文本在注意力层中双向交互。

### MM-DiT 架构图

```
                    MM-DiT Block
  ┌────────────────────────────────────────────┐
  │                                            │
  │  图像 tokens        文本 tokens             │
  │  x_img [B,N,D]     x_txt [B,M,D]          │
  │    │                  │                     │
  │    ▼                  ▼                     │
  │  adaLN (γ₁,β₁)     adaLN (γ₁',β₁')       │
  │    │                  │                     │
  │    ▼                  ▼                     │
  │  ┌─────┐           ┌─────┐                 │
  │  │Q_img│           │Q_txt│  ← 独立 QKV 投影│
  │  │K_img│           │K_txt│                 │
  │  │V_img│           │V_txt│                 │
  │  └──┬──┘           └──┬──┘                 │
  │     │                 │                     │
  │     ▼ concat          ▼ concat              │
  │  K = [K_img ; K_txt]                       │
  │  V = [V_img ; V_txt]                       │
  │     │                                       │
  │     ▼ Joint Attention                       │
  │  Attn([Q_img;Q_txt], [K_img;K_txt],        │
  │       [V_img;V_txt])                        │
  │     │                                       │
  │     ▼ split                                 │
  │  out_img    out_txt                         │
  │    │          │                              │
  │    ▼×α₁      ▼×α₁'                         │
  │  x_img +r   x_txt +r                       │
  │    │          │                              │
  │   ...MLP... ...MLP... (各自独立)             │
  │    │          │                              │
  │  输出 x_img'  输出 x_txt'                    │
  └────────────────────────────────────────────┘
```

### 联合注意力的数学表达

给定图像 tokens $x_{\text{img}} \in \mathbb{R}^{N \times D}$ 和文本 tokens $x_{\text{txt}} \in \mathbb{R}^{M \times D}$：

$$Q_{\text{img}} = x_{\text{img}} W_Q^{\text{img}}, \quad K_{\text{img}} = x_{\text{img}} W_K^{\text{img}}, \quad V_{\text{img}} = x_{\text{img}} W_V^{\text{img}}$$

$$Q_{\text{txt}} = x_{\text{txt}} W_Q^{\text{txt}}, \quad K_{\text{txt}} = x_{\text{txt}} W_K^{\text{txt}}, \quad V_{\text{txt}} = x_{\text{txt}} W_V^{\text{txt}}$$

拼接后做联合注意力：

$$Q = [Q_{\text{img}}; Q_{\text{txt}}], \quad K = [K_{\text{img}}; K_{\text{txt}}], \quad V = [V_{\text{img}}; V_{\text{txt}}]$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

然后将输出按原始长度拆分回图像和文本部分。

**为什么用 Joint Attention 而不是 Cross-Attention？**

| 机制 | SD 1.x (Cross-Attn) | SD3 (Joint-Attn) |
|------|---------------------|-------------------|
| 信息流 | 文本→图像（单向） | 文本↔图像（双向） |
| QKV 投影 | 图像提供 Q，文本提供 KV | 各自独立投影，拼接后联合计算 |
| 文本表示 | 在所有层中保持不变 | 随层数逐渐被图像信息丰富 |
| 效果 | 文本指导图像生成 | 图像和文本共同演化，对齐更好 |

### SD3 的其他关键设计

**Rectified Flow（整流流）训练**：SD3 不使用 DDPM 的噪声预测，而是使用 Flow Matching 框架（参见 [Flow Matching](../02-models-zoo/06-flow-matching)）。具体地，SD3 使用 Rectified Flow——学习从噪声到数据的直线路径：

$$v_\theta(x_t, t) = x_1 - x_0 \quad \text{（目标是直线方向向量）}$$

其中 $x_t = (1-t)x_0 + t\epsilon$，$t \in [0, 1]$。

**Logit-Normal 时间步采样**：训练时对时间步 $t$ 的采样分布进行调整——使用 logit-normal 分布而非均匀分布：

$$t \sim \sigma(\mathcal{N}(0, 1))$$

这使得模型在中间时间步（$t \approx 0.5$）获得更多训练信号，因为这些时间步是去噪过程中最关键的（既不是纯噪声也不是纯信号）。

---

## DiT vs U-Net 全面对比

| 维度 | U-Net (SD 1.5/SDXL) | DiT (原版) | MM-DiT (SD3) |
|------|---------------------|-----------|-------------|
| 参数量 | 860M / 2.6B | 33M~675M | 2B / 8B |
| 条件注入 | Cross-Attention + timestep | adaLN-Zero | Joint Attention + adaLN-Zero |
| 文本表示 | 层间不变 | 不直接处理文本 | 层间与图像共演化 |
| 归纳偏置 | 强（多尺度+局部性） | 弱（全局注意力） | 弱+双流设计 |
| Scaling | ~3B 后困难 | 遵循 Scaling Laws | 遵循 Scaling Laws |
| 内存峰值 | Skip connection 占用 | 序列长时注意力矩阵大 | 联合序列更长 |
| 训练稳定性 | 成熟，技巧少 | 需要零初始化等技巧 | 继承 DiT 的训练技巧 |
| 推理速度 | 较快（架构高效） | 中等 | 较慢（联合序列长） |
| 可变分辨率 | 困难 | 天然支持（patch 化） | 天然支持 |
| 与 LLM 统一 | 困难 | 天然统一（标准 Transformer） | 天然统一 |

---

## 小结

| 概念 | 要点 |
|------|------|
| Patchification | 潜变量切成 $p \times p$ patch → 线性投影 + 2D 位置编码，序列长度 $(H/p) \times (W/p)$ |
| adaLN-Zero | 6 个调制参数（$\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$），$\alpha$ 零初始化确保训练稳定 |
| Scaling Laws | 33M→675M 参数，FID 68.4→2.27，规模越大效果越好 |
| MM-DiT | 图像+文本独立 QKV 投影→拼接→联合注意力→拆分，实现双向交互 |
| SD3 设计 | Rectified Flow + Logit-Normal 时间步 + 三编码器 + MM-DiT |
| DiT vs U-Net | DiT 赢在扩展性和 LLM 生态统一，U-Net 赢在效率和成熟生态 |

---

> **下一篇**：[文生视频技术](./03-text-to-video) — 从单帧到时间序列，扩散模型如何处理时间维度。

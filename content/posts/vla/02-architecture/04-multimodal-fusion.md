---
title: "多模态输入融合"
date: 2026-04-20T16:34:15.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 多模态输入融合

> ⚙️ 进阶 | 前置知识：[动作解码头设计](./03-action-head)，[具身感知基础](../01-foundations/02-embodied-perception)

## 融合什么：VLA 的输入模态

VLA 模型需要融合多种异构输入——每种模态的数据格式、维度、语义层级都不同。如何有效融合这些信息，直接决定了 VLA 的感知能力上限。

| 模态 | 数据形式 | 维度示例 | 信息内容 | 时变性 |
|------|---------|---------|---------|--------|
| **RGB 图像** | 3D 张量 (1-3 视角) | [3, 224, 224] × N_cam | 场景外观、物体识别 | 帧级变化 |
| **语言指令** | Token 序列 | [1, L] (L~77) | 任务目标、约束条件 | 任务级不变 |
| **本体感觉** | 1D 向量 | [1, 7-20] | 关节角度/速度/力矩 | 步级变化 |
| **深度图** | 2D 张量 (可选) | [1, 224, 224] | 3D 几何 | 帧级变化 |
| **力/力矩** | 1D 向量 (可选) | [1, 6] | 接触力信息 | 步级变化 |
| **历史帧** | 时序图像 (可选) | [T, 3, 224, 224] | 运动趋势 | 窗口级 |

核心挑战：**如何让 Transformer 同时"看到"图像中的杯子、"理解"指令中的"放到左边"、"知道"手臂当前在哪？**

## 两大融合范式

### 早期融合（Early Fusion）

**核心思想**：将所有模态**先 Token 化，再拼接成一个长序列**，统一送入单一 Transformer。

```
┌──────────────────────────────────────────────────────────────┐
│                     Early Fusion                              │
│                                                              │
│  图像 (cam1) → [ViT] → 256 个视觉 Token  ─┐                  │
│  图像 (cam2) → [ViT] → 256 个视觉 Token  ─┤                  │
│  语言指令 → [Tokenizer] → ~20 个文本 Token ─┼→ 拼接 → [LLM]  │
│  本体感觉 → [MLP] → 1-2 个状态 Token      ─┤                  │
│  历史帧 → [ViT] → K×256 个历史 Token      ─┘                  │
│                                                              │
│  总序列长度: 256×N_cam + 20 + 2 + 256×K                       │
│  例: 2 相机 + 2 帧历史 = 256×2 + 20 + 2 + 256×2 = 1046 Token  │
│                                                              │
│  所有 Token 在同一 Transformer 中通过 Self-Attention 交互       │
└──────────────────────────────────────────────────────────────┘
```

**优势：**
- 架构最简洁——直接复用 LLM/VLM 的标准 Transformer
- 模态间可以通过 Self-Attention **自由交互**——模型可以学到"看到红色杯子"和"指令说拿红色的"之间的对应关系
- 不需要设计特殊的融合模块

**劣势：**
- 序列长度大——视觉 Token 通常占 80%+，自注意力计算量 $O(N^2)$ 随 Token 数平方增长
- 不同模态的"重要性"被平等对待——视觉可能淹没本体感觉的微小信号
- 训练时不同模态的学习速度可能不同，导致优化困难

**代表模型：** RT-2、OpenVLA、π₀

### 晚期融合（Late Fusion）

**核心思想**：不同模态**分别用独立编码器处理**，在解码阶段通过特定机制融合。

```
┌──────────────────────────────────────────────────────────────┐
│                      Late Fusion                              │
│                                                              │
│  图像 → [视觉编码器] → 视觉特征 f_v  ─┐                       │
│                                        │                     │
│  语言 → [语言编码器] → 语言特征 f_l  ─┤→ [融合模块] → 融合特征 │
│                                        │                     │
│  本体 → [状态编码器] → 状态特征 f_s  ─┘                       │
│                                                              │
│  融合方式:                                                    │
│    ① FiLM: f_l 调制 f_v 的 scale/shift                       │
│    ② Cross-Attention: f_v 作为 Q, f_l 作为 K/V              │
│    ③ 特征拼接: concat(f_v, f_l, f_s) → MLP                  │
│    ④ 分层注意力: 低层融合 v+s, 高层融合 l                      │
└──────────────────────────────────────────────────────────────┘
```

**优势：**
- 各模态可用**最适合的编码器**（如视觉用 ViT，语言用 T5）
- 计算量可控——编码器独立运行，融合模块通常较小
- 可以在融合前对各模态做预处理（如视觉 Token 压缩）

**劣势：**
- 需要手工设计融合模块——设计空间大，最优选择不明确
- 模态间的交互受限于融合模块的容量
- 架构复杂度更高

**代表模型：** Octo（分层 Transformer）、部分 Diffusion Policy 变体

## 具体融合机制详解

### FiLM（Feature-wise Linear Modulation）

**FiLM（特征级线性调制）** 是最轻量的条件注入方式——用条件特征（语言/状态）生成 scale $\gamma$ 和 shift $\beta$，调制目标特征（视觉）：

$$\text{FiLM}(F, c) = \gamma(c) \odot F + \beta(c)$$

```python
class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net = nn.Linear(cond_dim, feature_dim)
    
    def forward(self, feature, condition):
        gamma = self.gamma_net(condition)  # [B, D]
        beta = self.beta_net(condition)    # [B, D]
        return gamma * feature + beta      # 逐元素调制
```

**使用场景：** RT-1（语言→视觉）、Octo（本体感觉→视觉）

**优势：** 极轻量（仅两个线性层），不增加序列长度
**劣势：** 表达力有限——只能做全局的缩放和偏移，无法做空间选择性的注入

### Cross-Attention（交叉注意力）

**交叉注意力** 让一种模态（Query）"查询"另一种模态（Key/Value）的信息：

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
    
    def forward(self, x, context):
        """
        x: 目标模态 Token [B, N_x, D] (如视觉 Token)
        context: 条件模态 Token [B, N_c, D] (如语言 Token)
        """
        Q = self.to_q(x)         # 视觉生成 Query: "我想知道什么？"
        K = self.to_k(context)   # 语言生成 Key: "我有什么信息？"
        V = self.to_v(context)   # 语言生成 Value: "信息的内容"
        
        # 多头注意力
        attn = torch.softmax(Q @ K.T / math.sqrt(d_k), dim=-1)
        out = attn @ V  # 视觉 Token 获得了语言引导的上下文
        return out
```

**使用场景：** Stable Diffusion 中文本→图像的注入方式。在 VLA 中，Flamingo/RoboFlamingo 用此方式注入视觉历史。

### Perceiver Resampler（感知重采样器）

**Perceiver Resampler** 用一组**可学习的查询 Token**（Latent Queries）通过交叉注意力从输入 Token 中提取信息，实现 Token 数量压缩：

```
输入: 256 个视觉 Token → [Perceiver Resampler (K 个 Latent Query)] → K 个压缩 Token
                                                                     (K << 256, 如 K=32)
```

**使用场景：** Flamingo、BLIP-2、部分 VLA 的视觉 Token 压缩

## 本体感觉注入的四种方式

本体感觉（关节角度/末端位姿/夹爪状态）是低维连续向量，与高维视觉 Token 性质不同。注入方式选择影响信息利用效率：

| 方式 | 描述 | Token 数增加 | 代表模型 | 适合场景 |
|------|------|------------|---------|---------|
| **MLP Token 化** | 向量→MLP→1-2 个 Token，拼入序列 | +1-2 | OpenVLA, π₀ | 序列长度敏感时 |
| **FiLM 调制** | 用本体特征调制视觉特征的 γ/β | 0 | Octo | 不增加序列长度 |
| **拼接到动作头输入** | 编码后与 Readout/动作特征拼接 | 0 | Diffusion Policy | 只影响动作生成 |
| **通道拼接** | 将本体信息复制到图像通道维度 | 0 | 部分 CNN 方法 | 简单但粗糙 |

```python
# MLP Token 化示例
class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim=15, token_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            nn.GELU(),
            nn.Linear(256, token_dim),
        )
    
    def forward(self, proprio):
        # proprio: [B, 15] (7 关节角 + 7 关节速度 + 1 夹爪)
        return self.mlp(proprio).unsqueeze(1)  # [B, 1, 4096] 一个 Token
```

## 历史信息处理

VLA 通常不只看当前帧——历史观测提供运动趋势、物体被遮挡前的记忆、以及动作连贯性的上下文。

### 策略对比

| 策略 | 描述 | 额外 Token 数 | 代表模型 | 适合 |
|------|------|-------------|---------|------|
| **无历史** | 只看当前帧 | 0 | RT-2, OpenVLA | 反应式任务 |
| **帧堆叠** | 最近 K 帧的视觉 Token 全部拼接 | 256×K | RT-1 (K=6) | 简单但内存大 |
| **Perceiver 压缩** | K 帧编码后用 Perceiver 压缩 | 固定 M 个 | RoboFlamingo | 长历史 |
| **递归状态** | RNN/LSTM 维护隐状态 | 0 | BC-RNN | 隐式记忆 |
| **因果 Transformer** | 历史帧作为因果序列 | 256×K | Octo | 自然时序建模 |

### 历史帧数选择的经验法则

| 任务类型 | 推荐帧数 | 理由 |
|---------|---------|------|
| 静态抓取 | 0-1 | 当前帧足够判断 |
| 动态跟踪 | 2-4 | 需要速度估计 |
| 状态变化检测 | 2-6 | 如"门是否已打开" |
| 长序列操作 | 4-16 | 如折衣需要记住之前的折痕 |

## 主流模型的融合策略一览

| 模型 | 融合范式 | 视觉编码 | 语言注入 | 本体注入 | 历史 | 总 Token 数 |
|------|---------|---------|---------|---------|------|-----------|
| **RT-1** | 早期(FiLM+拼接) | EfficientNet+TokenLearner | FiLM | 不使用 | 6 帧 | 48 |
| **RT-2** | 早期拼接 | ViT (PaLI-X) | 文本 Token | 不使用 | 无 | ~300 |
| **Octo** | 分层 Transformer | ViT-B/16 | Task Token | FiLM | 2 帧 | ~550 |
| **OpenVLA** | 早期拼接 | SigLIP+DINOv2 | 文本 Token | 不使用 | 无 | ~530 |
| **π₀** | 早期拼接 | SigLIP (PaliGemma) | 文本 Token | MLP Token | 1 帧 | ~300 |
| **ACT** | 早期拼接(CVAE) | ResNet18 | 不使用 | 拼接 | 无 | ~100 |
| **RoboFlamingo** | 晚期(Cross-Attn) | CLIP ViT | Flamingo 交叉注意力 | 拼接 | 多帧 | ~300 |
| **Diffusion Policy** | 晚期(FiLM/拼接) | ResNet/ViT | 不使用/FiLM | 拼接 | 2 帧 | N/A |

## 融合策略选型指南

```
需要语言指令控制？
  │
  ├─ 是 → 使用 VLM 骨干？
  │         ├─ 是 (RT-2/OpenVLA/π₀ 路线) → 早期拼接（最简洁）
  │         └─ 否 → Cross-Attention 或 FiLM 注入语言
  │
  └─ 否 (目标图像/固定任务) → 本体感觉重要？
            ├─ 是 → FiLM 调制（Octo 风格）或拼接
            └─ 否 → 纯视觉 Token 即可
```

### 计算量 vs 信息量权衡

| 增加模态 | Token 增量 | 注意力计算增量 | 信息收益 |
|---------|-----------|-------------|---------|
| +1 个相机视角 | +256 | ~+50% (N=512→768) | 减少遮挡 |
| +语言指令 | +20 | ~+4% | 任务指定 |
| +本体感觉 | +1-2 | ~<1% | 自身状态 |
| +2 帧历史 | +512 | ~+100% (N=512→1024) | 运动趋势 |

**关键经验**：语言和本体感觉的信息密度极高（很少的 Token 带来大量信息），是最"性价比"高的模态。额外相机和历史帧的 Token 数量大，需要权衡计算成本。

## 小结

| 概念 | 要点 |
|------|------|
| 早期融合 | 所有模态 Token 化→拼接→统一 Transformer，最简洁 |
| 晚期融合 | 各模态独立编码→融合模块合并，更灵活 |
| FiLM | 用条件特征的 γ/β 调制目标特征，极轻量 |
| Cross-Attention | 一种模态"查询"另一种模态的信息 |
| 本体感觉 | MLP Token 化或 FiLM 注入，信息密度高 |
| 历史帧 | 帧堆叠/Perceiver 压缩/因果 Transformer |
| 主流选择 | VLM-based VLA 多用早期拼接（RT-2/OpenVLA/π₀） |

---

> **下一篇**：[VLA 架构对比表](./05-architecture-comparison) — 所有模型的四维度对比总览。

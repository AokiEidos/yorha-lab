---
title: "多模态与大模型融合"
date: 2026-04-20T17:32:38.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 多模态与大模型融合

> 🔬 深入 | 前置知识：[文生图技术栈](./01-text-to-image.md)，[DiT 架构](./02-dit.md)

## LLM + Diffusion：为什么需要融合？

大语言模型（LLM）擅长**理解和推理**——解析复杂指令、逻辑推理、多步规划。Diffusion 模型擅长**视觉生成**——高保真图像/视频合成。将两者结合是构建**通用多模态 AI** 的核心方向。

当前 LLM + Diffusion 的融合主要有三种架构范式，每种代表了不同的技术哲学和工程权衡。

---

## 架构 A：LLM 理解 + Diffusion 生成（管线式）🔰

### 核心思想

LLM 和 Diffusion 作为**独立模块**通过管线串联——LLM 负责理解用户意图和优化 prompt，Diffusion 负责根据优化后的 prompt 生成图像。

```
用户: "画一只在海滩上玩耍的柯基"
  │
  ▼
LLM (GPT-4等): 理解意图 → 重写 prompt → "A cute Welsh Corgi puppy playing
  on a sandy beach at golden hour, ocean waves, warm sunlight, 4K..."
  │
  ▼
Diffusion Model (DALL-E 3等): 根据详细 prompt 生成高质量图像
  │
  ▼
生成结果
```

### DALL-E 3 的 Prompt 重写机制

DALL-E 3 是架构 A 的最佳实践案例。其核心创新不在 Diffusion 模型本身，而在 **prompt 工程的自动化**：

1. **Caption Improvement**：用 GPT-4V 为训练图像重新生成详细准确的描述（替代原始的网络爬取 alt-text），显著提升训练数据质量
2. **Prompt Rewriting**：推理时 ChatGPT 自动将用户的简短描述扩展为详细的图像 prompt，弥补用户与 Diffusion 模型之间的"prompt 工程鸿沟"
3. **安全过滤**：LLM 层面进行安全审查，拒绝不当请求

**效果**：DALL-E 3 在 prompt 遵循度上大幅超越前代，核心原因不是模型架构更强，而是训练数据（caption 质量）和推理流程（prompt 重写）的改进。

### 优劣分析

| 优势 | 劣势 |
|------|------|
| 各模块独立优化，技术成熟 | 两阶段串行，无法端到端优化 |
| LLM 的推理能力增强 prompt 质量 | LLM 和 Diffusion 之间信息丢失 |
| 易于部署和维护（模块化） | 延迟较高（两次模型推理） |
| 可以随时替换任一模块 | LLM 无法直接感知/修正图像 |

---

## 架构 B：统一 Token 空间 ⚙️

### 核心思想

将图像和文本都表示为 token，在**同一个 Transformer** 中处理。模型不再区分"理解"和"生成"——所有模态在统一的 token 空间中交互。

```
输入: "描述这张图片" [img_tokens] "并画一张类似的"
  → 文本/图像统一 tokenize → 统一 Transformer → 自回归/扩散生成 → 文本+图像输出
```

### 图像 Token 化的两条路线

#### 路线 B1：离散 Token（VQ-VAE）

将图像编码为离散的 codebook 索引，与文本 token 完全同构：

离散化流程：图像 [B,3,256,256] → VQ-VAE Encoder → 量化为 codebook 索引 → [B,1024] 个离散 token（每个 0~8191）。图像和文本 token 共享同一词表（文本词表 [0,32000) + 图像词表 [32000,40192)）。

**Chameleon（Meta, 2024）** 采用此路线：
- 8192 大小的图像 codebook + 65536 大小的文本词表
- 图像编码为 1024 个离散 token（32×32）
- 自回归 Transformer 统一处理所有 token
- 模型参数 7B / 34B

| 优势 | 劣势 |
|------|------|
| 与 LLM 架构完全统一 | VQ 量化信息损失 → 图像质量有上限 |
| 可以直接用 LLM 训练框架 | 图像 token 序列长（1024+），自回归生成慢 |
| 推理简单（自回归采样） | 小 codebook → 细节丢失；大 codebook → 训练困难 |

#### 路线 B2：连续 Token（VAE + Diffusion）

图像保持连续潜变量表示，在同一 Transformer 中用扩散过程处理：

**Show-o（2024）** 的混合方案：同一个 Transformer 交替两种训练模式——理解模式（图像通过 MAGVIT-v2 编码为离散 token，自回归生成文本）和生成模式（文本作为前缀条件，图像 token 通过离散扩散 mask & predict 方式生成）。这种设计使单一模型同时具备多模态理解和生成能力。

### 离散 vs 连续 Token 对比

| 维度 | 离散 Token (VQ-VAE) | 连续 Token (VAE) |
|------|--------------------|-----------------|
| 表示形式 | 整数索引 (0~8191) | 连续向量 (float) |
| 信息保真度 | 有损（量化误差） | 高保真 |
| 与文本统一性 | 完全统一（都是离散 token） | 需要特殊处理（不同的损失函数） |
| 生成方式 | 自回归 / Masked Prediction | 扩散去噪 |
| 图像质量上限 | 受限于 codebook 大小 | 取决于 VAE 质量 |
| 序列长度 | 长（~1024 tokens for 256² 图像） | 较短（可在潜空间工作） |
| 训练难度 | 较简单（标准 CE loss） | 较复杂（混合损失） |

---

## 架构 C：Diffusion Head on LLM 🔬

### 核心思想

保持 LLM 的自回归 backbone 不变，在其顶部接一个 **Diffusion 解码头（Diffusion Head）**——LLM 的隐状态作为 Diffusion 的条件。

```
[text tokens] [IMG tokens] → 共享 Transformer Backbone
  ├→ 文本 logits → LM Head (交叉熵损失) → 文本输出
  └→ 图像隐状态 → Diffusion Head (去噪损失) → 图像输出
```

### Transfusion（Meta, 2024）的训练目标

**Transfusion** 是架构 C 的代表作，核心创新是在同一 Transformer 上混合两种训练目标：

$$\mathcal{L}_{\text{Transfusion}} = \underbrace{\mathcal{L}_{\text{LM}}}_{\text{文本: 自回归}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{DDPM}}}_{\text{图像: 去噪}}$$

具体实现：

```python
class TransfusionModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.transformer = TransformerDecoder(d_model, n_layers, n_heads)  # 共享 backbone
        self.text_embed = nn.Embedding(vocab_size, d_model)               # 文本嵌入
        self.lm_head = nn.Linear(d_model, vocab_size)                     # 文本 LM head
        self.image_in = nn.Linear(latent_dim, d_model)                    # 图像输入投影
        self.image_out = nn.Linear(d_model, latent_dim)                   # 图像输出（噪声预测）
    
    def forward(self, text_tokens, image_latents, timestep, noise):
        text_emb = self.text_embed(text_tokens)                     # [B, L_text, D]
        noised = add_noise(image_latents, timestep, noise)
        image_emb = self.image_in(noised)                           # [B, L_img, D]
        
        # 拼接序列：[文本] [BOI] [图像patches] [EOI]，共享 Transformer
        hidden = self.transformer(concat([text_emb, BOI, image_emb, EOI]))
        
        loss_lm = F.cross_entropy(self.lm_head(hidden[:, :L_text]), text_targets)
        loss_ddpm = F.mse_loss(self.image_out(hidden[:, L_text+1:L_text+1+L_img]), noise)
        return loss_lm + lambda_img * loss_ddpm
```

### Transfusion 关键实验结果

| 模型规模 | 文本能力 (vs 纯 LLM) | 图像 FID (CC12M) | 说明 |
|---------|-------------------|--------------------|------|
| 0.16B | 可比 | 14.4 | 小规模验证 |
| 0.37B | 可比 | 9.67 | 文本性能不下降 |
| 0.76B | 可比 | 6.78 | 图像质量与专用模型可比 |
| 7B | 接近 Llama 2 7B | ~5.0 (估计) | 统一模型无明显性能损失 |

**关键发现**：文本和图像的联合训练**没有显著损害任一模态的性能**——这证明了统一架构的可行性。

---

## 三种架构全面对比

| 维度 | 架构 A (Pipeline) | 架构 B (Unified Token) | 架构 C (Diffusion Head) |
|------|-------------------|----------------------|------------------------|
| 代表 | DALL-E 3, GPT-4o(推测) | Chameleon, Show-o | Transfusion |
| 端到端训练 | 否 | 是 | 是 |
| 图像质量 | 最高（专用模型） | 中（VQ 量化损失） | 中高 |
| 文本能力 | 不影响 LLM | 可能略有损失 | 基本不影响 |
| 推理速度 | 慢（两次推理） | 中（长序列自回归） | 中（混合推理） |
| 工程复杂度 | 低（模块化） | 高（统一训练） | 高（混合损失） |
| 理解+生成统一 | 否（分离模块） | 是 | 是 |
| 实时交互 | 困难 | 可能 | 可能 |
| 多轮图像对话 | LLM 层面支持 | 原生支持 | 原生支持 |

---

## Diffusion 在多模态 Agent 中的角色 ⚙️

多模态 Agent 体系中，Diffusion 扮演的角色正在从"生成工具"演变为"世界模型组件"：

### 工具使用范式（当前主流）

LLM Agent 分析需求，通过 API 调用 Diffusion 模型完成视觉生成任务：

```
用户: "帮我设计一个蓝色主题的 logo"
  → LLM Agent: 分析需求 → 生成 prompt
    → tool_call: text_to_image("minimalist blue logo, geometric...")
    → tool_call: image_edit(result, "add gradient effect")
    → 评估结果（用视觉模型检查）→ 返回 / 迭代修改
```

这种范式下 Diffusion 是 Agent 工具箱中的一个"视觉生成工具"，与搜索、代码执行等工具并列。

### 世界模型范式（前沿方向）

Diffusion 不再仅仅是"生成工具"，而是成为 Agent 的**想象力引擎**——生成"如果执行动作 A，世界会变成什么样"的预测图像，辅助 Agent 规划和决策。

这与自动驾驶中的**世界模型（World Model）** 理念相通（参见 [世界模型](../06-autonomous-driving/06-world-model.md)），也是 VLA（Vision-Language-Action）架构的核心思想——Agent 通过 Diffusion World Model 预测动作后果，选择最优策略。

| 范式 | Diffusion 角色 | 交互方式 | 代表 |
|------|---------------|---------|------|
| 工具使用 | 外部工具 | API 调用 | ChatGPT + DALL-E 3 |
| 世界模型 | 内部组件 | 隐式预测 | Genie, UniSim |
| 端到端生成 | 模型本身 | 原生输出 | GPT-4o (推测) |

---

## 发展趋势与未来方向

### 趋势一：自回归 + 扩散混合

纯自回归（Chameleon）图像质量受 VQ 量化限制；纯扩散推理灵活性不足。**混合方案**是当前最有前景的方向：

- 文本用自回归：保持 LLM 推理和上下文学习能力
- 图像用扩散：保持高保真生成质量
- 共享 Transformer backbone：统一训练，跨模态知识互通

### 趋势二：规模与模态扩展

| 阶段 | 时间 | 模型规模 | 模态 | 代表 |
|------|------|---------|------|------|
| 专用模型 | 2022-2023 | ~1-3B | 文本→图像 | SD 1.5, DALL-E 2 |
| 大模型 | 2023-2024 | ~7-13B | 文本↔图像↔视频 | FLUX, SD3 |
| 基础模型 | 2024-2025 | ~30-100B | +音频+3D+代码 | GPT-4o, Gemini |
| 下一代 | 2025+ | ~100B+ | +动作+物理模拟 | 通用世界模型 |

### 趋势三：训练效率优化

| 技术 | 效果 | 应用 |
|------|------|------|
| 渐进式分辨率训练 | 降低 3-5x 训练成本 | SDXL, FLUX |
| 合成数据自举 | 减少对人工标注的依赖 | DALL-E 3 caption |
| 模态专家 (MoE) | 减少模态间干扰 | Gemini |
| 蒸馏 | 小模型继承大模型能力 | SDXL Turbo |

模态统一的终极方向与**具身智能（Embodied AI）** 密切相关——扩散模型作为"世界模型"组件，为 Agent 提供对物理世界的预测和理解能力，推动 AI 从"理解语言和图像"走向"理解和交互物理世界"。

---

## 小结

| 概念 | 要点 |
|------|------|
| 架构 A (Pipeline) | LLM 改写 prompt + Diffusion 生成，模块化成熟，DALL-E 3 是最佳实践 |
| 架构 B (Unified) | 统一 token 空间，离散(Chameleon)或混合(Show-o)，理解+生成统一 |
| 架构 C (Diff Head) | LLM backbone + Diffusion Head，Transfusion 证明文本和图像不冲突 |
| 离散 vs 连续 | 离散 token 统一性好但质量受限；连续潜变量质量高但需混合训练 |
| Agent 中的角色 | 工具使用（当前）→ 世界模型（未来），与 VLA 方向密切相关 |
| 融合趋势 | 自回归+扩散混合、更大规模、更多模态、更高效训练 |

---

> **下一篇**：[自动驾驶与 Diffusion 全景](../06-autonomous-driving/01-overview.md) — 进入模块六，Diffusion 在自动驾驶中的应用。

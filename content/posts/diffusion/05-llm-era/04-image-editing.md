---
title: "图像编辑"
date: 2026-04-20T17:31:53.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 图像编辑

> ⚙️ 进阶 | 前置知识：[DDIM 详解](../02-models-zoo/02-ddim)，[CFG](../03-conditional-generation/03-cfg)，[采样算法](../01-fundamentals/04-sampling-algorithms)

## 编辑 vs 生成

图像生成从纯噪声出发创造全新图像；图像编辑则要**修改已有图像**——保留不需要改变的部分，精确修改目标区域。Diffusion 模型天然的"加噪→去噪"过程为编辑提供了优雅的框架：加噪等价于"遗忘"原图信息，去噪等价于用新条件"重新构建"。控制加噪程度就控制了编辑强度。

本文涵盖五大类编辑方法，从简单到复杂依次展开。

---

## 一、SDEdit：加噪再去噪 🔰

**SDEdit（Stochastic Differential Editing）**（Meng et al., 2021）是最简单直接的编辑方法：

### 工作流程

```
原图 x₀ ──加噪到 t₀──▶ x_{t₀} ──用新 prompt 去噪──▶ 编辑后的图像

                  噪声强度 t₀ 控制编辑强度
   t₀ = 0.2                    t₀ = 0.5                    t₀ = 0.8
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ 微调颜色/风格│            │ 改变内容     │            │ 接近重新生成│
│ 结构保持完好 │            │ 大结构保持   │            │ 结构大幅改变│
│ 编辑可控性高 │            │ 平衡点       │            │ 原图信息少 │
└─────────────┘            └─────────────┘            └─────────────┘
```

### 噪声强度 vs 编辑强度的量化分析

| $t_0$ 范围 | 效果 | LPIPS (与原图) | SSIM |
|-----------|------|---------------|------|
| 0.1 - 0.2 | 色调/风格微调，结构完好 | ~0.05 | ~0.98 |
| 0.3 - 0.5 | 内容替换，大结构保持 | ~0.3 | ~0.6 |
| 0.6 - 0.8 | 构图大幅改变，接近重新生成 | ~0.6 | ~0.3 |
| 0.9 - 1.0 | 等价于从噪声生成 | ~0.7 | ~0.2 |

**数学理解**：加噪到 $t_0$ 后，$x_{t_0}$ 中包含的原图信息量约为 $\bar{\alpha}_{t_0}$（噪声调度的累积系数）。当 $t_0$ 较小时 $\bar{\alpha}_{t_0} \approx 1$，原图信息几乎完全保留；当 $t_0$ 较大时 $\bar{\alpha}_{t_0} \approx 0$，原图信息被噪声淹没。

**局限**：SDEdit 是全局操作，无法做精细的局部编辑。编辑强度和结构保持之间存在不可避免的权衡。

---

## 二、DDIM Inversion + 编辑 ⚙️

### 原理

**DDIM Inversion（DDIM 反演）** 利用 DDIM 采样的确定性可逆性，将真实图像"映射回"噪声空间，然后从同一噪声出发用新条件去噪：

```
        DDIM Inversion (前向)                  DDIM Sampling (反向)
原图 x₀ ───────────────────▶ x_T  ───────────────────────▶ 编辑后图像
         逐步加噪+预测修正          用新 prompt 逐步去噪
         保留结构信息                语义发生变化
```

因为噪声 $x_T$ 编码了原图的结构信息，用新条件去噪时结构大体保留，但语义会根据新条件发生变化。

### DDIM Inversion 的数学推导

DDIM 采样公式（从 $x_t$ 到 $x_{t-1}$）：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{预测的 } x_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**反演（Inversion）** 就是反过来——已知 $x_{t-1}$，求 $x_t$。假设 $\epsilon_\theta(x_t, t) \approx \epsilon_\theta(x_{t-1}, t-1)$（关键近似），可得：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot \frac{x_{t-1} - \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_{t-1}, t-1)}{\sqrt{\bar{\alpha}_{t-1}}} + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(x_{t-1}, t-1)$$

### 为什么 Euler Inversion 会累积误差？🔬

上面的关键近似 $\epsilon_\theta(x_t, t) \approx \epsilon_\theta(x_{t-1}, t-1)$ 在每一步都引入一个小的局部截断误差 $\delta_t$。经过 $T$ 步反演后，总误差累积为：

$$\|x_T^{\text{inv}} - x_T^{\text{exact}}\| \approx \sum_{t=1}^{T} \delta_t \cdot \prod_{s=t+1}^{T} L_s$$

其中 $L_s$ 是去噪网络的局部 Lipschitz 常数。当 $T$ 较大且 $L_s > 1$ 时，误差会**指数级放大**。

**实际影响**：用 50 步 DDIM Inversion 反演后再重建，PSNR 通常仅为 25-30 dB（肉眼可见差异）。在 CFG 引导下误差更大，因为 CFG 放大了条件与无条件预测之间的差异。

### Null-text Inversion：精确反演 🔬

**Null-text Inversion**（Mokady et al., 2023）通过优化无条件文本嵌入 $\varnothing_t$ 来消除反演误差：

```python
# Null-text Inversion 核心算法
def null_text_inversion(model, x_0, prompt, num_steps=50, opt_steps=10):
    x_t_list = ddim_inversion(model, x_0, prompt, num_steps)  # 标准 DDIM Inversion
    null_embeds = [model.null_embed.clone().requires_grad_(True) for _ in range(num_steps)]
    
    x_t = x_t_list[-1]
    for t in reversed(range(num_steps)):
        optimizer = torch.optim.Adam([null_embeds[t]], lr=1e-2)
        for _ in range(opt_steps):
            noise_cond = model.predict_noise(x_t, t, prompt_embed)
            noise_uncond = model.predict_noise(x_t, t, null_embeds[t])
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            loss = F.mse_loss(ddim_step(x_t, noise_pred, t), x_t_list[t])
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        x_t = ddim_step_with_null(x_t, t, prompt_embed, null_embeds[t])
    return x_t, null_embeds  # x_t ≈ x_0 (精确重建)
```

编辑时只需换 prompt，保留优化后的 $\varnothing_t$ 序列。

---

## 三、Prompt-to-Prompt：注意力图操控 🔬

**Prompt-to-Prompt**（Hertz et al., 2022）是 Diffusion 编辑的里程碑工作。核心发现：**Cross-Attention map 编码了"文本词→图像区域"的对应关系**，操控注意力 map 就能精确编辑图像。

### 核心观察

在 U-Net 的 Cross-Attention 层中：
- Query $Q$ 来自图像特征
- Key $K$, Value $V$ 来自文本嵌入
- Attention map $A = \text{softmax}(QK^T/\sqrt{d})$ 的每一行表示"这个图像位置关注哪些文本 token"

例如 prompt "a **cat** sitting on a **chair**"：
- "cat" 对应的注意力 map 高亮在猫的区域
- "chair" 对应的注意力 map 高亮在椅子的区域

### 三种编辑操作的伪代码

```python
class PromptToPromptEditor:
    """Prompt-to-Prompt 注意力操控编辑器"""
    
    def word_swap(self, attn_maps_src, attn_maps_tgt, swap_indices, 
                  cross_replace_steps=0.8):
        """
        词替换：将 "cat" 换成 "dog"
        策略：在早期步 (t > 0.2T) 用源注意力 map 替换目标注意力 map
        保证空间布局不变，只改变语义内容
        """
        T = len(attn_maps_src)
        for t in range(T):
            if t / T < cross_replace_steps:  # 前 80% 的步数
                for idx in swap_indices:      # 被替换词的位置
                    attn_maps_tgt[t][:, :, idx] = attn_maps_src[t][:, :, idx]
        return attn_maps_tgt
    
    def attention_refine(self, attn_maps_src, attn_maps_tgt, 
                         self_replace_steps=0.4):
        """
        添加修饰词："a cat" → "a fluffy cat"
        策略：保留源的 self-attention map（保持布局），
              但允许新词的 cross-attention 生效
        """
        T = len(attn_maps_src)
        for t in range(T):
            if t / T < self_replace_steps:    # 前 40% 的步数
                # Self-attention map 来自源图
                attn_maps_tgt[t]['self'] = attn_maps_src[t]['self']
        return attn_maps_tgt
    
    def attention_reweight(self, attn_maps, word_idx, scale_factor):
        """
        注意力权重调整：增强/减弱某个词的影响
        例如：增强 "red" 的权重使颜色更鲜艳
        """
        for t in range(len(attn_maps)):
            # 直接缩放特定词的注意力权重
            attn_maps[t][:, :, word_idx] *= scale_factor
            # 重新归一化
            attn_maps[t] = attn_maps[t] / attn_maps[t].sum(dim=-1, keepdim=True)
        return attn_maps
```

### 操作效果对比

| 操作 | 源 Prompt | 目标 Prompt | 效果 | 控制精度 |
|------|----------|------------|------|---------|
| 词替换 | "a **cat** on a chair" | "a **dog** on a chair" | 猫→狗，椅子不变 | 高 |
| 修饰添加 | "a cat on a chair" | "a **fluffy** cat on a chair" | 猫变蓬松，其他不变 | 中高 |
| 权重调整 | "a **red** car" (w=1.0) | "a **red** car" (w=2.0) | 红色更鲜艳 | 高 |
| 全局替换 | "a photo of a street" | "a painting of a street" | 照片→绘画风格 | 中 |

---

## 四、InstructPix2Pix：指令式编辑 ⚙️

**InstructPix2Pix**（Brooks et al., 2023）将编辑建模为端到端的条件生成任务——给定原图和自然语言指令，直接输出编辑后的图像。

### 训练数据生成管线 🔬

InstructPix2Pix 最大的创新在于**如何构建训练数据**——完全合成，无需人工标注：

```
Step 1: GPT-3 生成编辑指令对
┌──────────────────────────────────────────────────────┐
│ 输入: "a photo of a girl riding a horse"             │
│                                                      │
│ GPT-3 输出:                                          │
│   指令: "have her ride a dragon"                     │
│   编辑后 caption: "a photo of a girl riding a dragon"│
│                                                      │
│ 生成 ~450K 编辑指令对                                 │
└──────────────────────────────────────────────────────┘
          │
          ▼
Step 2: Prompt-to-Prompt 生成图像对
┌──────────────────────────────────────────────────────┐
│ 源 prompt → SD 生成源图像                             │
│ 目标 prompt → P2P 编辑生成目标图像                     │
│                                                      │
│ 过滤：CLIP 方向一致性 + LPIPS 差异阈值               │
│ 最终 ~300K 有效三元组 (源图, 指令, 目标图)             │
└──────────────────────────────────────────────────────┘
          │
          ▼
Step 3: 训练 InstructPix2Pix 模型
┌──────────────────────────────────────────────────────┐
│ 架构：SD 1.5 + 额外输入通道（原图拼接）               │
│ 输入：[z_t (4ch) | encode(源图) (4ch)] = 8 通道       │
│ 条件：编辑指令（通过 Cross-Attention）                 │
│ 目标：预测噪声 ε → 还原目标图的潜变量                  │
│                                                      │
│ 双 CFG 推理：                                         │
│   ε = ε_uncond                                       │
│     + s_I · (ε_img - ε_uncond)       # 图像引导强度   │
│     + s_T · (ε_full - ε_img)         # 文本引导强度   │
└──────────────────────────────────────────────────────┘
```

**双 CFG 的直觉**：$s_I$ 控制"保持原图多少"（越大越像原图），$s_T$ 控制"遵循编辑指令多少"（越大编辑越强）。两个旋钮独立调节。

---

## 五、Inpainting：局部修复与替换 ⚙️

**Inpainting（图像修复/局部替换）** 给定图像和二值遮罩，重新生成遮罩区域的内容。

### 架构设计

SD Inpainting 模型在标准 U-Net 上做了最小修改——扩展输入通道：

```
标准 SD:      [z_t]           → 4 通道输入 → U-Net → 预测噪声
                                │
SD Inpainting: [z_t | z_masked | mask] → 4+4+1 = 9 通道输入 → U-Net → 预测噪声
                │       │        │
              噪声潜变量  被遮罩的   二值遮罩
              (去噪目标)  原图潜变量  (1=需生成)
```

新增的 5 个输入通道的权重从零初始化，确保可以从预训练 SD 权重微调而不破坏已学到的知识。

**Blended Diffusion** 技巧：无需专门训练 inpainting 模型——每个去噪步中，遮罩外区域直接替换为原图加噪到对应时间步的值：`z_t = mask * z_denoised + (1 - mask) * add_noise(z_original, t)`。

---

## 编辑方法全面对比

| 方法 | 需要 Inversion | 需要微调 | 编辑粒度 | 交互方式 | 推理速度 | 效果质量 |
|------|---------------|---------|---------|---------|---------|---------|
| SDEdit | 否 | 否 | 全局 | prompt + 强度 | 快 | 中 |
| DDIM Inv + 新 prompt | 是 | 否 | 全局 | 新 prompt | 中 | 中高 |
| Null-text Inversion | 是 | 否（但需优化） | 全局 | 新 prompt | 慢（~5min） | 高 |
| Prompt-to-Prompt | 是 | 否 | 词级别 | prompt 编辑 | 中 | 高 |
| InstructPix2Pix | 否 | 否 | 全局/局部 | 自然语言指令 | 快（~5s） | 中高 |
| SD Inpainting | 否 | 否 | 遮罩区域 | 遮罩 + prompt | 快 | 高 |
| DragDiffusion | 是 | 否 | 点级别 | 拖拽控制点 | 慢 | 高 |

### 选型建议

| 需求场景 | 推荐方法 | 原因 |
|---------|---------|------|
| 全局风格/氛围变换 | SDEdit ($t_0 \approx 0.5$) | 简单高效，无需训练 |
| 精细语义编辑（换物体） | Prompt-to-Prompt | 词级控制，空间精确 |
| 自然语言指令编辑 | InstructPix2Pix | 用户体验最自然，一步完成 |
| 局部替换/移除物体 | Inpainting | 专门优化局部生成的一致性 |
| 精确保持结构的重绘 | Null-text Inversion + 新 prompt | 重建精度最高 |
| 位置/形态调整 | DragDiffusion | 直觉的拖拽交互 |

---

## 小结

| 概念 | 要点 |
|------|------|
| SDEdit | 加噪 $t_0$ 控制编辑强度，简单但只能全局编辑 |
| DDIM Inversion | Euler 近似带来累积误差，CFG 下误差更大 |
| Null-text Inversion | 逐步优化无条件嵌入 $\varnothing_t$ 消除反演误差 |
| Prompt-to-Prompt | 操控 Cross-Attention map 实现词替换/修饰/权重调整 |
| InstructPix2Pix | GPT-3+P2P 合成训练数据，双 CFG 独立控制图像/文本引导 |
| Inpainting | 扩展输入通道（+5ch），遮罩区域条件生成 |

---

> **下一篇**：[超分辨率与图像修复](./05-super-resolution) — Diffusion 如何在底层视觉任务中超越传统方法。

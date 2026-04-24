---
title: "Classifier-Free Guidance (CFG)"
date: 2026-04-20T17:19:50.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Classifier-Free Guidance (CFG)

> ⚙️ 进阶 | 前置知识：[Classifier Guidance](./02-classifier-guidance.md)

## 核心动机

Classifier Guidance 需要额外训练分类器，且难以推广到文本等条件。**无分类器引导（Classifier-Free Guidance, CFG）**（Ho & Salimans, 2022）优雅地解决了这个问题：让**同一个扩散模型同时学习条件和无条件生成**，推理时用两者的差作为引导信号。

CFG 是当前几乎所有文生图/文生视频模型的标配技术。从 Stable Diffusion 到 DALL-E 3，从 Imagen 到 FLUX，无一例外地使用 CFG。

## 训练方式

训练时，以概率 $p_{\text{uncond}}$（通常 10-20%）随机将条件 $c$ 替换为空条件 $\varnothing$（如空文本 ""）：

```python
def cfg_training_step(model, x_0, condition, p_uncond=0.1):
    """CFG 训练：随机丢弃条件"""
    t = torch.randint(0, T, (batch_size,))
    eps = torch.randn_like(x_0)
    x_t = forward_diffuse(x_0, t, eps)
    
    # 随机丢弃条件（核心技巧）
    mask = torch.rand(batch_size) < p_uncond
    cond_input = [null_cond if m else c for m, c in zip(mask, condition)]
    
    eps_pred = model(x_t, t, cond_input)
    loss = F.mse_loss(eps_pred, eps)
    return loss
```

这样，模型学会了两种模式：
- $\epsilon_\theta(x_t, t, c)$：条件预测（"给定文本描述，图像应该是什么样"）
- $\epsilon_\theta(x_t, t, \varnothing)$：无条件预测（"图像一般应该是什么样"）

**为什么这么简单就够了？** 因为相同的网络结构已经有足够的容量学习两种模式——条件输入为空时，cross-attention 层接收空嵌入，信息完全来自图像自身；条件不为空时，cross-attention 层将文本语义注入图像特征。

## 推理公式

采样时，将条件和无条件预测做线性组合：

$$\boxed{\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \left[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\right]}$$

其中 $w$ 是**引导强度（Guidance Scale）**。

### 公式解读

$$\hat{\epsilon} = \underbrace{\epsilon_\theta(\varnothing)}_{\text{无条件基线}} + w \cdot \underbrace{[\epsilon_\theta(c) - \epsilon_\theta(\varnothing)]}_{\text{条件方向}}$$

- $\epsilon_\theta(\varnothing)$：无条件预测——"一般的图像"
- $\epsilon_\theta(c) - \epsilon_\theta(\varnothing)$：条件方向——"文本条件让图像偏移的方向"
- $w$：沿这个方向走多远

当 $w = 1$ 时，等价于标准条件生成 $\epsilon_\theta(c)$。当 $w > 1$ 时，沿条件方向"过度"偏移——生成更忠于文本但更极端的图像。

## 与 Classifier Guidance 的数学等价性

🔬 深入

**定理**：CFG 隐式等价于 Classifier Guidance，其中隐式分类器为：

$$\log p_{\text{implicit}}(c|x_t) = \log p(x_t|c) - \log p(x_t) + \text{const}$$

**推导**：

CFG 的修改后得分函数为：

$$\nabla_{x_t} \log p_w(x_t|c) = \nabla_{x_t} \log p(x_t) + w \cdot [\nabla_{x_t} \log p(x_t|c) - \nabla_{x_t} \log p(x_t)]$$

整理得：

$$= (1 - w) \nabla_{x_t} \log p(x_t) + w \cdot \nabla_{x_t} \log p(x_t|c)$$

$$= \nabla_{x_t} \log p(x_t) + w \cdot [\nabla_{x_t} \log p(x_t|c) - \nabla_{x_t} \log p(x_t)]$$

由贝叶斯定理 $\nabla_{x_t} \log p(x_t|c) - \nabla_{x_t} \log p(x_t) = \nabla_{x_t} \log p(c|x_t)$，因此：

$$\nabla_{x_t} \log p_w(x_t|c) = \nabla_{x_t} \log p(x_t) + w \cdot \nabla_{x_t} \log p(c|x_t)$$

这**恰好**是 Classifier Guidance 公式，其中引导强度 $s = w$，隐式分类器 $p(c|x_t) \propto \frac{p(x_t|c)}{p(x_t)}$。

CFG 的巧妙之处在于：它不需要显式训练分类器，而是让扩散模型自身隐式地"学会"了分类。

## Guidance Scale 的效果

$w$ 对生成结果有显著影响：

| $w$ 值 | 效果 | 典型场景 | FID (COCO) | CLIP Score |
|--------|------|---------|-----------|------------|
| 1.0 | 标准条件生成，多样性高 | 探索性生成 | ~12.0 | 0.28 |
| 3.0-5.0 | 平衡质量和多样性 | 一般用途 | ~9.5 | 0.31 |
| 7.0-8.5 | 高质量、高文本匹配 | **SD 默认** | ~8.0 | 0.33 |
| 10.0-15.0 | 非常忠于文本，但过度饱和 | 特殊需求 | ~10.5 | 0.34 |
| 20.0+ | 过度引导，伪影明显 | 不推荐 | ~18.0 | 0.35 |

### 质量-多样性权衡的可视化

```
多样性高                                    文本匹配度高
<----------------------------------------------->
w=1      w=3      w=5      w=7.5    w=15    w=30
随机      温和      平衡      精确      过度     崩坏
多样      引导      区域      匹配      饱和     伪影

FID:  12.0    9.5     8.5     8.0     10.5    18.0
      \___________ 最优区间 ___________/
```

$w$ 过大时的典型问题：
- 颜色过度饱和（蓝天变成纯蓝）
- 高频细节丢失（过于平滑）
- 图像"塑料感"——缺乏自然纹理
- 极端情况下出现高对比度伪影

### 为什么高 $w$ 会产生过度饱和？

🔬 深入

从数学上看，高 $w$ 让噪声预测 $\hat{\epsilon}$ 的幅度远超正常范围。在 $x_0$ 预测空间中：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

$\hat{\epsilon}$ 幅度过大 $\Rightarrow$ $\hat{x}_0$ 的像素值超出 $[-1, 1]$ 范围 $\Rightarrow$ clipping 后丢失细节 $\Rightarrow$ 颜色饱和、纹理消失。

## 动态引导策略

### 时间步自适应 Guidance

一些改进方法在不同时间步使用不同的 $w$：

```python
def dynamic_cfg_sample(model, x_t, t, condition, w_max=7.5, w_min=1.5):
    """动态 CFG：高噪声时强引导，低噪声时弱引导"""
    # 线性衰减
    w = w_max - (w_max - w_min) * (1 - t / T)
    
    # 或余弦衰减
    # w = w_min + (w_max - w_min) * 0.5 * (1 + cos(pi * (1 - t/T)))
    
    eps_uncond = model(x_t, t, null_condition)
    eps_cond = model(x_t, t, condition)
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    return eps
```

**直觉**：
- **高噪声时** ($t$ 大)：高 $w$ 确定全局结构和语义布局
- **低噪声时** ($t$ 小)：低 $w$ 保留细节和多样性，避免过度饱和

### CFG++（Chung et al., 2024）

**CFG++** 对标准 CFG 做了一个简单但有效的修改——将引导信号应用到 $x_0$ 预测空间而非 $\epsilon$ 空间：

$$\hat{x}_0^{\text{CFG++}} = x_0^{\text{uncond}} + w \cdot (x_0^{\text{cond}} - x_0^{\text{uncond}})$$

然后再从 $\hat{x}_0$ 反推 $\hat{\epsilon}$。

**优势**：
- 在 $x_0$ 空间做引导，幅度更可控
- 相同 $w$ 下过度饱和更少
- 生成质量在高 $w$ 时更稳定
- 无需修改训练流程，纯推理时改动

## 负提示（Negative Prompting）

### 机制

负提示是 CFG 的自然扩展：将无条件预测 $\epsilon_\theta(\varnothing)$ 替换为**负面条件**预测 $\epsilon_\theta(c_{\text{neg}})$：

$$\hat{\epsilon} = \epsilon_\theta(c_{\text{neg}}) + w \cdot [\epsilon_\theta(c_{\text{pos}}) - \epsilon_\theta(c_{\text{neg}})]$$

**直觉**：不再是"从一般图像方向走向目标图像"，而是"从负面图像方向走向目标图像"——条件方向同时远离不想要的内容。

常见负提示：`"low quality, blurry, deformed, ugly, bad anatomy"`

```python
def negative_prompt_cfg(model, x_t, t, pos_cond, neg_cond, w=7.5):
    """带负提示的 CFG"""
    eps_neg = model(x_t, t, neg_cond)      # 替代无条件预测
    eps_pos = model(x_t, t, pos_cond)
    eps = eps_neg + w * (eps_pos - eps_neg)
    return eps
```

### 负提示的效果

| 设置 | FID | 人类偏好率 |
|------|-----|-----------|
| 无负提示 | 8.10 | 基准 |
| 标准负提示（质量相关） | 7.85 | +15% |
| 定制负提示（场景相关） | 7.62 | +22% |

## 批处理优化实现

### 标准实现（2 次前向传播）

```python
@torch.no_grad()
def cfg_sample_step_naive(model, x_t, t, condition, w=7.5):
    """朴素实现：两次单独前向传播"""
    eps_uncond = model(x_t, t, null_condition)    # 第 1 次前向
    eps_cond = model(x_t, t, condition)            # 第 2 次前向
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    return eps
```

### 批处理优化（1 次前向传播）

```python
@torch.no_grad()
def cfg_sample_step_batched(model, x_t, t, condition, w=7.5):
    """
    批处理优化：将两次前向传播合并为一次
    
    原理：将条件和无条件输入拼成 2B 的 batch，
    利用 GPU 并行性在一次前向中完成两次计算。
    实测在 A100 上比两次单独前向快 ~30%。
    """
    B = x_t.shape[0]
    
    # 拼接输入
    x_input = torch.cat([x_t, x_t], dim=0)                      # [2B, C, H, W]
    t_input = torch.cat([t, t], dim=0)                            # [2B]
    c_input = torch.cat([null_condition.expand(B, -1, -1),        # [2B, seq_len, dim]
                         condition], dim=0)
    
    # 单次前向传播
    eps_both = model(x_input, t_input, c_input)                   # [2B, C, H, W]
    
    # 拆分结果
    eps_uncond, eps_cond = eps_both.chunk(2, dim=0)               # 各 [B, C, H, W]
    
    # CFG 组合
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    return eps
```

### 内存与速度权衡

| 实现方式 | GPU 内存 | 计算时间 | 适用场景 |
|---------|---------|---------|---------|
| 两次前向 | 1x | 2x | GPU 内存紧张 |
| Batch 拼接 | ~1.7x | ~1.3x | **推荐（默认）** |
| 交替计算 + 缓存 | 1x + cache | ~1.5x | 大模型内存不足 |

## 不同模型的 CFG 配置

| 模型 | 默认 $w$ | $p_{\text{uncond}}$ | 空条件 $\varnothing$ | 特殊说明 |
|------|---------|---------------------|---------------------|---------|
| SD 1.5 | 7.5 | 10% | 空文本 "" 的 CLIP 嵌入 | 最经典配置 |
| SDXL | 5.0-7.0 | 10% | 同上 | 更大模型需要更低 $w$ |
| SD3 | 4.0-5.0 | 10% | 空文本（三编码器） | Flow Matching，$w$ 更低 |
| FLUX | 3.5 | — | — | 部分版本无 CFG |
| DALL-E 3 | — | — | — | 使用改进版 CFG |
| Imagen | 10-15 | 10% | 空文本的 T5 嵌入 | T5 编码器，$w$ 偏高 |

**趋势**：随着模型能力增强，所需的 $w$ 越来越小。SD 1.5 需要 7.5，SD3 只需要 4.0-5.0，FLUX 甚至可以不用 CFG——说明更强的模型本身的条件能力更好，不需要那么多"额外推动"。

## 为什么 CFG 成为主流

1. **无需额外模型**：只需训练一个模型（偶尔丢弃条件）
2. **通用性**：适用于任何条件类型（文本、图像、类别标签等）
3. **效果优异**：简单但效果极好
4. **可调节**：引导强度 $w$ 提供直觉的质量-多样性控制
5. **与其他方法兼容**：可以和 ControlNet、IP-Adapter、LoRA 等自由组合

## 小结

| 概念 | 要点 |
|------|------|
| 训练 | 随机丢弃条件（10-20%），模型学会条件+无条件 |
| 推理公式 | $\hat{\epsilon} = \epsilon(\varnothing) + w[\epsilon(c) - \epsilon(\varnothing)]$ |
| 隐式分类器 | 数学上等价于 Classifier Guidance，$p(c\|x) \propto p(x\|c)/p(x)$ |
| $w$ 参数 | 控制文本匹配强度，SD 1.5 常用 7-8.5，SD3 常用 4-5 |
| 负提示 | 将 $\epsilon(\varnothing)$ 替换为 $\epsilon(c_{\text{neg}})$ |
| 动态引导 | 高噪声时大 $w$，低噪声时小 $w$ |
| CFG++ | 在 $x_0$ 空间做引导，减少过度饱和 |
| 批处理优化 | 拼接 batch 单次前向，速度提升约 30% |
| 地位 | 当前主流条件生成方法，几乎所有模型标配 |

---

> **下一篇**：[ControlNet](./04-controlnet.md)

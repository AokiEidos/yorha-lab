---
title: "Classifier Guidance"
date: 2026-04-20T17:18:25.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Classifier Guidance

> ⚙️ 进阶 | 前置知识：[得分函数与得分匹配](../01-fundamentals/06-score-matching.md)，[条件生成概述](./01-overview.md)

## 核心思想

**分类器引导（Classifier Guidance）**（Dhariwal & Nichol, 2021）是第一个让扩散模型在 ImageNet 上超越 GAN 的方法。核心思想：训练一个能在噪声图像上工作的分类器，用它的梯度在采样时引导生成方向。

这篇论文（通常被称为"ADM"论文，Ablated Diffusion Model）在扩散模型历史上具有里程碑意义——它终结了"GAN 是唯一能生成高质量图像的方法"这一认知。

## 完整数学推导

### 从贝叶斯定理出发

我们想从条件分布 $p(x_t|y)$ 采样（$y$ 是类别标签）。利用贝叶斯定理：

$$p(x_t|y) = \frac{p(y|x_t) p(x_t)}{p(y)}$$

两边取对数：

$$\log p(x_t|y) = \log p(y|x_t) + \log p(x_t) - \log p(y)$$

对 $x_t$ 求梯度（$\log p(y)$ 与 $x_t$ 无关，梯度为零）：

$$\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t)$$

### 与得分函数的联系

回顾：扩散模型训练的得分网络 $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$，而噪声预测网络与得分函数的关系为：

$$\nabla_{x_t} \log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

因此条件得分可以写为：

$$\nabla_{x_t} \log p(x_t|y) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}} + \nabla_{x_t} \log p_\phi(y|x_t)$$

这等价于修改噪声预测：

$$\hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

### 引导强度缩放

引入缩放因子 $s$（guidance scale）来控制引导强度：

$$\nabla_{x_t} \log p_s(x_t|y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{无条件得分}} + s \cdot \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{分类器梯度}}$$

- $s = 0$：无条件生成（完全忽略类别）
- $s = 1$：标准贝叶斯后验条件生成
- $s > 1$：增强条件效果（等价于从"锐化"的分布 $p(y|x_t)^s$ 中采样）

🔬 深入

当 $s > 1$ 时，数学上等价于从温度缩放后的分布采样：

$$p_s(x_t|y) \propto p(x_t) \cdot p(y|x_t)^s$$

这将分类器的概率密度"锐化"——高概率区域变得更高，低概率区域变得更低。生成的图像更"典型"但多样性降低。

## 带噪分类器的训练

### 为什么需要噪声分类器

关键细节：分类器需要能在**任意噪声水平**的图像上工作。普通的 ImageNet 分类器只在干净图像上训练过，面对噪声图像会失效——梯度不可靠，甚至指向错误方向。

### 训练方法

在扩散过程的各噪声水平上训练分类器。具体做法：

```python
def train_noisy_classifier(classifier, dataloader, noise_schedule):
    """
    训练噪声分类器：在不同噪声水平上都能正确分类
    
    与正常分类器训练的唯一区别：输入是加噪图像，且噪声水平 t 作为额外输入
    """
    for x_0, y in dataloader:
        # 1. 随机采样噪声水平
        t = torch.randint(0, T, (x_0.shape[0],))
        
        # 2. 对干净图像加噪
        noise = torch.randn_like(x_0)
        alpha_bar_t = noise_schedule.alpha_bar[t]
        x_t = torch.sqrt(alpha_bar_t)[:, None, None, None] * x_0 + \
              torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * noise
        
        # 3. 分类器前向（输入噪声图像 + 时间步）
        logits = classifier(x_t, t)
        
        # 4. 交叉熵损失
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
```

### 分类器架构

Dhariwal & Nichol 使用与 U-Net 下采样部分类似的架构，包含时间步嵌入：

```
噪声图像 x_t ─→ [ResBlock + t_emb] ─→ [ResBlock + t_emb] ─→ ... ─→ [Global Pool] ─→ [FC] ─→ 1000类
                                                                        8×8×512        512      logits
```

具体使用 U-Net 编码器的前半部分（下采样路径），在 $8 \times 8$ 的特征图上做全局平均池化，然后接全连接分类头。

## ADM（Ablated Diffusion Model）架构

Dhariwal & Nichol 不仅提出了 Classifier Guidance，还对 U-Net 架构做了大量消融实验，提出了 ADM 架构：

| 组件 | 选择 | 依据 |
|------|------|------|
| 注意力分辨率 | 32, 16, 8 | 更多注意力层提升质量 |
| 注意力头维度 | 64 | 优于固定头数 |
| 残差连接 | $\frac{1}{\sqrt{2}}$ 缩放 | 稳定深层网络训练 |
| BigGAN 上下采样 | 是 | 学习的上下采样优于固定 |
| 自适应 GroupNorm | AdaGN (时间+类别) | 优于简单 GroupNorm |
| 模型大小 | 256 通道, 2 ResBlock/级 | ~554M 参数 |

ADM 的完整架构（ImageNet 256x256）：

```
输入: 256×256×3 + 时间步 t + 类别 y
│
├── Encoder (下采样路径):
│   256×256×256 ─→ 128×128×256 ─→ 64×64×512 ─→ 32×32×512
│   [ResBlock×2]    [ResBlock×2     [ResBlock×2     [ResBlock×2
│                    + Attn]         + Attn]         + Attn]
│
├── Bottleneck: 32×32×512 → [ResBlock + Attn + ResBlock]
│
├── Decoder (上采样路径):
│   32×32×512 ─→ 64×64×512 ─→ 128×128×256 ─→ 256×256×256 ─→ 256×256×3
│   [ResBlock×3     [ResBlock×3     [ResBlock×3      [ResBlock×3]
│    + Attn]         + Attn]         + Attn]
│
总参数: ~554M (ADM-G 模型)
```

## 完整采样算法

### 梯度计算伪代码

```python
def classifier_guided_sample(diffusion_model, classifier, y, shape, 
                              guidance_scale=1.0, num_steps=250):
    """
    完整的 Classifier Guidance 采样算法
    
    diffusion_model: 预训练的无条件扩散模型 ε_θ
    classifier: 在噪声图像上训练的分类器 p_φ(y|x_t)
    y: 目标类别（整数）
    guidance_scale: 引导强度 s
    """
    x = torch.randn(shape)  # x_T ~ N(0, I)
    
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        
        # === 1. 扩散模型预测噪声 ===
        with torch.no_grad():
            eps_pred = diffusion_model(x, t_tensor)
        
        # === 2. 计算分类器梯度 ===
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t_tensor)
        log_prob = F.log_softmax(logits, dim=-1)
        selected = log_prob[range(shape[0]), y]  # 选择目标类别的 log prob
        grad = torch.autograd.grad(selected.sum(), x_in)[0]
        
        # === 3. 修改噪声预测（加入引导） ===
        # ε̂ = ε_θ - √(1-ᾱ_t) · s · ∇_x log p(y|x_t)
        eps_guided = eps_pred - torch.sqrt(1 - alpha_bar[t]) * guidance_scale * grad
        
        # === 4. DDPM 采样步 ===
        mu = compute_posterior_mean(x, eps_guided, t)
        if t > 0:
            sigma = torch.sqrt(betas[t])
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    
    return x
```

### 计算开销分析

每步采样需要：
1. 扩散模型前向传播：$O(1)$（~554M 参数）
2. 分类器前向传播 + 反向传播（求梯度）：$O(2)$（相当于两次前向）

因此总计算量约为正常采样的 **3 倍**（1 次扩散 + 1 次分类器前向 + 1 次分类器反向）。

## ImageNet 256x256 完整基准测试

### 不同引导强度的效果

| 引导强度 $s$ | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|-------------|-------|------|------------|---------|
| 0.0（无条件） | 10.94 | 100.98 | 0.69 | 0.63 |
| 0.25 | 8.56 | 131.94 | 0.73 | 0.60 |
| 0.5 | 5.98 | 163.67 | 0.78 | 0.57 |
| 1.0（标准） | 4.59 | 186.70 | 0.82 | 0.52 |
| 2.0 | 5.02 | 199.21 | 0.85 | 0.44 |
| 4.0 | 7.43 | 214.12 | 0.88 | 0.36 |
| 10.0 | 14.82 | 225.81 | 0.90 | 0.24 |

**关键观察**：
- FID 在 $s = 1.0$ 左右最优（FID 同时衡量质量和多样性）
- IS 和 Precision 随 $s$ 单调增加（质量持续提升）
- Recall 随 $s$ 单调下降（多样性持续降低）
- 这揭示了**质量-多样性权衡（Quality-Diversity Tradeoff）**的本质

### 与其他模型的横向对比

| 模型 | 类别 | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|------|------|-------|------|------------|---------|
| BigGAN-deep | GAN | 6.95 | 198.2 | 0.87 | 0.28 |
| StyleGAN-XL | GAN | 2.30 | 265.12 | 0.78 | 0.53 |
| ADM (无引导) | Diffusion | 10.94 | 100.98 | 0.69 | 0.63 |
| ADM + CG ($s=1$) | Diffusion | **4.59** | 186.70 | 0.82 | 0.52 |
| ADM-U + CG | Diffusion | **3.94** | 215.84 | 0.83 | 0.53 |
| CDM | Cascade Diff | 4.88 | 158.71 | — | — |

ADM + Classifier Guidance 在 FID 上首次超越 BigGAN-deep（4.59 vs 6.95），同时在 Recall 上也更优（0.52 vs 0.28），说明扩散模型在多样性方面有天然优势。

### Precision-Recall 曲线解读

```
Precision（质量）
 1.0 ┤                                    ●  s=10
     │                               ●  s=4
     │                          ●  s=2
 0.8 ┤                    ● s=1         ← FID 最优点
     │               ● s=0.5
     │          ● s=0.25
 0.7 ┤     ● s=0 (无条件)
     │
 0.6 ┤
     └──────┬──────┬──────┬──────┬──────
          0.2    0.3    0.4    0.5    0.6    Recall（多样性）

引导强度 s 在 Precision-Recall 空间中画出一条 Pareto 前沿。
s 越大 → 越右上角（高质量低多样性）→ 类似 GAN 的行为。
s=0 时 → 左下角（高多样性低质量）→ 纯扩散模型行为。
```

## 局限性分析

| 局限 | 详细说明 | 后续解决方案 |
|------|---------|------------|
| 需要额外分类器 | 增加训练成本和复杂度 | CFG 消除了分类器依赖 |
| 分类器须支持噪声输入 | 不能复用现有的 ImageNet 分类器 | CFG 无需任何额外模型 |
| 仅支持分类任务 | 难以推广到文本/图像等复杂条件 | CFG 对任意条件通用 |
| 每步额外计算 | 分类器前向 + 反向（约 3x 开销） | CFG 仅 2x 开销 |
| 梯度质量依赖分类器 | 分类器质量差则引导不稳定 | CFG 自包含，无外部依赖 |
| 模式坍缩风险 | 高 $s$ 时生成结果高度集中 | 对所有引导方法都存在 |

这些局限直接催生了 Classifier-Free Guidance 的出现——CFG 用一个巧妙的训练技巧（随机丢弃条件）替代了整个分类器。

## 历史意义

Classifier Guidance 虽然在实践中已被 CFG 取代，但其理论贡献深远：

1. **条件得分分解**：$\nabla \log p(x|y) = \nabla \log p(x) + \nabla \log p(y|x)$ 是所有引导方法的数学基础
2. **质量-多样性权衡**的首次系统研究
3. 证明了扩散模型的潜力不亚于 GAN，改变了生成模型领域的研究方向
4. ADM 架构的消融实验为后续所有 U-Net 设计提供了参考

## 小结

| 概念 | 要点 |
|------|------|
| 核心公式 | $\nabla \log p(x_t\|y) = \nabla \log p(x_t) + s \cdot \nabla \log p(y\|x_t)$ |
| 贝叶斯推导 | 条件得分 = 无条件得分 + 分类器梯度 |
| 引导强度 $s$ | 控制条件强度，$s>1$ 增强但降低多样性 |
| 带噪分类器 | 必须在各噪声水平上训练 |
| ADM 架构 | ~554M 参数 U-Net，自适应 GroupNorm，多分辨率注意力 |
| 历史意义 | 首次让扩散模型超越 GAN（FID 4.59 vs 6.95） |
| 局限 | 需要额外分类器，难以推广到文本条件 |

---

> **下一篇**：[Classifier-Free Guidance](./03-cfg.md) -- 不需要分类器的引导方法，当前主流。

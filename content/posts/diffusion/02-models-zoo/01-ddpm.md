---
title: "DDPM 详解"
date: 2026-04-17T17:46:46.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# DDPM 详解

> ⚙️ 进阶 | 前置知识：模块一全部内容，了解 CNN 和注意力机制基础

## 历史地位

**DDPM（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）**（Ho et al., 2020）是扩散模型从理论走向实用的里程碑。虽然扩散概率模型的概念早在 2015 年（Sohl-Dickstein et al.）就被提出，但 DDPM 首次证明扩散模型可以生成与 GAN 媲美的高质量图像，并且训练过程极为简单稳定。

## U-Net 架构

DDPM 使用 **U-Net** 作为去噪骨干网络。U-Net 最初由 Ronneberger et al.（2015）为医学图像分割提出，其对称的编码器-解码器结构天然适合像素级预测任务。

### 整体结构

```
输入: xₜ (带噪图像) + t (时间步)
                    │
            ┌───────┴───────┐
            │   时间步嵌入    │  ← Sinusoidal + MLP
            └───────┬───────┘
                    │
    ┌───────────────┴───────────────┐
    │         编码器 (下采样)         │
    │  ResBlock → Attn → ↓          │
    │  ResBlock → Attn → ↓          │──→ Skip Connection
    │  ResBlock → Attn → ↓          │
    ├───────────────────────────────┤
    │         瓶颈 (Bottleneck)      │
    │  ResBlock → Attn → ResBlock   │
    ├───────────────────────────────┤
    │         解码器 (上采样)         │
    │  ↑ → Concat Skip → ResBlock → Attn  │
    │  ↑ → Concat Skip → ResBlock → Attn  │
    │  ↑ → Concat Skip → ResBlock → Attn  │
    └───────────────┬───────────────┘
                    │
            ┌───────┴───────┐
            │   输出卷积层    │
            └───────┬───────┘
                    │
输出: εθ (预测噪声，与输入同尺寸)
```

### 核心组件

#### 1. 时间步嵌入（Timestep Embedding）

网络需要知道当前在第几步——因为不同噪声水平下的去噪策略应该不同。

DDPM 借鉴了 Transformer 的**正弦位置编码（Sinusoidal Positional Encoding）**：

$$\text{emb}(t)_{2i} = \sin(t / 10000^{2i/d}), \quad \text{emb}(t)_{2i+1} = \cos(t / 10000^{2i/d})$$

然后通过两层 MLP（全连接层 + SiLU 激活 + 全连接层）映射到每层 ResBlock 所需的维度。

```python
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, 4 * dim),
        )
    
    def forward(self, t):
        emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(emb)
```

#### 2. 残差块（ResBlock）

每个 ResBlock 包含两组 "GroupNorm → SiLU → Conv" 的序列，中间注入时间步嵌入，加上跳跃连接（如果维度不同则用 1×1 卷积调整）：

```python
class ResBlock(nn.Module):
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间步信息（加法或 scale-shift）
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)  # 残差连接
```

**GroupNorm（组归一化）** 取代了 BatchNorm，因为扩散模型训练中 batch size 通常较小，GroupNorm 更稳定。

**SiLU（Sigmoid Linear Unit）**，也叫 Swish，$\text{SiLU}(x) = x \cdot \sigma(x)$，在扩散模型中比 ReLU 表现更好。

#### 3. 自注意力层（Self-Attention）

在较低分辨率的特征图（如 16×16 和 8×8）上使用**多头自注意力（Multi-Head Self-Attention）**，使模型能捕捉全局结构和长距离依赖。

高分辨率特征图（如 64×64）通常不使用注意力——计算量太大（注意力的计算复杂度与序列长度的平方成正比）。

#### 4. 下采样与上采样

- **下采样**：步长为 2 的卷积（或平均池化 + 卷积）
- **上采样**：最近邻插值 + 卷积（避免棋盘格伪影）
- **跳跃连接**：编码器每层的输出与解码器对应层的输入拼接（channel-wise concatenation）

## 完整训练算法

```python
# DDPM 训练（完整版）
def train(model, dataloader, optimizer, alpha_bars, T, epochs):
    for epoch in range(epochs):
        for x_0 in dataloader:  # x_0: 干净图像 [B, C, H, W]
            # 1. 随机时间步
            t = torch.randint(0, T, (x_0.shape[0],))
            
            # 2. 采样噪声
            epsilon = torch.randn_like(x_0)
            
            # 3. 前向过程（封闭解）
            a_bar = alpha_bars[t].reshape(-1, 1, 1, 1)
            x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * epsilon
            
            # 4. 预测噪声
            epsilon_pred = model(x_t, t)
            
            # 5. 简化损失
            loss = F.mse_loss(epsilon_pred, epsilon)
            
            # 6. 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 训练细节

| 超参数 | DDPM 原始设置 | 说明 |
|--------|-------------|------|
| $T$ | 1000 | 总时间步数 |
| $\beta$ 调度 | 线性，$10^{-4}$ 到 $0.02$ | 噪声系数 |
| 优化器 | Adam, lr=$2 \times 10^{-4}$ | — |
| EMA | 0.9999 | 指数移动平均 |
| Batch size | 128 | — |
| 分辨率 | 256×256 | LSUN, CelebA-HQ |

**EMA（Exponential Moving Average，指数移动平均）**：对模型参数维护一个平滑版本 $\theta_{\text{ema}} \leftarrow \lambda \theta_{\text{ema}} + (1-\lambda)\theta$，采样时使用 EMA 参数，能显著提升生成质量。

## 完整采样算法

```python
@torch.no_grad()
def sample(model, shape, T, betas, alphas, alpha_bars):
    x = torch.randn(shape)  # 纯噪声
    
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long)
        eps = model(x, t_batch)
        
        # 后验均值
        mu = (1 / torch.sqrt(alphas[t])) * (
            x - betas[t] / torch.sqrt(1 - alpha_bars[t]) * eps
        )
        
        if t > 0:
            sigma = torch.sqrt(betas[t])
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    
    return x
```

## 与 GAN、VAE 的深度对比

### 生成质量

| 模型 | FID (CIFAR-10) | FID (LSUN Bedroom 256) | 备注 |
|------|---------------|----------------------|------|
| DDPM | 3.17 | 4.89 | 2020 年 |
| StyleGAN2 | 2.92 | 2.65 | 需要精细调参 |
| VAE | ~80+ | — | 模糊 |

DDPM 首次在 FID 上接近 GAN，且无需对抗训练的复杂调参。

### 训练稳定性

- **GAN**：判别器和生成器的平衡极为敏感；模式崩塌、训练震荡是常见问题；不同数据集需要不同的训练技巧
- **DDPM**：MSE 损失 + 标准 Adam 优化器，几乎不需要调参；Loss 曲线平滑下降；在不同数据集上表现一致

### 多样性

- **GAN**：容易出现模式崩塌（生成器只学会生成少数几种样本）
- **DDPM**：理论上覆盖整个数据分布，多样性显著更好
- DDPM 在 Recall（多样性度量）上通常远超 GAN，在 Precision（质量度量）上相当

### 可控性

- **GAN**：条件生成需要额外设计（如 Conditional GAN、StyleGAN 的隐空间编辑）
- **DDPM**：天然支持 Classifier Guidance 和后来的 CFG，条件控制更灵活

### 速度

这是 DDPM 最大的劣势：

| 模型 | 生成一张 256×256 图像 |
|------|---------------------|
| GAN | ~0.05 秒（1 次前向传播） |
| DDPM | ~20 秒（1000 次前向传播） |

这 400 倍的速度差距催生了 DDIM、DPM-Solver、蒸馏等大量加速研究。

## DDPM 的改进：Improved DDPM

Nichol & Dhariwal（2021）在原始 DDPM 基础上做了几个重要改进：

1. **余弦噪声调度**：替代线性调度，改善中间 SNR 覆盖
2. **可学习方差**：网络额外输出方差参数 $v$，在 $\beta_t$ 和 $\tilde{\beta}_t$ 间插值
3. **混合损失**：$L_{\text{simple}} + \lambda L_{\text{VLB}}$，兼顾样本质量和似然
4. **降低步数**：通过改进的调度和方差学习，50-100 步即可获得不错的质量

这些改进将 DDPM 在 ImageNet 256×256 上的对数似然从 3.75 bits/dim 提升到 2.94 bits/dim，首次在似然指标上超越自回归模型。

## 小结

| 概念 | 要点 |
|------|------|
| U-Net | 编码器-瓶颈-解码器 + 跳跃连接，像素级预测 |
| 时间步嵌入 | 正弦编码 + MLP，注入 ResBlock |
| ResBlock | GroupNorm + SiLU + Conv + 时间步注入 + 残差 |
| 训练 | 随机 $t$ + 加噪 + 预测噪声 + MSE，极其简单 |
| 优势 | 训练稳定、多样性好、条件控制灵活 |
| 劣势 | 采样慢（~1000 步） |

---

> **下一篇**：[DDIM 详解](./02-ddim) — 如何将采样步数从 1000 步降到 50 步，以及确定性采样的奥秘。

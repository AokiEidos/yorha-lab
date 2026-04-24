---
title: "ControlNet"
date: 2026-04-20T17:21:13.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# ControlNet

> ⚙️ 进阶 | 前置知识：[Latent Diffusion 与 Stable Diffusion](../02-models-zoo/04-latent-diffusion)，[CFG](./03-cfg)

## 解决什么问题

文本描述擅长表达"画什么"，但不擅长表达"怎么画"——精确的空间结构（如人物姿态、建筑轮廓、深度关系）很难用自然语言准确传达。**ControlNet**（Zhang et al., 2023）让用户可以用边缘图、深度图、骨骼图等视觉条件精确控制生成图像的空间结构。

ControlNet 是扩散模型社区影响力最大的条件控制方法之一——它让"精确可控生成"从研究概念变成了实用工具。

## 架构设计

### 核心设计原则

ControlNet 的设计遵循一个关键原则：**在不破坏预训练 Stable Diffusion 权重的前提下，添加空间条件控制能力。**

这个原则的重要性在于：SD 的权重蕴含了大量关于图像生成的知识（在数十亿图像上训练），如果训练过程破坏了这些知识，就需要重新训练整个模型——代价不可接受。

### 零卷积（Zero Convolution）的数学分析

**零卷积** 是权重和偏置都初始化为零的 1x1 卷积层。

$$\text{ZeroConv}(x) = W \cdot x + b, \quad W = \mathbf{0}, \quad b = \mathbf{0}$$

**为什么零初始化是关键？**

考虑训练开始时的梯度流：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}$$

即使 $W = 0$，梯度 $\frac{\partial L}{\partial W}$ 不为零（只要 $x \neq 0$ 且 $\frac{\partial L}{\partial y} \neq 0$）。所以零卷积可以正常学习，只是初始输出为零。

**与其他初始化的对比**：

| 初始化方式 | 初始输出 | 对预训练模型的影响 | 训练稳定性 |
|-----------|---------|------------------|-----------|
| 随机初始化 | 随机噪声 | 严重破坏（随机干扰） | 差 |
| 小值初始化 | 微小扰动 | 轻微破坏 | 中等 |
| **零初始化** | **恒为零** | **零影响（完美保护）** | **最好** |
| 恒等初始化 | 全量传递 | 完全覆盖（信息冲突） | 差 |

```python
class ZeroConv(nn.Module):
    """零卷积：权重和偏置初始化为零的 1x1 卷积"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        nn.init.zeros_(self.conv.weight)    # 关键：零初始化
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)
```

**渐进式注入过程**（可视化）：

```
训练进度:  0%        25%        50%        75%        100%
零卷积输出: |0.00|   |0.01|    |0.08|    |0.35|    |1.00|
控制强度:   无影响    微弱感知    明显控制    强控制    完全控制

→ 控制信号从无到有渐进式注入，预训练知识平滑过渡到受控生成
```

### 完整架构与特征维度

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ControlNet 完整架构                                │
│                                                                      │
│  文本 → CLIP Text Encoder → [77, 768] → Cross-Attention              │
│                                                                      │
│  条件图 (H×W×3)                                                      │
│      ↓                                                               │
│  [Conv 3×3, 16] → [Conv 3×3, 32] → [Conv 3×3, 64] → [Conv 3×3, 128]│
│  (4层卷积将条件图编码为 H/8 × W/8 × 128 的特征图)                     │
│      ↓                                                               │
│  ┌──────── 控制分支 (可训练) ──────┐  ┌──── 主 U-Net (冻结) ────┐     │
│  │                                │  │                          │     │
│  │ Enc1: [320ch, H/8] ──ZeroConv──(+)─→ Enc1: [320ch, H/8]     │     │
│  │ Enc2: [320ch, H/8] ──ZeroConv──(+)─→ Enc2: [320ch, H/8]     │     │
│  │ Enc3: [640ch, H/16]──ZeroConv──(+)─→ Enc3: [640ch, H/16]    │     │
│  │ Enc4: [640ch, H/16]──ZeroConv──(+)─→ Enc4: [640ch, H/16]    │     │
│  │ Enc5: [1280ch,H/32]──ZeroConv──(+)─→ Enc5: [1280ch, H/32]   │     │
│  │ Enc6: [1280ch,H/32]──ZeroConv──(+)─→ Enc6: [1280ch, H/32]   │     │
│  │ Enc7: [1280ch,H/64]──ZeroConv──(+)─→ Enc7: [1280ch, H/64]   │     │
│  │ Enc8: [1280ch,H/64]──ZeroConv──(+)─→ Enc8: [1280ch, H/64]   │     │
│  │                                │  │                          │     │
│  │ Mid:  [1280ch,H/64]──ZeroConv──(+)─→ Mid: [1280ch, H/64]    │     │
│  │                                │  │                          │     │
│  └────────────────────────────────┘  │  Decoder (上采样)...      │     │
│                                       │  → 输出 [4ch, H/8]       │     │
│  控制分支参数: ~361M                   │  主 U-Net 参数: ~860M     │     │
│  (= SD 1.5 编码器 + 中间层的完整副本)  └──────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### 为什么复制编码器而非设计新架构？

1. **特征对齐**：复制的编码器与主网络共享相同的特征空间，零卷积只需学习"加多少"，不需要学习"特征变换"
2. **可用预训练初始化**：控制分支可以从 SD 权重热启动，大幅加速收敛
3. **工程简洁**：不需要设计新的模块，直接复制 + 零卷积连接

## 支持的条件类型与预处理

| 条件类型 | 预处理工具 | 输入要求 | 典型分辨率 | 适用场景 |
|---------|-----------|---------|-----------|---------|
| Canny 边缘 | OpenCV Canny | RGB 图像 | 512x512 | 保持物体轮廓 |
| HED 边缘 | HED 网络 | RGB 图像 | 512x512 | 更自然的柔和边缘 |
| 深度图 | MiDaS v3.1 | RGB 图像 | 512x512 | 3D 空间关系 |
| 法线图 | MiDaS + BAE | RGB 图像 | 512x512 | 表面朝向细节 |
| 人体姿态 | OpenPose | RGB 图像 | 512x512 | 人物姿态控制 |
| 语义分割 | ADE20K / COCO | RGB 图像 | 512x512 | 区域语义控制 |
| 涂鸦 | 手绘输入 | 用户草图 | 任意 | 草图上色 |
| M-LSD 直线 | M-LSD 检测器 | RGB 图像 | 512x512 | 建筑直线结构 |

### 预处理流水线示例

```python
# Canny 边缘检测预处理
import cv2
def canny_preprocess(image, low_threshold=100, high_threshold=200):
    """标准 Canny 边缘预处理"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = edges[:, :, None]                    # [H, W, 1]
    edges = np.concatenate([edges] * 3, axis=2)  # [H, W, 3]
    return edges  # 黑白边缘图

# 深度图估计预处理
def depth_preprocess(image):
    """MiDaS 深度估计预处理"""
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    input_batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth = midas(input_batch)
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # 归一化到 [0,1]
    return depth

# OpenPose 姿态检测预处理
def pose_preprocess(image):
    """OpenPose 骨骼点检测"""
    from controlnet_aux import OpenposeDetector
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose = openpose(image)  # 返回骨骼可视化图
    return pose
```

## 训练详细配方

### 数据准备

训练数据是 (原始图像, 条件图, 文本描述) 三元组：

```
训练样本:
├── image.jpg          # 原始图像
├── condition.jpg      # 条件图（Canny/Depth/Pose 等，自动提取）
└── prompt.txt         # 文本描述（从 BLIP/BLIP-2 自动生成或人工标注）
```

### 训练超参数（基于 SD 1.5）

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| 学习率 | 1e-5 | 控制分支的学习率 |
| 优化器 | AdamW | $\beta_1=0.9, \beta_2=0.999$ |
| Batch size | 4-8 (per GPU) | 总 batch 通常 256-512 |
| 训练步数 | 50k-200k | 视数据量和条件类型 |
| 分辨率 | 512x512 | 与 SD 1.5 一致 |
| 混合精度 | FP16 | 节省显存 |
| GPU | 8x A100 40GB | 约 3-5 天完成训练 |
| 数据量 | 100k-3M | Canny: ~3M, Pose: ~150k |

```python
def controlnet_train_step(controlnet, sd_model, x_0, condition_img, text, t):
    """ControlNet 训练步骤"""
    # 1. 编码到潜在空间
    z_0 = sd_model.vae.encode(x_0).latent_dist.sample() * 0.18215
    
    # 2. 前向扩散
    eps = torch.randn_like(z_0)
    z_t = noise_schedule.add_noise(z_0, eps, t)
    
    # 3. 文本编码
    text_emb = sd_model.text_encoder(text)   # [B, 77, 768]
    
    # 4. 控制分支提取特征
    control_features = controlnet(z_t, t, condition_img, text_emb)
    # control_features: 每层一个 tensor, 经过 ZeroConv
    
    # 5. 主网络使用控制特征
    eps_pred = sd_model.unet(z_t, t, text_emb, control_features=control_features)
    
    # 6. 损失（只更新 controlnet 参数）
    loss = F.mse_loss(eps_pred, eps)
    loss.backward()
    # sd_model 的梯度不计算（冻结）
    controlnet_optimizer.step()
```

## Multi-ControlNet 权重平衡

### 基本组合

可以同时使用多个 ControlNet，各自的控制特征加权求和：

$$\text{features}_i = w_1 \cdot \text{ControlNet}_1(\text{edge})_i + w_2 \cdot \text{ControlNet}_2(\text{depth})_i$$

其中 $i$ 是 U-Net 的第 $i$ 层。

### 权重调整策略

```python
def multi_controlnet_inference(controlnets, conditions, weights, sd_model, z_t, t, text_emb):
    """
    Multi-ControlNet 推理
    
    controlnets: [cn_edge, cn_depth, cn_pose, ...]
    conditions: [edge_img, depth_img, pose_img, ...]
    weights: [0.7, 0.5, 0.8, ...]  各 ControlNet 的权重
    """
    all_features = [None] * num_layers
    
    for cn, cond, w in zip(controlnets, conditions, weights):
        features = cn(z_t, t, cond, text_emb)
        for i, feat in enumerate(features):
            if all_features[i] is None:
                all_features[i] = w * feat
            else:
                all_features[i] += w * feat
    
    eps_pred = sd_model.unet(z_t, t, text_emb, control_features=all_features)
    return eps_pred
```

**权重调优经验**：

| 组合 | 边缘权重 | 深度权重 | 效果 |
|------|---------|---------|------|
| 边缘 + 深度 | 0.7 | 0.5 | 轮廓清晰 + 空间感好 |
| 边缘 + 姿态 | 0.5 | 0.8 | 人物姿态准确 + 环境轮廓 |
| 深度 + 姿态 | 0.6 | 0.8 | 3D 空间 + 人物控制 |
| 三者全用 | 0.4 | 0.4 | 0.6 | 全方位但需仔细调权 |

**原则**：权重之和不宜超过 1.5，否则控制信号过强会压制文本条件的创造力。

## ControlNet-XS 轻量变体

**ControlNet-XS**（Zavadski et al., 2023）是 ControlNet 的轻量版本：

| 方面 | ControlNet | ControlNet-XS |
|------|-----------|---------------|
| 额外参数 | ~361M | ~55M（减少 85%） |
| 结构 | 完整编码器副本 | 小型编码器 + 跨层连接 |
| 训练显存 | ~40GB | ~16GB |
| 控制精度 | 最高 | 略低（~95%） |
| 推理速度 | +40% 开销 | +10% 开销 |

ControlNet-XS 不复制完整编码器，而是使用一个更小的控制网络，通过跨层注意力与主网络交互。适合资源受限场景。

## 小结

| 概念 | 要点 |
|------|------|
| 核心设计 | 复制编码器作为控制分支 + 零卷积连接 |
| 零卷积原理 | 零初始化保护预训练权重，梯度非零所以能学习 |
| 参数量 | 控制分支 ~361M（SD 1.5 编码器的副本） |
| 训练 | 只训练控制分支，冻结主网络；lr=1e-5，50k-200k 步 |
| 预处理 | Canny/MiDaS/OpenPose 等工具自动提取条件 |
| Multi-ControlNet | 多个控制分支特征加权组合，权重和不超过 1.5 |
| ControlNet-XS | 轻量变体，参数减少 85%，控制精度约 95% |
| 生态 | 社区最丰富的控制方法，数十种条件类型 |

---

> **下一篇**：[IP-Adapter](./05-ip-adapter)

---
title: "推理优化工程实践"
date: 2026-04-20T17:29:49.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 推理优化工程实践

> ⚙️ 进阶 | 前置知识：了解 GPU 推理基础，[采样加速概述](./01-overview.md)

## 优化维度总览 🔰

推理优化从四个维度入手：**计算优化**（减少 FLOPs）、**内存优化**（降低带宽瓶颈）、**IO 优化**（减少数据传输）、**编排优化**（流水线并行）。这些优化独立于算法层面叠加使用。

```
            推理优化技术栈
               │
    ┌──────────┼──────────┐──────────┐
    │          │          │          │
  计算优化    内存优化    IO优化     编排优化
    │          │          │          │
  FlashAttn  DeepCache  CPU Offload CUDA Graph
  xFormers   VAE Tiling  模型并行   Batch调度
  ToMe       Attn Slice  梯度检查点  异步推理
  TensorRT   KV缓存     管道并行    动态批处理
  torch.compile
```

## Flash Attention：分块计算的高效注意力 ⚙️

### 标准注意力的内存瓶颈

**标准注意力（Standard Attention）** 的计算过程：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对序列长度 $N$、头维度 $d$，标准实现需要：
1. 计算 $S = QK^T/\sqrt{d}$：$O(N^2 d)$ FLOPs，生成 $N \times N$ 矩阵
2. 计算 $P = \text{softmax}(S)$：读写 $N \times N$ 矩阵
3. 计算 $O = PV$：$O(N^2 d)$ FLOPs

**关键瓶颈**：完整的 $N \times N$ 注意力矩阵必须写入 GPU 的 **HBM（High Bandwidth Memory，高带宽显存）**，然后再读回来做 softmax。这是**内存带宽受限（Memory-Bound）** 的操作，而非计算受限。

在 Stable Diffusion 中，64x64 latent 的 Self-Attention 序列长度 $N = 4096$，注意力矩阵大小为 $4096^2 \times 2 = 32$ MB (FP16)——对于 8 个头来说是 256 MB。

### Flash Attention 的分块策略

**Flash Attention（Dao et al., 2022）** 的核心思想：将 Q, K, V 分块加载到 GPU 的 **SRAM（片上高速缓存，约 20MB）** 中计算，避免将完整注意力矩阵写入 HBM。

核心算法（概念伪代码）：

```python
def flash_attention(Q, K, V, block_size=256):
    """Flash Attention: 分块计算避免将 N×N 矩阵写入 HBM"""
    N, d = Q.shape
    O, l, m = torch.zeros_like(Q), torch.zeros(N), torch.full((N,), -float('inf'))
    
    for j in range(0, N, block_size):          # 外循环: K,V 块
        K_j, V_j = K[j:j+block_size], V[j:j+block_size]  # HBM → SRAM
        for i in range(0, N, block_size):      # 内循环: Q 块
            S_ij = Q[i:i+block_size] @ K_j.T / math.sqrt(d)  # SRAM 内计算
            # 在线 softmax 更新 + 累积输出（省略数值稳定细节）
            m_new = torch.max(m[i:i+block_size], S_ij.max(dim=1).values)
            P_ij = torch.exp(S_ij - m_new.unsqueeze(1))
            # 增量更新 O, l, m（无需存储完整 P 矩阵）
    return O
```

### 复杂度对比

| 指标 | 标准注意力 | Flash Attention | Flash Attention 2 |
|------|-----------|----------------|-------------------|
| 计算 FLOPs | $O(N^2 d)$ | $O(N^2 d)$ (相同) | $O(N^2 d)$ (相同) |
| HBM 读写 | $O(N^2 + Nd)$ | $O(N^2 d^2 / M)$ | $O(N^2 d^2 / M)$ (更优常数) |
| 峰值显存 | $O(N^2)$ | $O(N)$ | $O(N)$ |
| 实际加速 (SD) | 1× | 2-3× | 3-4× |

其中 $M$ 是 SRAM 大小。当 $M \gg d^2$ 时，HBM 访问量趋近 $O(Nd)$，即线性。

### 在 Stable Diffusion 中的效果

| 模型 | 分辨率 | 无 FA 延迟 | FA2 延迟 | 加速比 | 注意力占比 |
|------|--------|-----------|---------|--------|-----------|
| SD 1.5 | 512x512 | 1.20s | 0.85s | 1.41× | ~35% |
| SD 1.5 | 768x768 | 3.10s | 1.85s | 1.68× | ~45% |
| SDXL | 1024x1024 | 6.50s | 3.80s | 1.71× | ~50% |
| SD 3.0 (DiT) | 1024x1024 | 8.20s | 4.10s | 2.00× | ~65% |

> 分辨率越高、注意力层越多（DiT 架构），Flash Attention 的收益越大。

## xFormers：Memory-Efficient Attention ⚙️

**xFormers** 是 Meta 开发的高效 Transformer 库，提供了 **memory_efficient_attention** 算子，原理与 Flash Attention 类似但实现不同：

- 支持更多注意力变体（带 bias、causal mask 等）
- 在某些硬件上（如较老的 GPU）性能可能优于 Flash Attention
- 集成在 diffusers 库中，使用简单

```python
# 在 diffusers 中启用 xFormers
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.enable_xformers_memory_efficient_attention()  # 一行启用

# 或在最新版本中使用原生 SDPA
pipe.enable_attention_slicing()  # 用于低显存场景
```

**PyTorch 2.0+** 的 SDPA 已集成 Flash Attention 和 memory-efficient attention，大多数情况下不再需要手动启用 xFormers。

## Token Merging (ToMe)：序列压缩 ⚙️

### 算法原理

**Token Merging（ToMe，Token 合并）**（Bolya & Hoffman, 2023）的核心思想：在注意力计算前，合并相似的 token 以缩短序列长度。

算法步骤：(1) 将 token 交替分为源/目标两组，(2) 计算源-目标余弦相似度，(3) 二分图匹配找到最相似的对，(4) 合并（加权平均）相似度最高的 $r \times N$ 对。合并后序列长度变为 $N(1-r)$，注意力计算量降低为 $(1-r)^2$ 倍。

### 在 Stable Diffusion 中的效果

| ToMe 合并比例 | 序列长度变化 | 延迟加速 | FID 变化 | CLIP Score 变化 |
|-------------|------------|---------|---------|----------------|
| 0% (无合并) | 4096 → 4096 | 1× | 基准 | 基准 |
| 20% | 4096 → 3277 | 1.15× | +0.1 | -0.001 |
| 40% | 4096 → 2458 | 1.35× | +0.3 | -0.003 |
| 50% | 4096 → 2048 | 1.50× | +0.5 | -0.005 |
| 60% | 4096 → 1638 | 1.65× | +1.2 | -0.010 |

推荐 30-50% 的合并比例。

## DeepCache：时间步特征缓存 ⚙️

### 核心观察

**DeepCache（Ma et al., 2024）** 基于一个关键观察：U-Net 在相邻时间步的**中间特征（Intermediate Features）** 高度相似。特别是浅层（低分辨率）的特征变化最小。

量化分析：深层（Mid Block, 8x8）相邻步余弦相似度高达 0.995，而浅层（Up Block 1, 64x64）仅 0.93。越深层变化越小，缓存收益越大。

### 缓存策略

DeepCache 的策略：每 $N$ 步做一次完整的 U-Net 前向传播，中间步只计算浅层（变化大的部分），深层特征直接复用缓存。

```
时间步:   t=20   t=19   t=18   t=17   t=16   t=15   ...
           │      │      │      │      │      │
完整计算:  ████   ──     ──     ████   ──     ──     ...
           ████   ██     ██     ████   ██     ██
           ████   cached cached ████   cached cached
           ████   cached cached ████   cached cached
           ████   ██     ██     ████   ██     ██
           ████   ──     ──     ████   ──     ──

           █ = 计算   ── = 跳过   cached = 复用缓存
```

```python
def deepcache_sampling(model, x_T, timesteps, cache_interval=3, cache_depth=2):
    """DeepCache: 每 N 步完整计算，中间步复用深层缓存"""
    x, cached = x_T, None
    for i, t in enumerate(timesteps):
        if i % cache_interval == 0:  # 完整前向，更新缓存
            output, feats = model.forward_with_cache_update(x, t)
            cached = {l: f.clone() for l, f in feats.items() if layer_depth(l) >= cache_depth}
        else:  # 部分前向，深层用缓存
            output = model.forward_with_cache_reuse(x, t, cached, cache_depth)
        x = scheduler.step(output, t, x)
    return x
```

### 缓存间隔分析

| 缓存间隔 $N$ | 完整计算比例 | 理论加速 | 实际加速 | FID 变化 (SD 1.5) |
|-------------|------------|---------|---------|------------------|
| 1 (无缓存) | 100% | 1× | 1× | 基准 |
| 2 | 50% | ~1.6× | 1.5× | +0.1 |
| 3 | 33% | ~2.0× | 1.8× | +0.2 |
| 4 | 25% | ~2.3× | 2.0× | +0.5 |
| 5 | 20% | ~2.5× | 2.1× | +1.0 |

推荐 $N=3$，在加速和质量之间平衡。注意实际加速低于理论值，因为浅层计算仍然需要执行。

## 显存优化技术 ⚙️

### VAE Tiling：超高分辨率支持

VAE 编解码器在高分辨率时显存占用急剧增加（$O(N^2)$ 的注意力）。**VAE Tiling** 将大图切块分别编解码：

原理：将大图切成 512x512 块（潜空间 64x64），每块独立 VAE 解码，重叠区域用线性插值混合消除接缝。这样即使 2048x2048 图像也只需 512x512 的 VAE 显存。

### CPU Offload 策略

| 策略 | 原理 | GPU 显存 | 速度影响 | 适用场景 |
|------|------|---------|---------|---------|
| 模型级 Offload | UNet/VAE/CLIP 轮流加载到 GPU | ~3 GB | 慢 30-50% | 4 GB 显卡 |
| 顺序级 Offload | 每层用完立即转移到 CPU | ~1.5 GB | 慢 200-300% | 2 GB 显卡 |
| 注意力 Slicing | 分片计算注意力 | 峰值降 30-50% | 慢 10-20% | 中等显存 |

```python
# diffusers 中的 CPU Offload 使用
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# 模型级 offload（推荐）
pipe.enable_model_cpu_offload()
# 或顺序级 offload（极致省显存）
pipe.enable_sequential_cpu_offload()

# 搭配使用
pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")
```

## TensorRT 编译加速 ⚙️

### 编译流水线

```python
import torch

# 方法 1: torch.compile + TensorRT 后端 (PyTorch 2.0+)
pipe.unet = torch.compile(pipe.unet, backend="torch_tensorrt",
    options={"precision": torch.float16, "min_block_size": 1})

# 方法 2: CUDA Graph（消除 kernel launch 开销）
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")  # 自动 CUDA Graph

# 方法 3: ONNX → TRT 离线编译（见 04-quantization.md）
# 预热（首次推理触发编译）
_ = pipe("warmup", num_inference_steps=1)
```

### 编译加速效果

| 编译方式 | 首次编译耗时 | 单步加速 | 端到端加速 (20步) | 兼容性 |
|---------|------------|---------|----------------|--------|
| 无编译 (eager) | — | 1× | 1× | 完美 |
| torch.compile (inductor) | ~2min | 1.3-1.5× | 1.3-1.5× | 好 |
| torch.compile + CUDA Graph | ~3min | 1.5-2.0× | 1.5-2.0× | 中 |
| TensorRT FP16 | ~10min | 2.0-3.0× | 2.0-3.0× | 有限 |
| TensorRT INT8 | ~15min | 2.5-4.0× | 2.5-4.0× | 有限 |

## 端到端优化方案 🔬

### 方案一：快速生成（延迟优先，目标 <100ms）

配置：LCM 4步 + TensorRT FP16 + Flash Attention 2 + CUDA Graph

| 硬件 | 延迟 (512x512) | 瓶颈组件 |
|------|----------------|---------|
| A100 | ~60ms | UNet 48ms + VAE 8ms |
| RTX 4090 | ~95ms | UNet 75ms + VAE 12ms |
| RTX 3060 | ~280ms | UNet 220ms + VAE 40ms |

### 方案二：高质量生成（质量优先，目标 <1s）

配置：DPM-Solver++ 20步 + CFG 7.5 + FA2 + DeepCache(N=3) + torch.compile

| 优化叠加 | A100 | RTX 4090 |
|---------|------|----------|
| FP16 基线 | 1.20s | 1.60s |
| + FA2 + DeepCache + compile + ToMe | 0.38s | 0.52s |

### 方案三：低显存（4GB 显存运行 SD 1.5）

配置：DDIM 30步 + Model CPU Offload + Attention Slicing + FP16。峰值显存 ~3.2GB（延迟 8-25s）。

## 实测基准数据 🔬

### SD 1.5 (512x512, 20步 DPM-Solver++, batch=1)

| 优化组合 | A100 80GB | RTX 4090 | RTX 3060 | Jetson Orin |
|---------|----------|----------|----------|-------------|
| FP16 基线 | 1.20s | 1.60s | 4.80s | 8.0s |
| + Flash Attention 2 | 0.85s | 1.15s | 3.50s | 6.5s |
| + torch.compile | 0.65s | 0.90s | 2.80s | — |
| + TensorRT FP16 | 0.48s | 0.70s | 2.20s | 4.0s |
| + TensorRT INT8 | 0.35s | 0.50s | 1.70s | 3.0s |
| + DeepCache (N=3) | 0.25s | 0.35s | 1.20s | 2.1s |

### SDXL (1024x1024, 20步) 与 LCM 快速模式

| 配置 | A100 | RTX 4090 |
|------|------|----------|
| SDXL FP16 基线 | 6.50s | 8.80s |
| SDXL + FA2 + TRT + DeepCache + ToMe | 1.30s | 1.90s |
| SD 1.5 LCM 4步 + TRT FP16 | 60ms | 95ms |
| SD 1.5 LCM 4步 + TRT INT8 | 40ms | 65ms |
| SDXL LCM 4步 + TRT FP16 | 280ms | 400ms |

## 优化方案选择指南 ⚙️

| 首要约束 | 推荐方案 |
|---------|---------|
| 延迟 < 100ms | LCM 4步 + TRT + INT8（需 A100/H100） |
| 延迟 < 1s | DPM-Solver++ 20步 + FA2 + DeepCache + compile |
| 显存 < 4GB | SD 1.5 + model offload + attention slicing |
| 显存 < 8GB | SD 1.5 任意方案；SDXL 需 CPU offload |
| 吞吐量优先 | 大 batch + TRT + FP16/INT8 + 动态批处理 |

## 加速方案综合对比表

| 方法 | 类型 | 步数减少 | 每步加速 | 额外训练 | 质量损失 | 可组合 | 实现难度 |
|------|------|---------|---------|---------|---------|--------|---------|
| DPM-Solver++ | 求解器 | 50→20 | — | 否 | 极小 | 高 | 低 |
| LCM | 蒸馏 | 50→4 | — | 是 | 小 | 高 | 中 |
| SDXL-Turbo | 蒸馏 | 50→1 | — | 是 | 中 | 有限 | 高 |
| TensorRT | 编译 | — | 2-3× | 否 | 无 | 高 | 中 |
| torch.compile | 编译 | — | 1.3-1.5× | 否 | 无 | 高 | 低 |
| Flash Attention 2 | 算子 | — | 注意力 2-4× | 否 | 无 | 高 | 低 |
| DeepCache | 缓存 | — | ~1.8× | 否 | 极小 | 中 | 低 |
| ToMe (30%) | 压缩 | — | ~1.3× | 否 | 极小 | 高 | 低 |
| INT8 量化 | 量化 | — | 1.3-1.5× | 否/是 | 小 | 高 | 中 |
| CPU Offload | 内存 | — | 0.5-0.7× (更慢) | 否 | 无 | 高 | 低 |

## 小结

| 要点 | 说明 |
|------|------|
| 最大收益方法 | Flash Attention 2（无成本，大幅加速） |
| 编译优化 | TensorRT > torch.compile > eager |
| 缓存策略 | DeepCache N=3 是质量-速度的甜蜜点 |
| 低显存方案 | Model Offload + Attention Slicing 可在 4GB 运行 |
| 推荐起步 | FP16 + Flash Attention 2 + torch.compile（零训练成本） |
| 生产部署 | LCM + TensorRT + FP16 是当前最佳实践 |

---
> **下一篇**：[文生图技术栈](../05-llm-era/01-text-to-image.md) — 进入模块五，探索 LLM 时代的扩散模型应用。

---
title: "模型量化"
date: 2026-04-20T17:26:50.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# 模型量化

> ⚙️ 进阶 | 前置知识：了解基本的模型推理概念，[采样加速概述](./01-overview.md)

## 量化基础 🔰

**量化（Quantization）** 是将模型权重和/或激活值从高精度浮点数（FP32/FP16）压缩到低精度整数（INT8/INT4）的技术。目标是减少内存占用、降低带宽需求、并利用整数运算单元加速推理。

### 量化公式

**均匀量化（Uniform Quantization）** 的基本公式：

$$Q(x) = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z, \; 0, \; 2^b - 1\right)$$

$$\hat{x} = s \cdot (Q(x) - z)$$

其中：
- $s$（scale）是缩放因子：$s = \frac{x_{\max} - x_{\min}}{2^b - 1}$
- $z$（zero point）是零点偏移
- $b$ 是位宽（8 for INT8, 4 for INT4）
- $\lfloor \cdot \rceil$ 是四舍五入

### 精度层级

| 精度 | 位宽 | 模型大小 (SD 1.5 UNet) | 内存带宽需求 | 计算吞吐量 |
|------|------|----------------------|------------|-----------|
| FP32 | 32 bit | 3.4 GB | 1× | 1× |
| FP16/BF16 | 16 bit | 1.7 GB | 2× | 2× |
| INT8 | 8 bit | 0.85 GB | 4× | 4× (INT8 tensor core) |
| INT4 | 4 bit | 0.43 GB | 8× | 理论 8×（实际受限） |
| NF4 | 4 bit | 0.43 GB | 8× | 同上，更好的精度分布 |

> **NF4（NormalFloat4）** 是 QLoRA 提出的 4 bit 格式，量化级别按正态分布间隔划分，更适合权重的实际分布。

### 量化方式分类

| 缩写 | 含义 | 说明 | 难度 |
|------|------|------|------|
| W8A16 | 权重 INT8，激活 FP16 | 仅压缩权重，不加速计算 | 简单 |
| W8A8 | 权重和激活都 INT8 | 可利用 INT8 tensor core | 中等 |
| W4A16 | 权重 INT4，激活 FP16 | 激进压缩，权重 only | 简单 |
| W4A8 | 权重 INT4，激活 INT8 | 最激进的实用配置 | 困难 |

## 扩散模型量化的特殊挑战 ⚙️

与单次前向传播的分类模型不同，扩散模型的量化面临三个独特挑战：

### 挑战 1：多步累积误差

扩散模型需要 4-50 步迭代去噪。每步的量化误差 $\delta_t$ 在后续步骤中被放大：

$$x_{t-1} = f(x_t) + \delta_t \implies x_0 = f^{(T)}(x_T) + \sum_{t=1}^{T} \prod_{k=0}^{t-1} J_k \cdot \delta_t$$

其中 $J_k$ 是第 $k$ 步的雅可比矩阵。在最坏情况下，误差可以指数级放大。这意味着扩散模型对量化误差的容忍度远低于单次推理模型。

### 挑战 2：时间步相关的激活分布

这是扩散模型量化中**最核心的难题**。不同时间步 $t$ 的输入 $x_t$ 包含不同程度的噪声，导致网络内部的激活值分布剧烈变化：

```
激活值范围
  │
  │    ████          ← t=999（高噪声）：激活值范围大，方差高
  │  ████████
  │████████████
  │
  │       ██         ← t=500（中噪声）：中间状态
  │     ██████
  │   ██████████
  │
  │        █         ← t=10（低噪声）：激活值集中，方差小
  │       ███
  │      █████
  └──────────────── 激活值
```

如果用全局统一的量化参数（一个 scale 和 zero point），则：
- 为高噪声步优化 → 低噪声步精度不足
- 为低噪声步优化 → 高噪声步发生截断

### 挑战 3：注意力层的敏感性

Self-Attention 和 Cross-Attention 中的 softmax 操作对输入数值的微小偏差极为敏感：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$QK^T$ 的量化误差在 softmax 中被指数级放大，可能导致注意力权重分布严重偏移。

## PTQ：训练后量化 ⚙️

**PTQ（Post-Training Quantization，训练后量化）** 不需要重新训练模型，只需要少量校准数据来确定量化参数。

### 标准 PTQ 流程

```python
def ptq_pipeline_for_diffusion(model, calib_data, num_calib_steps=20):
    """扩散模型 PTQ 量化流程（简化）"""
    activation_stats = {}
    
    # 步骤 1: 覆盖多时间步收集激活值统计
    for x_0 in calib_data:  # 128-512 张图像
        for t in range(0, 1000, 1000 // num_calib_steps):
            x_t = q_sample(x_0, t, torch.randn_like(x_0))
            hooks = register_activation_hooks(model)
            with torch.no_grad():
                model(x_t, t)
            for name, act in hooks.items():
                activation_stats.setdefault(name, []).append(
                    {'t': t, 'min': act.min(), 'max': act.max(), 'std': act.std()})
    
    # 步骤 2: 确定量化参数（per-channel + 99.99 percentile）
    quant_params = {name: compute_quant_params(stats, granularity='per_channel')
                    for name, stats in activation_stats.items()}
    
    # 步骤 3: 应用量化
    return apply_quantization(model, quant_params, weight_bits=8, act_bits=8)
```

### 校准数据的选择

| 策略 | 说明 | 效果 |
|------|------|------|
| 随机噪声 | 不需要真实数据 | 差——不能代表真实激活分布 |
| 真实图像 + 均匀时间步 | 最常用 | 好 |
| 真实图像 + 重点采样中间时间步 | 中间步对质量影响最大 | 更好 |
| 生成图像（教师模型生成） | 无需训练数据 | 好（与真实图像接近） |

### Per-Channel vs Per-Tensor 量化

$$\text{Per-tensor:} \quad s = \frac{x_{\max}^{\text{global}} - x_{\min}^{\text{global}}}{2^b - 1}$$

$$\text{Per-channel:} \quad s_c = \frac{x_{\max}^{(c)} - x_{\min}^{(c)}}{2^b - 1}, \quad c = 1, \ldots, C$$

Per-channel 为每个输出通道独立计算量化参数，精度更高但需要更多存储。**对扩散模型强烈推荐 per-channel**，因为不同通道在不同时间步的激活范围差异极大。

## Q-Diffusion 与 PTQD 🔬

### Q-Diffusion（Li et al., 2023）

**Q-Diffusion** 针对扩散模型的 PTQ 方法，核心创新：

1. **分步校准（Timestep-Calibrated）**：不使用全局校准统计，而是对每个时间步范围分别校准
2. **敏感度分析**：系统评估每层在不同时间步的量化敏感度
3. **自适应精度分配**：敏感层保持高精度（FP16），不敏感层使用低精度（INT4/INT8）

### PTQD（He et al., 2024）

**PTQD（Post-Training Quantization for Diffusion）** 进一步改进：

1. **时间步感知校正（Timestep-Aware Correction）**：训练一个轻量校正网络，在每步量化后补偿误差
2. **多时间步校准分组**：将 T 个时间步分为 K 组，每组使用独立的量化参数

$$\text{group}(t) = \left\lfloor \frac{t \cdot K}{T} \right\rfloor, \quad s_{\text{layer}} = s_{\text{group}(t)}$$

## 时间步感知混合精度策略 🔬

### 算法

```python
def timestep_aware_mixed_precision(model, calib_data):
    """时间步感知混合精度：为不同层/时间步分配精度"""
    # 步骤 1: 逐层、逐时间步敏感度分析（FP16 vs INT8 输出 MSE）
    sensitivity = {}
    for layer in model.layer_names():
        for t in range(0, 1000, 50):
            fp16_out = run_layer_fp16(model, layer, calib_data, t)
            int8_out = run_layer_int8(model, layer, calib_data, t)
            sensitivity[(layer, t)] = F.mse_loss(fp16_out, int8_out).item()
    
    # 步骤 2: 按敏感度排序，前 20% 保持 FP16，其余 INT8
    entries = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    cutoff = int(len(entries) * 0.2)
    config = {k: 'fp16' if i < cutoff else 'int8' for i, (k, _) in enumerate(entries)}
    
    # 步骤 3: 注意力 QKV 始终 FP16（硬规则）
    for layer in model.layer_names():
        if 'attention' in layer and 'qkv' in layer:
            for t in range(0, 1000, 50):
                config[(layer, t)] = 'fp16'
    return config
```

### 层级敏感度经验规则

| 层类型 | 量化敏感度 | 推荐精度 | 原因 |
|--------|-----------|---------|------|
| Cross-Attention QKV | 极高 | FP16 | 文本-图像对齐关键 |
| Self-Attention QKV | 高 | FP16 或 INT8 (per-channel) | softmax 放大误差 |
| Attention Output Proj | 中 | INT8 | 相对鲁棒 |
| FFN 第一层 | 中 | INT8 | 激活函数前 |
| FFN 第二层 | 低 | INT8 或 INT4 | 最不敏感 |
| Conv 层 | 低-中 | INT8 | 空间信息处理 |
| 时间步嵌入 MLP | 高 | FP16 | 条件信号关键通路 |
| GroupNorm | — | 保持 FP32 | 统计量需要高精度 |

## QAT：量化感知训练 🔬

**QAT（Quantization-Aware Training，量化感知训练）** 在训练过程中模拟量化效果，让模型学会适应量化误差。

### 直通估计器（Straight-Through Estimator, STE）

QAT 的核心技巧：在前向传播中执行量化（不可微），在反向传播中用恒等函数近似梯度：

$$\text{前向}: \hat{w} = Q(w) = s \cdot \left(\left\lfloor \frac{w}{s} \right\rceil + z - z\right)$$

$$\text{反向}: \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}} \cdot \mathbb{1}_{|w| \leq w_{\max}}$$

STE 假设量化函数的梯度为 1（在有效范围内），让梯度可以穿过量化操作进行反向传播。

### 扩散模型 QAT 训练

```python
def qat_training_step(model, x_0, t, quantizer):
    """扩散模型量化感知训练单步"""
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps)
    
    # 在前向传播中插入伪量化（Fake Quantization）
    with quantizer.enable_fake_quant():
        # 权重被伪量化: w -> Q(w)（前向）, dL/dw = dL/dQ(w)（反向，STE）
        # 激活被伪量化: a -> Q(a)（在每层输出后）
        eps_pred = model(x_t, t)
    
    loss = F.mse_loss(eps_pred, eps)
    return loss
```

QAT 通常比 PTQ 质量更好，但需要额外的训练成本：

| 量化配置 | PTQ FID 增量 | QAT FID 增量 | QAT 训练成本 |
|---------|-------------|-------------|-------------|
| W8A8 | +0.3-0.5 | +0.1-0.2 | ~10% 原始训练 |
| W4A8 | +1.0-2.0 | +0.3-0.5 | ~20% 原始训练 |
| W4A4 | +3.0-5.0 | +1.0-2.0 | ~30% 原始训练 |

## 实践基准：SD 1.5 / SDXL 量化效果 ⚙️

### Stable Diffusion 1.5 (512x512, 20步 DPM-Solver++)

| 配置 | UNet 大小 | 总显存 | 延迟 (A100) | 延迟 (RTX 4090) | FID (COCO 5K) |
|------|----------|--------|-----------|----------------|--------------|
| FP32 | 3.4 GB | ~6.5 GB | 1.35s | 1.80s | 23.1 (基准) |
| FP16 (基线) | 1.7 GB | ~3.8 GB | 1.20s | 1.60s | 23.1 |
| W8A8 PTQ | 0.85 GB | ~2.6 GB | 0.82s | 1.10s | 23.4 (+0.3) |
| W8A8 QAT | 0.85 GB | ~2.6 GB | 0.82s | 1.10s | 23.2 (+0.1) |
| W4A16 GPTQ | 0.43 GB | ~2.1 GB | 1.05s | 1.40s | 23.9 (+0.8) |
| W4A8 PTQ | 0.43 GB | ~2.1 GB | 0.65s | 0.88s | 24.5 (+1.4) |
| W4A8 QAT | 0.43 GB | ~2.1 GB | 0.65s | 0.88s | 23.6 (+0.5) |

### SDXL (1024x1024, 20步 DPM-Solver++)

| 配置 | UNet 大小 | 总显存 | 延迟 (A100) | FID |
|------|----------|--------|-----------|-----|
| FP16 (基线) | 5.1 GB | ~10.5 GB | 6.5s | 24.0 |
| W8A8 PTQ | 2.6 GB | ~7.0 GB | 4.8s | 24.5 (+0.5) |
| W4A16 | 1.3 GB | ~5.5 GB | 5.6s | 25.2 (+1.2) |
| W4A8 PTQ | 1.3 GB | ~5.5 GB | 3.5s | 26.0 (+2.0) |
| W4A8 + TP 混合精度 | 1.3 GB | ~5.5 GB | 3.5s | 24.8 (+0.8) |

> TP 混合精度 = Timestep-aware mixed Precision，对敏感层/敏感时间步保持 FP16。

## 部署流水线：PyTorch → ONNX → TensorRT ⚙️

### 完整流程

```python
import torch
import tensorrt as trt

# 步骤 1: PyTorch → ONNX
def export_unet_onnx(unet, output_path):
    dummy = (torch.randn(1, 4, 64, 64).cuda().half(),  # latent
             torch.tensor([999]).cuda(),                  # timestep
             torch.randn(1, 77, 768).cuda().half())       # text embed
    torch.onnx.export(unet, dummy, output_path, opset_version=17,
                      input_names=['sample', 'timestep', 'encoder_hidden_states'],
                      output_names=['noise_pred'],
                      dynamic_axes={'sample': {0: 'batch'}, 'noise_pred': {0: 'batch'}})

# 步骤 2: ONNX → TensorRT INT8
def build_trt_engine(onnx_path, engine_path, calibrator, precision='int8'):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt.OnnxParser(network, logger).parse(open(onnx_path, 'rb').read())
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    if precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator  # 需实现 IInt8EntropyCalibrator2
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_serialized_network(network, config)
    open(engine_path, 'wb').write(engine)
```

注意事项：ONNX 导出需设置 `dynamic_axes`；INT8 校准至少 500 张样本覆盖多时间步；FID 退化过大时对敏感层设置 FP16 fallback。

## 小结

| 要点 | 说明 |
|------|------|
| 核心挑战 | 时间步相关的激活分布变化，多步误差累积 |
| 推荐起步 | W8A8 PTQ + per-channel（简单、效果好） |
| 进阶方案 | 时间步感知混合精度（敏感层 FP16，其余 INT8） |
| 极致压缩 | W4A8 + QAT（需要训练，但 4bit 压缩） |
| 部署路径 | PyTorch → ONNX → TensorRT INT8 是最成熟的流水线 |
| 注意事项 | 注意力层 QKV 和 GroupNorm 需要保持高精度 |

---

> **下一篇**：[推理优化工程实践](./05-inference-optimization.md) — 算子、缓存、编排的全方位优化

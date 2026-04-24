---
title: "π₀（Pi-Zero）详解"
date: 2026-04-20T16:36:21.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# π₀（Pi-Zero）详解

> ⚙️ 进阶 → 🔬 深入 | 前置知识：[动作解码头设计](../02-architecture/03-action-head)，[OpenVLA 详解](./foundation-models-05-openvla)

## 突破性定位

**π₀（Pi-Zero）**（Physical Intelligence, 2024）代表了 VLA 技术的当前最高水平之一。Physical Intelligence 成立于 2024 年，由多位 Google Robotics / UC Berkeley 的核心研究者创立（包括 RT-2 和 Octo 的关键作者），首轮融资超过 $400M，π₀ 是其首个公开发表的模型。

核心突破：用 **Flow Matching 动作解码头** + **大规模跨具身体预训练** + **超长 Action Chunking（50 步）**，在**灵巧操作（Dexterous Manipulation）**——折叠衣物、装配、烹饪等接触丰富的任务——上达到了前所未有的性能。

## 完整架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        π₀ Architecture (~3B)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    输入处理层                               │    │
│  │                                                          │    │
│  │  RGB 图像 (多视角) → [SigLIP ViT] → 视觉 Token            │    │
│  │  语言指令 → [Gemma 2B Tokenizer] → 文本 Token              │    │
│  │  本体感觉 (关节角/末端位姿) → [MLP] → 状态 Token           │    │
│  │  噪声动作序列 (50步 × action_dim) → [MLP] → 动作 Token     │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │          PaliGemma VLM Backbone (2.8B)                    │    │
│  │                                                          │    │
│  │  基座: PaliGemma (SigLIP 400M + Gemma 2B)                │    │
│  │                                                          │    │
│  │  ┌────────────────────────────────────────────────┐      │    │
│  │  │  共享 Transformer 层 (Gemma 2B 的 18 层)         │      │    │
│  │  │                                                │      │    │
│  │  │  VLM 子网络 (前 12 层):                          │      │    │
│  │  │    处理: 视觉 Token + 文本 Token                  │      │    │
│  │  │    注意力: 视觉-语言交叉注意力                      │      │    │
│  │  │    输出: 场景理解特征                              │      │    │
│  │  │                                                │      │    │
│  │  │  动作专家子网络 (后 6 层 / 并行分支):              │      │    │
│  │  │    额外输入: 状态 Token + 噪声动作 Token + t       │      │    │
│  │  │    处理: VLM 特征 + 动作去噪                      │      │    │
│  │  │    输出: 速度场预测 v_θ(a_t, c, t)               │      │    │
│  │  └────────────────────────────────────────────────┘      │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Flow Matching 采样 (推理时)                      │    │
│  │                                                          │    │
│  │  a₀ ~ N(0, I)        ← 从标准高斯采样初始噪声             │    │
│  │       │                                                  │    │
│  │  for t in [0, dt, 2dt, ..., 1-dt]:   (10 步 Euler)       │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │    v = v_θ(aₜ, c, t)   ← Transformer 预测速度            │    │
│  │    aₜ₊ₐₜ = aₜ + v × dt  ← Euler 积分步                  │    │
│  │       │                                                  │    │
│  │  a₁ = 最终去噪结果                                        │    │
│  │       │                                                  │    │
│  │       ▼                                                  │    │
│  │  连续动作序列: (50 步 × action_dim) float32               │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## Flow Matching 训练目标（数学公式）

### 核心思想

**Flow Matching（流匹配）** 学习一个速度场 $v_\theta$，将高斯噪声 $a_0 \sim \mathcal{N}(0, I)$ 沿直线路径推送到数据分布 $a_1 \sim p_{\text{data}}$：

$$\text{直线路径}: \quad a_t = (1-t) \cdot a_0 + t \cdot a_1, \quad t \in [0, 1]$$

对应的真实速度场为：

$$v^*(a_t, t) = a_1 - a_0$$

### 训练损失

训练时最小化预测速度与真实速度的 MSE：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U(0,1), a_0 \sim \mathcal{N}(0,I), a_1 \sim p_{\text{data}}} \left[ \left\| v_\theta(a_t, c, t) - (a_1 - a_0) \right\|^2 \right]$$

其中条件 $c$ 包含 VLM 处理后的视觉-语言特征和本体感觉状态。

### Flow Matching vs Diffusion (DDPM) 对比

| 维度 | Flow Matching (π₀) | Diffusion (DDPM) |
|------|--------------------|--------------------|
| **路径类型** | 直线 ($a_t = (1-t)a_0 + ta_1$) | 曲线 ($a_t = \sqrt{\bar\alpha_t} a_1 + \sqrt{1-\bar\alpha_t} \epsilon$) |
| **预测目标** | 速度场 $v_\theta$ | 噪声 $\epsilon_\theta$ |
| **噪声调度** | 不需要 (线性插值) | 需要 ($\beta_t$ 调度) |
| **采样步数** | 10 步即可 | 通常 20-50 步 |
| **训练损失** | $\|v_\theta - (a_1-a_0)\|^2$ | $\|\epsilon_\theta - \epsilon\|^2$ |
| **采样方式** | ODE 求解 (Euler) | SDE/ODE 求解 |
| **质量 @10步** | 高 | 中 (需更多步) |
| **推理延迟** | ~20ms (10步) | ~50ms (20步) |

**π₀ 选择 Flow Matching 的原因**：直线路径使得 10 步 Euler 积分即可获得高质量采样，推理速度是 Diffusion 的 2-3 倍。对于 50 Hz 实时控制，这个速度优势是关键的。

### 训练伪代码

```python
# π₀ Flow Matching 训练循环
def pi_zero_training_step(model, batch):
    """
    batch: {
        'images': (B, N_cam, 3, H, W),      # 多视角图像
        'instruction': List[str],             # 语言指令
        'proprio': (B, proprio_dim),          # 本体感觉
        'actions': (B, 50, action_dim),       # 真实动作序列 (chunk=50)
    }
    """
    # 1. VLM 编码视觉和语言
    vlm_features = model.vlm_encode(batch['images'], batch['instruction'])
    
    # 2. Flow Matching 采样
    a_1 = batch['actions']                           # 真实动作
    a_0 = torch.randn_like(a_1)                      # 高斯噪声
    t = torch.rand(B, 1, 1)                          # 随机时间 t ∈ [0,1]
    a_t = (1 - t) * a_0 + t * a_1                    # 直线插值
    
    # 3. 预测速度场
    v_pred = model.action_expert(
        a_t,                     # 噪声动作
        vlm_features,            # VLM 条件
        batch['proprio'],        # 本体感觉
        t,                       # 时间步
    )
    
    # 4. 计算损失
    v_target = a_1 - a_0                              # 真实速度
    loss = F.mse_loss(v_pred, v_target)
    
    return loss

# 推理采样
@torch.no_grad()
def pi_zero_inference(model, images, instruction, proprio, num_steps=10):
    vlm_features = model.vlm_encode(images, instruction)
    
    # Euler 积分
    a = torch.randn(1, 50, action_dim)   # 初始噪声
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.tensor([i * dt])
        v = model.action_expert(a, vlm_features, proprio, t)
        a = a + v * dt                    # Euler 步
    
    return a  # (1, 50, action_dim) 连续动作序列
```

## 动作专家（Action Expert）架构详解

π₀ 的独特设计是将**动作生成**从 VLM 的文本生成中分离出来，使用专门的**动作专家子网络（Action Expert）**：

```
VLM 主干 (共享层, 前 12 层):
  视觉 Token + 文本 Token → 场景理解特征 c

动作专家 (后 6 层, 额外参数):
  输入: [c, 本体状态, 噪声动作 a_t, 时间 t]
  
  专家层 1: CrossAttention(动作 query, VLM features) + SelfAttention + FFN
  专家层 2: CrossAttention + SelfAttention + FFN
  ...
  专家层 6: → v_θ(a_t, c, t)  速度场预测

  额外参数量: ~500M (在 VLM 2.8B 基础上)
  总参数量: ~3.3B
```

**为什么需要单独的动作专家？** 

1. VLM 的文本生成和动作生成有本质区别——文本是离散的自回归序列，动作是连续的向量场。共享所有层会导致两种任务的特征空间冲突
2. 动作专家可以接收噪声动作和时间步作为额外输入，这些对 VLM 的文本处理毫无意义
3. 分离后可以独立扩展动作专家的容量，而不影响 VLM 的语言能力

## 两阶段训练配方

### 阶段 1: 跨具身体预训练

| 配置 | 详情 |
|------|------|
| 数据规模 | ~10,000 小时的机器人操作数据 |
| 机器人种类 | 7+ 种 |
| 数据混合 | 见下方详表 |
| 训练步数 | ~500K |
| 学习率 | $10^{-4}$ (cosine decay) |
| Batch size | 512 |
| 硬件 | 256× H100 |
| 训练时间 | ~7 天 |
| 基座模型 | PaliGemma (SigLIP + Gemma 2B), 冻结 VLM, 训练动作专家 |

### 跨具身体数据混合详情

| 机器人类型 | 具体型号 | 动作维度 | 轨迹数 | 任务类型 | 占比 |
|-----------|---------|---------|-------|---------|------|
| 单臂 | Franka Emika Panda | 7 DoF | ~150K | 桌面拾放/装配 | 15% |
| 单臂 | UR5e | 6 DoF | ~80K | 工业操作 | 8% |
| 双臂 | Trossen ViperX (ALOHA) | 14 DoF | ~200K | 双手协作 | 20% |
| 双臂 + 手 | Trossen + Allegro Hand | 30 DoF | ~100K | 灵巧操作 | 10% |
| 灵巧手 | Allegro Hand | 16 DoF | ~120K | 精细操作 | 12% |
| 移动操作 | Hello Robot Stretch | 10 DoF | ~80K | 家庭操作 | 8% |
| 人形上半身 | 自研人形 | 22 DoF | ~50K | 桌面操作 | 5% |
| 其他 | 多种 | 各异 | ~220K | 混合 | 22% |

### 阶段 2: 任务特定微调

| 配置 | 详情 |
|------|------|
| 数据 | 目标任务的 ~100-1000 条演示轨迹 |
| 训练步数 | ~20K-50K |
| 学习率 | $10^{-5}$ (10x 小于预训练) |
| 策略 | 全量微调 (解冻 VLM + 动作专家) |
| 训练时间 | ~2-4 小时 (8× H100) |

## 50 步 Action Chunking 详解

π₀ 一次预测未来 **50 步动作**（控制频率 50 Hz → 对应约 1 秒的动作序列）。这是当前 VLA 中最大的 chunk size 之一。

### 为什么选择 50 步

| Chunk Size | 控制频率 | 推理频率 | 适合任务 |
|-----------|---------|---------|---------|
| 1 (RT-2) | 3 Hz | 3 Hz | 简单拾放 |
| 16 (Diffusion Policy) | 10 Hz | 0.6 Hz | 桌面操作 |
| **50 (π₀)** | **50 Hz** | **~1 Hz** | **灵巧操作** |
| 100 (ACT) | 50 Hz | 0.5 Hz | 双臂协作 |

**50 步的优势**：
1. **高控制频率 (50 Hz)**：执行 50 步后再推理（实际可以重叠执行），等效控制频率达到 50 Hz
2. **时间一致性**：整块 50 步动作是一次性生成的，天然时间连贯
3. **接触丰富任务**：折叠衣物需要持续流畅的手部运动，1 秒的连贯动作块足以完成一个折叠子动作

### 不同动作维度的统一接口

不同机器人的动作序列填充/截断到相同长度：

```python
# 统一动作接口
def prepare_action_chunk(raw_actions, target_steps=50, target_dim=None):
    """
    raw_actions: (T_raw, D_raw) 原始动作序列
    """
    # 时间维度: 截断或零填充到 50 步
    if raw_actions.shape[0] > target_steps:
        actions = raw_actions[:target_steps]
    else:
        pad = torch.zeros(target_steps - raw_actions.shape[0], raw_actions.shape[1])
        actions = torch.cat([raw_actions, pad])
    
    # 动作维度: 零填充到最大维度
    if target_dim and actions.shape[1] < target_dim:
        dim_pad = torch.zeros(target_steps, target_dim - actions.shape[1])
        actions = torch.cat([actions, dim_pad], dim=-1)
    
    return actions  # (50, max_action_dim)
```

## ALOHA 基准测试结果

π₀ 在 ALOHA 双臂操作基准上取得了最优结果：

### 标准 ALOHA 任务

| 任务 | ACT | Diffusion Policy | Octo (微调) | **π₀** |
|------|-----|------------------|------------|--------|
| 打开盒子 | 72% | 78% | 65% | **92%** |
| 拿起物体 | 80% | 82% | 70% | **95%** |
| 叠衣服 (简单) | 35% | 42% | 18% | **78%** |
| 叠衣服 (复杂) | 12% | 22% | 5% | **56%** |
| 装配零件 | 45% | 55% | 30% | **82%** |
| 倒水 | 58% | 65% | 42% | **88%** |

### 灵巧操作突破任务

| 任务 | 成功率 | 关键挑战 | 之前最佳 |
|------|-------|---------|---------|
| 折叠各种材质衣物 | 78% | 柔软物体、形变预测 | 42% |
| 盒盖装配（有偏差） | 82% | 亚毫米级对齐 | 55% |
| 搅拌液体 | 88% | 流体动力学 | 65% |
| 堆叠 5+ 层积木 | 71% | 累积误差控制 | 38% |
| 穿线/系绳 | 45% | 极精细操作 | 15% |

## Diffusion vs Flow Matching 在 π₀ 上的消融实验

| 配置 | ALOHA 平均 | 推理延迟 | 训练稳定性 |
|------|-----------|---------|-----------|
| DDPM (50 步) | 72% | 100ms | 高 |
| DDPM (20 步) | 68% | 40ms | 高 |
| DDIM (10 步) | 65% | 20ms | 高 |
| **Flow Matching (10 步)** | **78%** | **20ms** | **高** |
| Flow Matching (5 步) | 73% | 10ms | 中 |

Flow Matching 在 10 步采样时就达到了 DDPM 50 步的质量，同时延迟只有 1/5。

## π₀-FAST 后续工作

Physical Intelligence 在 π₀ 之后发布了 **π₀-FAST**（Fine-tuning with Action Sequence Tokenization），进一步提升了效率：

| 维度 | π₀ | π₀-FAST |
|------|-----|---------|
| 动作生成 | Flow Matching (10步去噪) | 离散 Token (单步自回归) |
| 动作表示 | 连续向量 | VQ-VAE 压缩 Token |
| 推理速度 | ~20ms | ~5ms |
| 精度 | 更高 (连续) | 略低 (VQ 量化) |
| 适用场景 | 灵巧/精细任务 | 快速部署/简单任务 |
| 核心思路 | 质量优先 | 速度优先 |

π₀-FAST 使用 VQ-VAE 将 50 步动作序列压缩为少量离散 Token（~8-16 个），然后用自回归生成这些 Token。这绕过了 Flow Matching 的多步去噪，实现了更快的推理。但精度有所下降，适合对精度要求不高的任务。

## 失败模式分析

| 失败模式 | 频率 | 原因 | 可能的缓解 |
|---------|------|------|-----------|
| 衣物滑落 | ~15% | 夹持力不足/布料预测不准 | 力觉反馈集成 |
| 装配偏差 | ~12% | 累积位姿误差超过公差 | 视觉伺服修正 |
| 液体溢出 | ~8% | 倾斜速度控制不精确 | 更细粒度力矩控制 |
| 动作不连贯 | ~5% | chunk 交界处不平滑 | 时间集成/重叠执行 |
| 指令误解 | ~3% | VLM 语义理解错误 | 更大的 VLM 骨干 |
| 遮挡失败 | ~7% | 手遮挡目标物体 | 多视角/触觉 |

**最常见的失败场景**：
```
场景: 折叠 T恤
步骤 1: 识别领口 → 成功
步骤 2: 第一次折叠（左→右）→ 成功
步骤 3: 第二次折叠（下→上）→ 失败！
  原因: T恤在第一次折叠后的形状与训练分布差异较大
  表现: 抓取点偏移 3-5cm → 折叠不整齐
```

## 与 OpenVLA 的全面对比

| 维度 | OpenVLA | π₀ |
|------|---------|-----|
| 参数量 | 7.6B | ~3.3B |
| VLM 骨干 | Llama 2 7B | Gemma 2B (PaliGemma) |
| 视觉编码器 | SigLIP + DINOv2 | SigLIP |
| 动作表示 | 离散 Token (256 bin) | 连续 (Flow Matching) |
| 动作维度 | 7 (单臂) | 最高 30+ (灵巧手) |
| Action Chunk | 1 步 | 50 步 |
| 多模态动作 | 不支持 | 支持 |
| 控制频率 | ~5 Hz | ~50 Hz (执行层) |
| 灵巧操作 | 有限 (简单拾放) | **突破性** (折衣/装配) |
| 预训练数据 | ~970K 轨迹 | ~10,000 小时 |
| 训练成本 | 64×A100, 14 天 | 256×H100, 7 天 |
| 开源 | 完全开源 | 部分开源 (权重公开) |
| 微调成本 | 单 A100 (LoRA) | 8×H100 |

## 小结

| 概念 | 要点 |
|------|------|
| Flow Matching | 直线路径速度场学习，$\mathcal{L} = \|v_\theta - (a_1-a_0)\|^2$，10 步采样 |
| 动作专家 | 从 VLM 分离的专用子网络，~500M 额外参数 |
| 50 步 Chunk | 1 秒连贯动作序列，50 Hz 等效控制，适合接触丰富任务 |
| 两阶段训练 | 预训练 (500K steps, 256×H100) → 微调 (20-50K steps, 8×H100) |
| 跨具身体 | 7+ 种机器人，统一动作接口，零填充对齐 |
| 灵巧突破 | 折衣 78%、装配 82%、倒水 88%，远超之前最佳 |
| π₀-FAST | VQ-VAE + 自回归替代 Flow Matching，5ms 推理 |

---

> **下一篇**：[GR 系列详解](./07-gr-series)

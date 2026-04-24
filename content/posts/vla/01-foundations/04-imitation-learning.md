---
title: "模仿学习基础"
date: 2026-04-20T16:48:44.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 模仿学习基础

> 🔰 入门 → ⚙️ 进阶 | 前置知识：[观测空间与动作空间](./03-observation-action-space)
> 交叉引用：[强化学习基础](./05-reinforcement-learning)、[Diffusion 动作生成](../07-foundation-models/03-diffusion-action)、[VLA 未来展望](../07-foundation-models/06-future)

## 什么是模仿学习

**模仿学习（Imitation Learning, IL）** 是让机器人通过观察人类演示来学习技能的方法——就像孩子看大人做然后照着做。这是 VLA 最主要的训练范式，也是当前机器人学习领域最实用的方法。

核心流程：
```
人类遥操作演示 → 收集 (观测, 动作) 数据对 → 训练模型 → 机器人自主执行
     │                    │                    │              │
  数据来源            数据集构成            学习算法        部署推理
  (遥操作器/VR)    D = {(o_i, a_i)}     (BC/IBC/ACT)    (闭环控制)
```

### 为什么 VLA 选择模仿学习？

| 理由 | 解释 |
|------|------|
| **数据高效** | 50-200 条演示即可学会单个任务（RL 需要 ~100K 次交互） |
| **安全** | 训练过程不需要在真机上试错 |
| **兼容预训练** | VLM 骨干用互联网数据预训练，然后用演示数据微调 |
| **直觉简单** | "看我做一遍"比"设计奖励函数"对操作员更友好 |
| **质量可控** | 演示质量直接决定策略上限，可筛选优质数据 |

## 行为克隆（Behavior Cloning, BC）

### 核心思想

**行为克隆（Behavior Cloning, BC）** 是最简单直接的模仿学习方法：把演示数据当作监督学习数据集，训练一个从观测到动作的映射。

$$\pi_\theta(a|o) \approx \pi_{\text{expert}}(a|o)$$

### 损失函数推导

**连续动作空间（MSE Loss）**：

当动作是连续值（如笛卡尔空间 $\Delta x, \Delta y, \Delta z, ...$）时，假设动作条件分布为高斯分布：

$$p_\theta(a|o) = \mathcal{N}(a; \mu_\theta(o), \sigma^2 I)$$

最大化对数似然等价于最小化 MSE：

$$L_{\text{MSE}} = \mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \| \mu_\theta(o) - a \|^2 \right]$$

**问题**：MSE 损失假设单峰高斯分布，但真实动作分布往往是**多峰的**（multimodal）。例如，面对一个物体时，从左边抓和从右边抓都是合理的——MSE 会取平均值，导致从中间抓（很可能失败）。

**离散动作空间（交叉熵 Loss）**：

当动作被 Token 化为离散值（如 RT-2 将每个维度量化为 256 bin）时：

$$L_{\text{CE}} = -\mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \sum_{d=1}^{D} \sum_{k=1}^{K} a_d^{(k)} \log \pi_\theta^{(k)}(o)_d \right]$$

其中 $D$ 是动作维度数，$K$ 是每个维度的离散 bin 数（如 256），$a_d^{(k)}$ 是第 $d$ 维动作在第 $k$ 个 bin 的 one-hot 标签。

**离散化的优势**：交叉熵天然支持多峰分布——模型可以在多个 bin 上都给出高概率。

### BC 训练伪代码

```python
# 行为克隆完整训练流程
import torch
from torch.utils.data import DataLoader

# 1. 数据准备
dataset = RobotDemonstrationDataset(
    path="demos/pick_place/",
    obs_keys=["rgb_static", "rgb_gripper", "proprioception"],
    action_key="actions",          # 7-d: [dx,dy,dz,droll,dpitch,dyaw,gripper]
    chunk_size=16,                 # Action Chunking: 一次预测 16 步
    augmentations=["color_jitter", "random_crop", "random_erasing"],
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. 模型定义 (简化版)
model = VLAModel(
    vision_encoder="SigLIP-SO400M",   # 预训练视觉编码器
    language_encoder="PaliGemma-3B",   # 预训练语言模型
    action_head="continuous_mlp",      # 连续动作输出头
    action_dim=7,
    chunk_size=16,
)

# 3. 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100_000)

for epoch in range(100):
    for batch in loader:
        obs = batch["observations"]        # 图像 + 本体感觉
        lang = batch["language"]           # 语言指令
        actions_gt = batch["actions"]      # 专家动作 [B, 16, 7]

        actions_pred = model(obs, lang)    # 模型预测 [B, 16, 7]
        loss = F.mse_loss(actions_pred, actions_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

### BC 就是 VLA 的训练方式

VLA 的训练本质上就是大规模行为克隆，只是规模和复杂度远超传统 BC：

| 维度 | 传统 BC | VLA 规模 BC |
|------|--------|------------|
| 模型大小 | 1-50M 参数 | 3-55B 参数 |
| 数据量 | 100-1000 条演示 | 100K-1M 条 |
| 观测模态 | 单相机 | 多相机 + 本体感觉 + 语言 |
| 预训练 | 无 / ImageNet | 互联网 VLM 预训练 |
| 任务数 | 单任务 | 100-1000+ 任务 |
| 泛化能力 | 几乎不泛化 | 语义泛化到新指令/新物体 |

## 分布偏移问题

⚙️ 进阶

### 问题描述

**分布偏移（Distribution Shift）** 是 BC 的核心理论缺陷：

训练时，模型看到的是专家轨迹上的观测。但推理时，模型的微小预测误差会导致实际状态偏离专家轨迹——进入一个训练时从未见过的状态。在这个陌生状态下，模型的预测更不准确，导致进一步偏离，误差像滚雪球一样累积。

```
专家轨迹:  s₀ ──→ s₁ ──→ s₂ ──→ s₃ ──→ s₄ ──→ 成功！
                                ↗
模型执行:  s₀ ──→ s₁ ──→ s₂' ──→ s₃' ──→ s₄' ──→ 失败...
                    ↑微小误差    ↑误差放大   ↑完全偏离
```

### Ross & Bagnell 误差累积定理（2010）

⚙️ 进阶

**定理**：设专家策略 $\pi^*$ 的期望代价为 $J(\pi^*)$，BC 学到的策略 $\hat{\pi}$ 的单步误差为 $\epsilon = \mathbb{E}_{s \sim d_{\pi^*}}[C(s, \hat{\pi}(s))]$，则在时间步 $T$ 的任务中：

$$J(\hat{\pi}) \leq J(\pi^*) + T^2 \epsilon$$

**证明核心思路**：

1. 在第 $t$ 步，策略 $\hat{\pi}$ 以概率 $\leq t\epsilon$ 偏离专家分布
2. 一旦偏离，后续所有步的代价上界为 $T - t$
3. 对所有时间步求和：$\sum_{t=0}^{T-1} t\epsilon \cdot 1 \leq T^2 \epsilon$

**直觉理解**：时间步数 $T$ 的影响是平方级的。对于短任务（$T=10$），$100\epsilon$ 可能还能接受。但对于长任务（$T=1000$），$10^6 \epsilon$ 意味着即使单步误差极小，策略也几乎必然失败。

### DAgger 算法

**DAgger（Dataset Aggregation，数据集聚合）**（Ross et al., 2011）通过迭代式数据收集将误差上界从 $O(T^2\epsilon)$ 降至 $O(T\epsilon)$：

```python
# DAgger 核心循环
D = collect_expert_demos(expert, env, n_demos=100)
policy = train_bc(D)

for i in range(n_iterations):
    beta_i = max(0.0, 1.0 - i / n_iterations)  # 专家比例递减
    new_data = []
    for _ in range(n_rollouts):
        obs = env.reset()
        for t in range(max_steps):
            # 以 beta_i 概率用专家执行，否则用学习策略
            action_exec = expert(obs) if random() < beta_i else policy(obs)
            action_label = expert(obs)  # 关键: 始终请专家标注正确动作
            new_data.append((obs, action_label))
            obs = env.step(action_exec)
    D = D + new_data
    policy = train_bc(D)  # 在聚合数据集上重新训练
```

**DAgger 的限制**：需要专家在线标注（成本高），不适用于危险场景。

## Implicit Behavior Cloning (IBC)

⚙️ 进阶

**隐式行为克隆（Implicit Behavior Cloning, IBC）**（Florence et al., 2022）用**能量函数**（Energy-Based Model, EBM）替代显式策略网络，天然处理多峰分布。

$$E_\theta(o, a) \in \mathbb{R} \quad \text{（低能量 = 好的动作，高能量 = 差的动作）}$$

$$\pi(a|o) = \frac{\exp(-E_\theta(o, a))}{Z(o)}, \quad Z(o) = \int \exp(-E_\theta(o, a)) da$$

### IBC vs 显式 BC 对比

```
场景: 物体可以从左边或右边绕过

显式 BC (MSE):
  两条演示: a_left, a_right
  MSE 平均 → 预测 a_middle (从中间撞上去!)

  P(a)
   ↑
   │    ╭─╮
   │    │ │
   └────┼─┼──────→ a
        a_mid
        (失败)

IBC (能量函数):
  能量在 a_left 和 a_right 处都低
  采样时会落在其中一个峰 → 成功

  E(a)
   ↑
   │  ╲          ╱
   │   ╲        ╱
   │    ╰──╮╭──╯
   └───────┼┼────→ a
          a_L a_R
          (都可以)
```

## BC 变体对比

| 方法 | 动作表示 | 分布假设 | 多峰处理 | 推理速度 | 精度 | 代表模型 |
|------|---------|---------|---------|---------|------|---------|
| **BC (MSE)** | 连续 | 单峰高斯 | 差 | 快 (1次前向) | 中 | RT-1 |
| **BC (CE)** | 离散 Token | 类别分布 | 较好 | 快 (自回归) | 中 | RT-2, OpenVLA |
| **BC-RNN** | 连续序列 | 单峰 (时序) | 差 | 中 | 中高 | robomimic |
| **IBC** | 连续 (能量) | 任意多峰 | 好 | 慢 (需采样) | 高 | Florence 2022 |
| **ACT** | 连续 Chunk | CVAE 多峰 | 好 | 快 | 高 | ALOHA |
| **Diffusion Policy** | 连续 Chunk | 扩散多峰 | 最好 | 慢 (需去噪) | 最高 | Chi et al. 2023 |
| **Flow Matching** | 连续 Chunk | 流式多峰 | 好 | 较快 | 高 | pi_0 |

### 关键 Benchmark 性能对比

在 robomimic 标准评测上（模拟器环境，200 条专家演示）：

| 方法 | Can (简单) | Square (中等) | Transport (困难) | 推理延迟 |
|------|-----------|-------------|-----------------|---------|
| BC-MLP | 82% | 42% | 18% | 2 ms |
| BC-RNN | 88% | 68% | 35% | 5 ms |
| BC-Transformer | 90% | 72% | 42% | 15 ms |
| IBC | 85% | 75% | 48% | 100 ms |
| ACT | 94% | 85% | 62% | 20 ms |
| Diffusion Policy | **96%** | **90%** | **72%** | 80 ms |

## 真实世界 BC 训练指南

### 数据收集最佳实践

| 因素 | 推荐值 | 理由 |
|------|--------|------|
| **单任务演示数** | 50-200 条 | <50 不够多样，>200 收益递减 |
| **演示成功率** | >95% | 失败数据会污染策略 |
| **遥操作速度** | 正常速度的 50-70% | 过快导致抖动、过慢不自然 |
| **场景多样性** | 每次稍微变动物体位置/朝向 | 覆盖更多状态空间 |
| **数据质量检查** | 人工抽检 10% | 删除异常轨迹 |

### 数据增强策略

```python
# VLA 训练常用数据增强
augmentations = {
    # 视觉增强 (几乎所有 VLA 都使用)
    "random_crop":       {"scale": (0.9, 1.0), "ratio": (0.9, 1.1)},
    "color_jitter":      {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3},
    "random_erasing":    {"p": 0.1, "scale": (0.02, 0.05)},   # 模拟遮挡
    "gaussian_blur":     {"kernel": 5, "sigma": (0.1, 2.0)},

    # 空间增强 (需要同步修改动作标签!)
    "random_translation": {"pixels": 10},  # 图像平移 → 动作相应偏移
    "random_rotation":    {"degrees": 5},  # 图像旋转 → 动作相应旋转

    # 时间增强
    "frame_skip":        {"p": 0.1},       # 随机跳帧，增加鲁棒性
    "speed_perturbation": {"range": (0.8, 1.2)},  # 随机变速
}
```

### 与 VLA 训练管线的连接

VLA 训练全流程分三阶段：**阶段 1**——VLM 互联网预训练（LAION-5B/WebLI，图文对比+VQA，获得通用视觉-语言表征）；**阶段 2**——机器人数据 BC 微调（Open X-Embodiment 数据，观测→动作映射，损失为 MSE/CE/Flow Matching）；**阶段 3**——可选的 RL 对齐微调（RLHF/DPO，人类偏好标注，详见[强化学习基础](./05-reinforcement-learning)）。

## 小结

| 概念 | 要点 |
|------|------|
| 行为克隆 | 从专家演示直接学习观测→动作映射，是 VLA 的核心训练范式 |
| MSE 损失 | 假设单峰高斯，无法处理多峰动作分布 |
| CE 损失 | 离散 Token 化动作，天然支持多峰（RT-2 路线） |
| 分布偏移 | 推理时误差累积 $O(T^2\epsilon)$，长任务尤其严重 |
| DAgger | 迭代式数据聚合，降至 $O(T\epsilon)$，但需在线专家 |
| IBC | 能量模型隐式表示多峰分布，但推理需要采样 |
| Diffusion Policy | 当前最佳多峰处理，精度最高但推理较慢 |
| VLA 的缓解策略 | 大数据 + Action Chunking + 数据增强 + 闭环反馈 |
| 训练实践 | 50-200 条演示/任务，丰富数据增强，三阶段训练管线 |

---

> **下一篇**：[强化学习基础](./05-reinforcement-learning)

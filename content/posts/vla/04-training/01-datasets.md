---
title: "4.1 大规模机器人数据集"
date: 2026-04-20T16:10:22.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 4.1 大规模机器人数据集

> **难度**: 🔰入门 | **前置阅读**: [1.1 什么是VLA](../foundations-01-what-is-vla), [2.1 整体架构](../02-architecture/01-vlm-to-vla)

## 为什么数据集是VLA的基石

**视觉-语言-动作模型（Vision-Language-Action Model, VLA）** 的核心能力来自于大规模、多样化的机器人操作数据。与纯视觉或语言模型不同，VLA需要同时学习"看到什么"、"理解什么指令"和"如何动作"三者之间的映射。数据集的规模与多样性直接决定了模型的泛化能力。

---

## 核心数据集一览

### Open X-Embodiment (OXE)

**Open X-Embodiment** 是由Google DeepMind联合34家机构发布的开放机器人数据集联盟，是目前最大的多机器人操作数据集。

| 属性 | 详情 |
|------|------|
| 规模 | ~100万条机器人轨迹 |
| 机器人类型 | 22种不同具身体（manipulator、mobile等） |
| 任务种类 | 抓取、推移、叠放、开关抽屉等500+任务 |
| 数据格式 | RLDS (Reinforcement Learning Datasets) |
| 开放协议 | Apache 2.0 |

```python
# OXE 数据集加载示例（基于 TensorFlow Datasets）
import tensorflow_datasets as tfds

# 加载 Bridge V2 子集
dataset = tfds.load(
    'bridge_dataset',
    split='train',
    data_dir='/path/to/oxe_data'
)

for episode in dataset.take(1):
    steps = episode['steps']
    for step in steps:
        image = step['observation']['image']       # RGB图像
        action = step['action']                     # 7-DoF动作
        instruction = step['language_instruction']  # 自然语言指令
```

### DROID

**DROID（Distributed Robot Interaction Dataset）** 是一个分布式收集的大规模真实机器人操作数据集。

| 属性 | 详情 |
|------|------|
| 规模 | 76,000+轨迹 |
| 收集方式 | 分布于多个实验室的Franka机器人 |
| 特点 | 多视角（眼-手-外部）、场景多样性极高 |
| 标注 | 自然语言指令 + 成功/失败标签 |

### RH20T

**RH20T（Robot Hand 20 Tasks）** 是由清华大学等机构发布的中国机器人操作数据集。

| 属性 | 详情 |
|------|------|
| 规模 | 110,000+轨迹 |
| 任务 | 20类日常操作任务 |
| 特点 | 包含力/触觉传感数据、多模态 |
| 机器人 | 多种机械臂 + 灵巧手 |

### Bridge V2

**Bridge V2** 是UC Berkeley发布的桌面操作数据集，是OXE的重要组成部分。

| 属性 | 详情 |
|------|------|
| 规模 | 60,000+轨迹 |
| 机器人 | WidowX 250 6-DoF机械臂 |
| 特点 | 场景/物体多样性高、标注质量好 |
| 用途 | VLA微调的常用benchmark |

---

## RLDS 标准格式 ⚙️进阶

**RLDS（Reinforcement Learning Datasets）** 是Google提出的机器人学习数据标准格式，旨在统一不同来源的机器人数据表示。

### 数据结构

```
Dataset
  └── Episode（一条完整轨迹）
        ├── metadata: {language_instruction, ...}
        └── Steps（时间步序列）
              ├── observation
              │     ├── image: (H, W, 3) uint8
              │     ├── wrist_image: (H, W, 3) uint8  # 可选
              │     └── state: (N,) float32            # 关节状态
              ├── action: (M,) float32                 # 动作向量
              ├── reward: float32
              ├── is_terminal: bool
              └── is_first / is_last: bool
```

### 为什么需要统一格式

| 问题 | RLDS的解决方案 |
|------|---------------|
| 不同机器人动作空间不一致 | 统一的Episode/Step层级结构 |
| 数据加载效率低 | 基于TFRecord，支持流式读取 |
| 跨数据集混合训练困难 | 标准化的特征命名与类型 |
| 缺乏元数据 | 内置episode级别的metadata |

```python
# RLDS 数据转换示例
import rlds

def transform_step(step):
    """将原始数据转换为RLDS标准格式"""
    return {
        'observation': {
            'image': tf.image.resize(step['rgb'], [256, 256]),
            'state': step['joint_positions'],
        },
        'action': step['target_joint_positions'],
        'is_terminal': step['done'],
        'language_instruction': step['task_description'],
    }

# 应用转换
standardized = raw_dataset.map(transform_step)
```

---

## 数据集规模对比

| 数据集 | 轨迹数 | 机器人数 | 多模态 | 开放 |
|--------|--------|---------|--------|------|
| **OXE** | ~1,000,000 | 22种 | 视觉+语言 | 是 |
| **DROID** | 76,000+ | 1种(多站点) | 多视角+语言 | 是 |
| **RH20T** | 110,000+ | 多种 | 视觉+力觉+语言 | 是 |
| **Bridge V2** | 60,000+ | 1种 | 视觉+语言 | 是 |
| **RT-1 Data** | 130,000+ | 1种 | 视觉+语言 | 部分 |

---

## 数据规模的影响 🔬深入

研究表明，VLA模型性能与数据规模呈现**对数线性（log-linear）** 关系：

- RT-2实验：数据量从1K增至130K轨迹，成功率从32%提升至73%
- OXE实验：混合多机器人数据后，单个机器人的性能反而提升（正迁移）
- 规模定律（Scaling Law）在机器人数据上同样成立，但数据获取成本远高于文本/图像

### 数据规模 vs. 成功率（实测参考值）

| 训练数据量 | RT-2成功率 | Octo成功率 | 说明 |
|-----------|-----------|-----------|------|
| 1K 轨迹 | 32% | 28% | 最低可用阈值 |
| 10K 轨迹 | 51% | 45% | 单任务可用 |
| 50K 轨迹 | 65% | 58% | 多任务初步泛化 |
| 130K 轨迹 | 73% | - | 强泛化 |
| 1M 轨迹 (OXE) | - | 68% | 跨具身体泛化 |

### 数据多样性的维度

除规模外，数据多样性同样关键。多样性可从以下维度衡量：

| 多样性维度 | 描述 | 对泛化的影响 |
|-----------|------|-------------|
| **物体多样性** | 训练中出现的物体种类数 | 提升新物体抓取成功率 |
| **场景多样性** | 背景、光照、桌面材质变化 | 减少视觉过拟合 |
| **任务多样性** | 抓取/推移/倾倒等不同技能 | 增强任务组合泛化 |
| **机器人多样性** | 不同具身体的数据 | 实现跨具身体迁移（见 [4.4](./04-cross-embodiment)） |
| **指令多样性** | 同一任务的不同语言描述 | 增强语言理解鲁棒性 |

```python
# 数据混合策略示例
def create_training_mixture(datasets, weights=None):
    """
    将多个数据集按权重混合
    OXE推荐：按数据集大小的平方根加权（防止大数据集主导）
    """
    if weights is None:
        # 平方根加权
        sizes = [len(d) for d in datasets]
        weights = [math.sqrt(s) for s in sizes]
        total = sum(weights)
        weights = [w / total for w in weights]
    
    return MixedDataset(datasets, sampling_weights=weights)

# 示例：混合OXE子集
mixture = create_training_mixture([
    bridge_v2,      # 60K轨迹 → 权重 ~0.35
    fractal,        # 100K轨迹 → 权重 ~0.45
    taco_play,      # 8K轨迹  → 权重 ~0.13
    jaco_play,      # 1K轨迹  → 权重 ~0.07（小数据集获得更高权重比例）
])
```

---

## 数据集获取与使用 🔰入门

大部分数据集可通过TensorFlow Datasets或HuggingFace Hub免费获取：

```bash
# 通过 TensorFlow Datasets 获取 OXE 数据
pip install tensorflow-datasets

# 通过 HuggingFace Hub 获取 LeRobot 格式数据
pip install lerobot
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/aloha_sim_transfer_cube_human')
print(f'Episodes: {ds.meta.total_episodes}, FPS: {ds.meta.fps}')
"
```

详细的数据加载与使用方法参见 [4.7 开源工具链](./07-toolchains)。

---

## 本节小结

| 要点 | 说明 |
|------|------|
| 核心数据集 | OXE（最大）、DROID（多站点）、RH20T（多模态）、Bridge V2（benchmark） |
| 统一格式 | RLDS是当前事实标准，Episode→Steps层级结构 |
| 规模影响 | 对数线性scaling，混合训练有正迁移效果 |
| 开放趋势 | 社区共建、数据共享是VLA发展的关键推动力 |

---

> **下一节**: [4.2 数据收集与标注](./02-data-collection) - 这些数据是如何获取的？

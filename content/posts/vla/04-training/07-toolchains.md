---
title: "4.7 开源工具链"
date: 2026-04-20T16:10:59.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 4.7 开源工具链

> **难度**: 🔰入门 → ⚙️进阶 | **前置阅读**: [4.1 大规模机器人数据集](./01-datasets), [4.3 预训练与微调范式](./03-pretrain-finetune)

## 概述

开源工具链极大降低了VLA研究和开发的门槛。本节介绍三个最重要的开源工具链：HuggingFace的LeRobot、robomimic以及DROID工具链，涵盖从数据管理到策略训练的完整流程。

---

## LeRobot（HuggingFace）🔰入门

**LeRobot** 是HuggingFace推出的开源机器人学习框架，目标是成为机器人领域的"Transformers库"。它提供了统一的数据格式、多种策略实现和端到端训练流程。

### 架构总览

```
LeRobot 架构
├── lerobot/
│   ├── common/
│   │   ├── datasets/        # 统一数据加载与格式
│   │   ├── envs/            # 仿真环境封装
│   │   ├── policies/        # 策略实现
│   │   └── robot_devices/   # 真实机器人接口
│   ├── configs/             # Hydra配置文件
│   └── scripts/
│       ├── train.py         # 统一训练入口
│       ├── eval.py          # 评估脚本
│       ├── push_dataset_to_hub.py
│       └── control_robot.py # 数据采集 / 部署
```

### 数据格式

LeRobot使用基于HuggingFace Datasets的标准格式，支持直接从Hub加载。

```python
# LeRobot 数据加载
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 从 HuggingFace Hub 加载
dataset = LeRobotDataset(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    root=None,  # 自动从Hub下载
)

# 数据结构
print(dataset[0].keys())
# dict_keys([
#     'observation.images.top',      # (C, H, W) 图像
#     'observation.state',           # (N,) 关节状态
#     'action',                      # (M,) 动作
#     'episode_index',               # 轨迹编号
#     'frame_index',                 # 帧编号
#     'timestamp',                   # 时间戳
#     'next.done',                   # 是否结束
# ])

# 数据集元信息
print(dataset.meta)
# fps: 50, robot_type: "aloha", total_episodes: 50, ...
```

### 支持的策略

| 策略 | 类型 | 论文 | 适用场景 |
|------|------|------|---------|
| **ACT** | 动作分块Transformer | Zhao et al., 2023 | 双臂精细操作 |
| **Diffusion Policy** | 扩散动作生成 | Chi et al., 2023 | 通用操作 |
| **TDMPC** | 基于模型的规划 | Hansen et al., 2022 | 需要规划的任务 |
| **VQ-BeT** | 离散化BeT | Lee et al., 2024 | 多模态动作 |
| **Pi0** | Flow Matching VLA | Physical Intelligence | 跨具身体通用 |

### 训练示例

```python
# LeRobot 训练命令
# ACT策略训练 ALOHA仿真任务
"""
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --training.num_epochs=2000 \
    --training.batch_size=8 \
    --training.lr=1e-5 \
    --wandb.enable=true
"""

# 等价的 Python API
from lerobot.scripts.train import train
from lerobot.common.policies.act.configuration_act import ACTConfig

policy_cfg = ACTConfig(
    chunk_size=100,              # 动作分块长度
    n_action_steps=100,
    dim_model=512,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=7,
)

train(
    policy_cfg=policy_cfg,
    dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
    num_epochs=2000,
    batch_size=8,
)
```

### 核心策略简介

**ACT（Action Chunking with Transformers）** 是ALOHA配套策略：一次预测未来K步动作（动作分块），基于CVAE架构，使用指数加权平均融合连续预测。详见相关论文 Zhao et al., 2023。

**扩散策略（Diffusion Policy）** 将动作生成建模为去噪扩散过程。训练时给动作序列加噪并学习去噪，推理时从纯噪声迭代还原出动作。擅长处理多模态动作分布（如同一物体可从左/右侧抓取）。详见 Chi et al., 2023。

**TDMPC（Temporal Difference Model Predictive Control）** 结合了模型学习与在线规划，学习世界模型后在推理时通过MPC搜索最优动作序列。适用于需要前瞻规划的长时域任务。

---

## robomimic ⚙️进阶

**robomimic** 是由Stanford和UT Austin开发的机器人模仿学习框架，提供了标准化的数据格式、多种基线算法和系统化的实验管理。

### 核心特性

| 特性 | 说明 |
|------|------|
| 算法库 | BC, BC-RNN, HBC, IRIS, IQL等 |
| 数据格式 | HDF5，支持图像/状态/动作 |
| 环境集成 | robosuite (MuJoCo) |
| 实验管理 | 完整的配置系统和日志 |
| 基准测试 | 标准化的6类操作任务 |

### 数据格式

```python
# robomimic HDF5 数据结构
import h5py

with h5py.File("dataset.hdf5", "r") as f:
    # 数据结构
    # f["data"]
    #   ├── demo_0/
    #   │   ├── obs/
    #   │   │   ├── agentview_image   (T, 84, 84, 3)
    #   │   │   ├── robot0_eef_pos    (T, 3)
    #   │   │   └── robot0_gripper    (T, 1)
    #   │   ├── actions               (T, 7)
    #   │   ├── rewards               (T,)
    #   │   └── dones                 (T,)
    #   ├── demo_1/
    #   └── ...
    
    demo = f["data/demo_0"]
    images = demo["obs/agentview_image"][:]  # (T, 84, 84, 3)
    actions = demo["actions"][:]              # (T, 7)
```

### 训练配置

robomimic使用JSON配置文件管理实验，核心字段包括：

| 配置项 | 示例值 | 说明 |
|--------|--------|------|
| `algo_name` | `"bc"` / `"bc_rnn"` / `"hbc"` | 选择算法 |
| `train.batch_size` | 100 | 训练批大小 |
| `train.num_epochs` | 2000 | 训练轮数 |
| `observation.modalities.obs.rgb` | `["agentview_image"]` | 图像观测键名 |
| `observation.encoder.rgb.core_class` | `"VisualCore"` | 视觉编码器（默认ResNet18） |
| `algo.rnn.enabled` | `true` | 是否启用RNN历史编码 |

---

## DROID 工具链 ⚙️进阶

**DROID工具链** 是伴随DROID数据集发布的完整数据收集和训练框架，专为分布式大规模数据采集设计。

### 工具链组成

```
DROID 工具链
├── 数据收集
│   ├── droid/robot_env.py       # Franka机器人控制
│   ├── droid/controllers/       # VR遥操作控制器
│   └── droid/camera_utils/      # 多视角相机管理
├── 数据处理
│   ├── data_processing/filter.py    # 质量过滤
│   ├── data_processing/convert.py   # RLDS格式转换
│   └── data_processing/augment.py   # 数据增强
├── 训练
│   ├── training/octo_finetune.py    # Octo微调
│   └── training/diffusion_policy.py # Diffusion Policy训练
└── 部署
    └── deployment/real_eval.py      # 真机评估
```

### 数据收集流程

```python
# DROID 数据收集伪代码
from droid.robot_env import RobotEnv
from droid.controllers import VRController

env = RobotEnv(
    robot="franka",
    cameras=["front", "wrist_left", "wrist_right"],  # 三视角
    recording_fps=15,
)
controller = VRController(device="quest_pro")

# 采集循环
task = input("请输入任务描述: ")  # "pick up the red cup"
env.start_recording(task_instruction=task)

while not controller.is_done():
    vr_action = controller.get_action()
    obs, reward, done, info = env.step(vr_action)

episode = env.stop_recording()
# 自动保存为标准格式
episode.save(path="/data/droid/episode_001.hdf5")
```

---

## 工具链选择指南

| 需求 | 推荐工具 | 原因 |
|------|---------|------|
| VLA策略训练入门 | LeRobot | 一键训练，Hub集成，社区活跃 |
| 标准模仿学习研究 | robomimic | 算法全面，实验管理成熟 |
| 大规模数据采集 | DROID工具链 | 分布式设计，质量控制完善 |
| ALOHA双臂任务 | LeRobot + ACT | 原生支持ALOHA硬件 |
| 扩散策略研究 | LeRobot | Diffusion Policy实现完整 |
| 快速原型验证 | LeRobot | 最少配置即可运行 |

### 工具链对比

| 维度 | LeRobot | robomimic | DROID |
|------|---------|-----------|-------|
| 维护方 | HuggingFace | Stanford/UT Austin | 多校联合 |
| 策略数量 | 5+ | 6+ | 2 |
| 数据格式 | HF Datasets | HDF5 | HDF5/RLDS |
| 仿真环境 | 多种 | robosuite | 无内置 |
| 真机支持 | 是（多种） | 有限 | 是（Franka） |
| Hub集成 | 原生 | 无 | 无 |
| VLA支持 | 是 (Pi0等) | 间接 | 是 (Octo) |
| 文档质量 | 优秀 | 良好 | 良好 |
| 活跃度 | 最高 | 中 | 中 |

---

## 快速上手：LeRobot 完整流程 🔰入门

```bash
# 1. 安装
pip install lerobot

# 2. 可视化已有数据集
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/aloha_sim_transfer_cube_human \
    --episode-index 0

# 3. 训练 ACT 策略
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --output_dir=outputs/act_aloha

# 4. 评估
python -m lerobot.scripts.eval \
    --policy.path=outputs/act_aloha/checkpoints/last/pretrained_model \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50

# 5. 上传到 Hub（可选）
huggingface-cli upload my-org/act-aloha outputs/act_aloha
```

---

## 本节小结

| 要点 | 说明 |
|------|------|
| LeRobot | HuggingFace出品，VLA训练的一站式框架，推荐首选 |
| 核心策略 | ACT（动作分块）、Diffusion Policy（扩散生成）、TDMPC（模型规划） |
| robomimic | 经典模仿学习框架，算法全面，适合学术研究 |
| DROID工具链 | 分布式数据采集+训练，适合大规模数据工程 |
| 入门路径 | LeRobot安装 → 加载数据 → 训练ACT → 仿真评估 |

---

> **模块总结**: 本模块（Module 4: Training）覆盖了VLA训练的完整流程。继续阅读 [Module 5](../deployment-) 了解如何将训练好的VLA部署到真实机器人上。

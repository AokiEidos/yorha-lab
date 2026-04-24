---
title: "4.2 数据收集与标注"
date: 2026-04-20T16:04:33.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 4.2 数据收集与标注

> **难度**: 🔰入门 → ⚙️进阶 | **前置阅读**: [4.1 大规模机器人数据集](./01-datasets)

## 概述

机器人数据的采集是VLA训练流水线中成本最高、最耗时的环节。与互联网上海量的文本/图像不同，机器人操作数据需要在物理世界中逐条录制。本节介绍主流的数据收集方法、标注策略和质量控制流程。

---

## 遥操作（Teleoperation）🔰入门

**遥操作（Teleoperation）** 是指人类操作者通过远程控制设备实时驱动机器人完成任务，同时记录全部传感数据和动作序列。这是当前VLA数据收集的主流方式。

### 遥操作的基本流程

```
操作者输入 → 控制信号映射 → 机器人执行 → 传感器记录
   ↓              ↓              ↓            ↓
 手柄/VR手套    坐标变换     关节运动     image + state + action
```

### 主要遥操作设备对比

| 设备类型 | 代表产品 | 自由度 | 力反馈 | 成本 | 适用场景 |
|---------|---------|--------|--------|------|---------|
| 主从机械臂 | Gello, ALOHA | 6-7 DoF | 有 | 高 | 精细双臂操作 |
| VR控制器 | Meta Quest, HTC Vive | 6 DoF | 振动 | 中 | 通用抓取/放置 |
| 3D鼠标 | SpaceMouse | 6 DoF | 无 | 低 | 简单桌面操作 |
| 手持示教 | 直接拖拽机械臂 | 原始DoF | 自然 | 无额外 | 快速演示 |
| 数据手套 | Manus, Rokoko | 手指级 | 部分 | 高 | 灵巧手操作 |

---

## ALOHA 双臂遥操作 ⚙️进阶

**ALOHA（A Low-cost Open-source Hardware System for Bimanual Teleoperation）** 是斯坦福大学开发的低成本双臂遥操作系统，在VLA社区中被广泛采用。

### 系统架构

```
主臂（Leader）×2          从臂（Follower）×2
  ViperX 300             ViperX 300
      ↓                       ↓
  关节角度读取    →    关节角度映射执行
                              ↓
                    4个摄像头同步录制
                    (顶部×1 + 手腕×2 + 侧面×1)
```

### 数据记录格式

```python
# ALOHA 单步数据结构
aloha_step = {
    'observation': {
        'images': {
            'top':        np.array([480, 640, 3]),   # 顶部相机
            'left_wrist': np.array([480, 640, 3]),   # 左手腕相机
            'right_wrist': np.array([480, 640, 3]),  # 右手腕相机
        },
        'qpos': np.array([14]),   # 双臂关节角度 (7+7)
        'qvel': np.array([14]),   # 双臂关节速度
    },
    'action': np.array([14]),     # 目标关节角度
}

# 典型采集频率：50Hz
# 单条轨迹长度：5-30秒 → 250-1500步
```

### ALOHA的优势

- **低成本**: 整套硬件约$20,000（对比工业遥操作系统$100,000+）
- **开源**: 硬件设计、软件栈、标定流程全部开源
- **双臂协调**: 天然支持需要双手配合的复杂任务（折叠、倾倒、装配）

---

## VR控制器数据收集 ⚙️进阶

使用VR控制器进行遥操作是DROID等大规模数据集的核心采集方式。

```python
# VR控制器遥操作伪代码
class VRTeleoperator:
    def __init__(self, robot, vr_device):
        self.robot = robot          # Franka Panda
        self.vr = vr_device         # Meta Quest Pro
        self.recorder = DataRecorder(fps=15)
    
    def collect_episode(self, task_instruction: str):
        """收集一条完整轨迹"""
        self.recorder.start(metadata={'instruction': task_instruction})
        
        while not self.is_episode_done():
            # 读取VR控制器位姿
            vr_pose = self.vr.get_controller_pose()  # (x,y,z,qx,qy,qz,qw)
            gripper = self.vr.get_trigger_value()     # [0, 1]
            
            # 坐标系映射：VR空间 → 机器人工作空间
            target_ee_pose = self.transform_vr_to_robot(vr_pose)
            action = np.concatenate([target_ee_pose, [gripper]])  # 7-DoF
            
            # 执行并记录
            self.robot.move_to(target_ee_pose)
            obs = self.robot.get_observation()  # 多视角图像 + 关节状态
            self.recorder.add_step(obs, action)
        
        self.recorder.save()
```

---

## 手持示教（Kinesthetic Teaching）🔰入门

**手持示教（Kinesthetic Teaching）** 是最直观的数据收集方式：操作者直接用手引导机器人完成任务动作。

- **优点**: 零额外设备成本、操作者门槛低、力觉自然
- **缺点**: 操作者需在机器人旁、采集速度慢、难以规模化
- **适用**: 小规模任务原型验证、精细力控任务

---

## 从人类视频提取动作 🔬深入

**人类视频动作提取（Action Extraction from Human Videos）** 是一种新兴的数据获取方式，旨在利用互联网上海量的人类操作视频作为机器人训练数据。

### 核心挑战

| 挑战 | 描述 | 当前方案 |
|------|------|---------|
| 具身体差异 | 人手 vs. 机械夹爪 | 关键点映射、手部姿态估计 |
| 缺少动作标注 | 视频只有图像无动作 | 逆动力学模型推断 |
| 视角差异 | 第一/第三人称混杂 | 视角不变特征学习 |
| 物理差异 | 抓取力学不同 | 域适应、仅学习高层语义 |

### 代表方法

```python
# 从人类视频提取伪动作标签（概念示意）
def extract_pseudo_actions(video_frames):
    """
    1. 手部检测与追踪
    2. 物体状态变化估计
    3. 逆运动学求解伪动作
    """
    hand_poses = hand_detector(video_frames)        # 检测手部关键点
    object_states = object_tracker(video_frames)    # 追踪物体位姿
    
    pseudo_actions = []
    for t in range(len(video_frames) - 1):
        # 通过相邻帧的状态变化推断动作
        delta_hand = hand_poses[t+1] - hand_poses[t]
        delta_obj = object_states[t+1] - object_states[t]
        pseudo_action = inverse_model(delta_hand, delta_obj)
        pseudo_actions.append(pseudo_action)
    
    return pseudo_actions
```

---

## 数据质量控制 ⚙️进阶

高质量数据对VLA训练至关重要。常见的质量控制策略包括：

### 采集阶段

- **操作者培训**: 标准化操作流程，减少个体差异
- **成功判定**: 自动检测任务完成标志（如物体到达目标位置）
- **实时反馈**: 可视化录制数据，操作者立即确认/重录

### 后处理阶段

```python
# 数据质量过滤流水线
def quality_filter_pipeline(episodes):
    filtered = []
    for ep in episodes:
        # 1. 长度检查：过短可能是误操作，过长可能是卡住
        if not (MIN_STEPS <= len(ep.steps) <= MAX_STEPS):
            continue
        
        # 2. 动作范围检查：异常大的动作可能是遥操作故障
        if np.any(np.abs(ep.actions) > ACTION_THRESHOLD):
            continue
        
        # 3. 图像质量：模糊/遮挡检测
        if any(is_blurry(step.image) for step in ep.steps):
            continue
        
        # 4. 成功标签验证
        if ep.metadata.get('success', True):
            filtered.append(ep)
    
    return filtered
```

### 标注增强

| 策略 | 说明 |
|------|------|
| 语言指令多样化 | 同一任务用不同表述（"pick up the cup" / "grab the mug"） |
| 自动标注 | 使用VLM为无标注轨迹生成语言描述 |
| 失败数据利用 | 标记失败轨迹用于对比学习或负样本训练 |
| 数据增强 | 图像裁剪/颜色抖动、动作噪声注入 |

---

## 数据收集成本对比

| 方法 | 每条轨迹成本 | 每小时产量 | 规模化潜力 | 数据质量 |
|------|-------------|-----------|-----------|---------|
| ALOHA遥操作 | ~$2-5 | 40-80条 | 中 | 高 |
| VR控制器 | ~$1-3 | 60-120条 | 高 | 中-高 |
| 手持示教 | ~$5-10 | 20-40条 | 低 | 高 |
| 人类视频提取 | ~$0.01 | 海量 | 极高 | 低 |
| 仿真生成 | ~$0.001 | 海量 | 极高 | 低(需迁移) |

---

## 本节小结

| 要点 | 说明 |
|------|------|
| 主流方式 | 遥操作是VLA数据收集的核心方法 |
| ALOHA | 低成本开源双臂遥操作系统，社区广泛采用 |
| 新兴方向 | 人类视频提取可大幅降低成本，但质量有待提高 |
| 质量控制 | 从采集到后处理的全流程过滤与增强不可或缺 |

---

> **下一节**: [4.3 预训练与微调范式](./03-pretrain-finetune) - 有了数据之后如何训练VLA？

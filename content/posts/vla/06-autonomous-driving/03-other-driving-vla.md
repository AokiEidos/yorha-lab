---
title: "其他驾驶 VLA 概述"
date: 2026-04-20T16:41:39.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 其他驾驶 VLA 概述

> ⚙️ 进阶 | 前置知识：[DriveVLM 详解](./02-drivevlm)、[动作头设计](../02-architecture/03-action-head)

## 概述

DriveVLM 之外，驾驶 VLA 领域涌现了多种不同思路的工作，各自在输入模态、推理方式、可解释性和评测方法上做出了差异化探索。本文详细介绍六个代表性模型：LMDrive、DriveGPT4、GPT-Driver、Dolphins、DriveLM 和 EMMA，并通过定量对比分析各方案的优劣。

## ⚙️ LMDrive：语言交互驾驶

### 核心思想

**LMDrive**（Shao et al., 2024）将自动驾驶建模为**语言条件闭环控制**（Language-Conditioned Closed-loop Control）——驾驶系统不仅接受静态导航指令，还能响应人类的实时语言反馈。

### 架构设计

```
多视角图像(3cam) + LiDAR点云(BEV) + 语言输入(导航+实时通知)
     │                    │                    │
     ▼                    ▼                    │
  CLIP ViT-L         PointPillar              │
  (视觉编码)          (点云编码)                │
     └────────┬───────────┘                    │
              ▼                                ▼
     ┌──────────────────────────────────────────────┐
     │  LLaMA-7B (多模态Token融合: 视觉+点云+文本)    │
     └────────────────────┬─────────────────────────┘
                          ▼
     控制输出头(MLP) → steering[-1,1] / throttle[0,1] / brake{0,1}
```

### 语言交互格式

LMDrive 定义了两类语言输入的统一处理：

```
[导航指令] (长期不变):
  "Follow the road and turn right at the next intersection."
  → 编码为 nav_token，在多个时间步持续有效

[实时通知] (随时出现):
  "There is a pedestrian crossing from the left side."
  → 编码为 notice_token，立即影响当前决策

[系统 Prompt]:
  "You are driving a vehicle. Based on the camera images,
   LiDAR data, navigation instruction, and real-time notice,
   predict the steering angle, throttle, and brake."
```

### CARLA 闭环实验结果

| 方法 | Driving Score↑ | Route Completion↑ | Infraction Score↑ | 语言响应 |
|------|---------------|-------------------|-------------------|---------|
| TransFuser | 54.5 | 78.4 | 0.69 | ❌ |
| InterFuser | 68.3 | 95.0 | 0.72 | ❌ |
| TCP | 75.1 | 89.6 | 0.84 | ❌ |
| LMDrive (无通知) | 71.2 | 91.3 | 0.78 | 部分 |
| LMDrive (有通知) | **76.1** | **93.2** | **0.82** | ✅ |

**分析**：
- 实时语言通知带来 +4.9 Driving Score 提升，证明语言通道确实能改善闭环驾驶
- 与 TCP 相当但增加了语言交互能力，是唯一支持人类实时语言反馈的闭环驾驶系统
- 在 CARLA Leaderboard Town05 Long benchmark 上评测

### 核心贡献与局限

**贡献**：首次实现语言条件闭环驾驶，统一导航+实时反馈为语言条件。
**局限**：仅在 CARLA 仿真验证；LiDAR 依赖增加传感器成本；语言指令需外部系统（如人类或上层 AI）提供。

## ⚙️ DriveGPT4：可解释驾驶决策

### 核心思想

**DriveGPT4**（Xu et al., 2024）强调**可解释性优先**——模型不仅做出驾驶决策，还必须用自然语言解释"为什么这样决策"。这回应了监管机构对自动驾驶可解释性的要求。

### 视频输入处理流程

与大多数驾驶 VLA 使用单帧或少量帧不同，DriveGPT4 处理**连续视频流**：

```
前视视频(30fps) → 时间采样(2fps) → ViT编码(每帧256 tokens)
  → 时间聚合器(8帧窗口, 4s上下文, temporal attention)
  → 视频特征(2048 tokens) + 文本prompt(~100 tokens)
  → LLM decoder → 动作 + 解释
```

### 解释生成示例

```
[输入视频]: 城市道路，前方卡车停靠在路边，部分占据车道

模型输出:
{
  "action": {
    "steering": -0.15,    // 轻微左转
    "throttle": 0.25,     // 低速通过
    "brake": 0.0
  },
  "explanation": "I observe a large truck parked on the right side of
    the road, partially blocking the current lane. The truck appears
    to be a delivery vehicle with its hazard lights on. The oncoming
    lane is clear of traffic. I am steering slightly to the left to
    safely pass the parked truck while maintaining a safe lateral
    distance of approximately 1.5 meters. I have reduced speed as
    a precaution in case the truck door opens or a person steps out."
}
```

### 评测指标

DriveGPT4 设计了解释质量的专项评测：

| 评测维度 | 指标 | DriveGPT4 得分 | 基线 (BDD-X caption) |
|---------|------|---------------|---------------------|
| 语言质量 | BLEU-4 | 12.8 | 9.4 |
| 语义相关性 | CIDEr | 89.3 | 62.1 |
| 决策覆盖率 | Action Coverage | 84.2% | — |
| 因果准确率 | Causal Accuracy | 76.5% | — |
| 人类评分 | 1-5 scale | 3.8 | 2.9 |

**核心局限**：仅有前视单目输入，不支持环视；解释生成增加延迟（额外 200-400ms）；无闭环驾驶评测。

## ⚙️ GPT-Driver：LLM 作为运动规划器

### 核心思想

**GPT-Driver**（Mao et al., 2023）是一个激进但有启发性的方案——**完全不使用视觉输入**，将 3D 感知结果转化为结构化文本，让 GPT-3.5 直接输出轨迹坐标。

### 文本化场景表示格式

```
输入文本:
  [Ego State] Position:(0,0), Heading:0.12rad, Vel:(8.52,0.31)m/s
  [Nearby Agents]
    Agent_1: Car at (12.3,-1.8), vel=(7.1,0.1)
    Agent_3: Pedestrian at (18.2,5.1), vel=(0.8,-0.5)
  [Map] Lane_3, width=3.5m, limit=50km/h, left_available=True
  [Command] "Continue forward, maintain lane."

GPT-3.5 输出:
  [Trajectory] (0,0),(0.02,4.25),(0.03,8.48),(0.01,12.65),(0,16.78),(-0.01,20.85)
  [Reasoning] "Maintaining lane. Agent_1 adjacent, no conflict. Monitoring pedestrian."
```

### nuScenes 规划结果

| 方法 | L2 1s(m) | L2 2s(m) | L2 3s(m) | 碰撞率(%) | 输入类型 |
|------|---------|---------|---------|----------|---------|
| NMP | 2.31 | 4.27 | 6.15 | 0.38 | LiDAR |
| FF | 0.55 | 1.20 | 2.04 | 0.14 | Camera |
| EO | 0.67 | 1.36 | 2.13 | 0.04 | Camera |
| GPT-Driver (zero-shot) | 2.08 | 3.15 | 4.52 | 0.45 | Text |
| GPT-Driver (fine-tuned) | 0.61 | 1.08 | 1.78 | 0.08 | Text |

**分析**：
- Zero-shot 性能差（L2 3s = 4.52m），说明 GPT-3.5 的通用知识不足以直接规划
- Fine-tuned 后显著改善，接近专用模型水平，证明 LLM 可以学会运动规划
- 根本局限：依赖完美的上游感知（GT 检测框），真实部署时感知误差会被放大

## ⚙️ Dolphins：多粒度视觉语言驾驶

### 核心思想

**Dolphins**（Ma et al., 2023）提出**多粒度视觉-语言对齐（Multi-Granularity VL Alignment）**，在不同粒度（场景级、物体级、动作级）上建立视觉与语言的对应关系。

### 架构特点

```
┌────────────────────────────────────────┐
│          Dolphins 多粒度架构             │
│                                         │
│  视觉输入 → ViT 编码器                   │
│       │                                 │
│       ├── 场景级特征 (全局 CLS token)     │
│       │   → "高速公路直行场景"             │
│       │                                 │
│       ├── 物体级特征 (RoI Align)          │
│       │   → "前方卡车" "右侧摩托车"        │
│       │                                 │
│       └── 动作级特征 (时序差分)            │
│           → "卡车正在减速" "摩托车加速"     │
│                                         │
│  三级特征 → 多粒度 Q-Former → LLM        │
│       → 多层次理解 + 动作生成             │
└────────────────────────────────────────┘
```

**关键结果**：在 BDD-X 数据集上，解释生成 CIDEr 达到 95.7（DriveGPT4: 89.3）；动作预测 MAE 降低 12%。

## ⚙️ DriveLM：图谱引导的驾驶推理

### 核心思想

**DriveLM**（Sima et al., 2023, Waymo & Shanghai AI Lab）构建了基于**图谱结构的 QA 数据集**——每个驾驶场景标注为"感知→预测→规划"三级 QA 链，形成推理图谱（Graph of QA）。

### QA 图谱结构

每个场景标注为三级 QA 链，节点间有因果边连接形成推理图谱：

```
[感知] Q:"What objects in front-left?" → A:"White sedan at 30m, cone at 15m"
  ↓
[预测] Q:"What will the sedan do?" → A:"Signaling left, likely merging into ego lane"
  ↓
[规划] Q:"What should ego do?" → A:"Decelerate to yield. Target: 35km/h"
```

### 数据集规模

| 数据集 | 场景数 | QA 对数 | 图谱节点数 | 数据来源 |
|--------|-------|---------|-----------|---------|
| DriveLM-nuScenes | 4,871 | 443K | 598K | nuScenes 关键帧 |
| DriveLM-CARLA | 2,000 | 320K | 410K | CARLA 仿真生成 |

### 基准评测结果

| 模型 | 感知 QA 准确率 | 预测 QA 准确率 | 规划 QA 准确率 | 整体 DriveLM Score |
|------|--------------|--------------|--------------|-------------------|
| LLaVA-7B | 58.3% | 42.1% | 35.7% | 45.4 |
| InstructBLIP | 62.7% | 48.5% | 41.2% | 50.8 |
| DriveLM-Agent | **71.2%** | **59.8%** | **52.3%** | **61.1** |

## 🔬 EMMA：端到端多模态自动驾驶

### 核心思想

**EMMA**（Hwang et al., 2024, Waymo）基于 Gemini 多模态大模型，将自动驾驶的多个任务（检测、预测、规划）统一为**文本生成任务**。

### 架构设计

```
EMMA 不引入新模块，直接利用 Gemini 的多模态能力:

输入: 前视相机图像序列 (最近 2s, 采样 4 帧)
     + 结构化场景 prompt
     + 任务指令 ("Plan the ego trajectory for next 3s")

Gemini 处理:
  [图像 tokens] + [文本 tokens] → Transformer → [输出 tokens]

输出: 文本化的轨迹坐标
  "Trajectory: (0.0,0.0), (0.3,4.2), (0.5,8.5),
   (0.4,12.8), (0.2,17.1), (0.1,21.3)"
```

### Waymo Open Motion 结果

| 方法 | minADE(m) | minFDE(m) | MissRate(%) |
|------|----------|----------|-------------|
| MTR | 0.60 | 1.22 | 13.4 |
| MotionLM | 0.56 | 1.15 | 12.8 |
| EMMA (Gemini-based) | **0.54** | **1.08** | **11.2** |

**核心洞察**：EMMA 证明足够强大的基础模型（Gemini）可以在不引入领域特定架构的情况下，仅通过 prompt 工程和微调实现竞争力的驾驶性能。但其依赖于 Gemini 的闭源大模型，可复现性和车端部署是问题。

## 🔬 综合对比

### 定量对比表

| 方法 | 年份 | 输入模态 | VLM 主干 | 参数量 | 核心创新 | 可解释 | 闭环评测 | 评测平台 |
|------|------|---------|---------|-------|---------|--------|---------|---------|
| DriveVLM | 2024 | 多视角图像 | InternVL-26B | ~26B | CoT 三阶段推理 | ✅ | ❌ | nuScenes |
| LMDrive | 2024 | 图像+LiDAR+语言 | LLaMA-7B | ~7B | 语言交互驾驶 | ✅ | ✅ CARLA | CARLA |
| DriveGPT4 | 2024 | 前视视频 | LLaMA-7B | ~7B | 决策+解释生成 | ✅ 强 | ❌ | BDD-X |
| GPT-Driver | 2023 | 结构化文本 | GPT-3.5 | ~175B | LLM 直接规划 | ✅ | ❌ | nuScenes |
| Dolphins | 2023 | 前视视频 | LLaMA-7B | ~7B | 多粒度对齐 | ✅ | ❌ | BDD-X |
| DriveLM | 2023 | 多视角图像 | 多种 VLM | 变化 | QA 图谱推理 | ✅ | ❌ | nuScenes |
| EMMA | 2024 | 前视图像序列 | Gemini | 未公开 | 统一文本生成 | ✅ | ❌ | Waymo |

### 技术路线演进

```
2023 ─── GPT-Driver: 文本化场景 → LLM 规划 (概念验证)
  │      Dolphins: 多粒度视觉语言对齐
  │      DriveLM: 结构化 QA 图谱
  │
2024 ─── DriveVLM: CoT 三阶段推理 (系统化方案)
  │      LMDrive: 语言交互闭环驾驶 (闭环突破)
  │      DriveGPT4: 可解释性优先
  │      EMMA: 基础模型直接驾驶 (Gemini)
  │
2025+ ── 趋势: Dual 架构 + 更强基础模型 + 闭环验证
```

## 小结

| 概念 | 要点 |
|------|------|
| LMDrive | 唯一实现 CARLA 闭环的语言交互驾驶系统，DS=76.1 |
| DriveGPT4 | 可解释性最强，CIDEr=89.3，但无闭环评测 |
| GPT-Driver | 证明 LLM 可学习运动规划，但依赖完美感知 |
| Dolphins | 多粒度视觉语言对齐，BDD-X CIDEr=95.7 |
| DriveLM | 首个大规模驾驶 QA 图谱，推动评测标准化 |
| EMMA | Gemini 直接驾驶，证明强基础模型的跨任务能力 |

---

> **下一篇**：[场景理解+决策一体化](./04-scene-understanding) — 深入分析可解释性与安全验证

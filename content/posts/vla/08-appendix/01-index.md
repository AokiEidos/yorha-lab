---
title: "VLA 技术文档系列 · 目录索引"
date: 2026-04-20T17:02:05.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# VLA 技术文档系列 · 目录索引

## 模块一：前置基础

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 1.1 | [VLM 回顾](../01-foundations/01-vlm-review) | 🔰 | CLIP/SigLIP/DINOv2 对比、VLM 三组件（视觉编码器+投影层+LLM）、视觉 Token 化流程、代表性 VLM（LLaVA/PaliGemma/InternVL） |
| 1.2 | [具身感知基础](../01-foundations/02-embodied-perception) | 🔰 | RGB 相机选型（RealSense D435i 详细规格）、5 种深度传感技术对比、点云处理管线（PointNet）、本体感觉编码、5 种触觉传感器对比（GelSight/DIGIT/BioTac）、多模态融合架构 |
| 1.3 | [观测空间与动作空间](../01-foundations/03-observation-action-space) | 🔰⚙️ | Franka 7 关节详细规格表、笛卡尔 vs 关节空间 + IK 解释、256-bin 量化公式与精度分析、Action Chunking 时序图与数学动机、控制频率 5 层层级架构、9 种机器人平台配置表 |
| 1.4 | [模仿学习基础](../01-foundations/04-imitation-learning) | 🔰⚙️ | BC 损失推导（MSE + CE）、Ross & Bagnell $O(T^2\epsilon)$ 定理与证明、DAgger 伪代码、IBC 能量模型、7 种 BC 变体对比（含 robomimic benchmark）、真实世界训练指南（数据量/增强策略） |
| 1.5 | [强化学习基础](../01-foundations/05-reinforcement-learning) | 🔰⚙️ | 完整 MDP 数学（Bellman 方程）、REINFORCE 伪代码、抓取/插入奖励函数代码示例、Reward Hacking 案例表、Isaac Gym PPO Sim-to-Real 管线与 benchmark、RLHF for Robots 设计、IL vs RL vs 混合方法 7 维对比 |

## 模块二：核心架构

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 2.1 | [从 VLM 到 VLA 的演化](../02-architecture/01-vlm-to-vla) | 🔰⚙️ | 5 层知识迁移分析（视觉识别→因果推理→任务常识）、RT-2 涌现验证、泛化能力与模型规模关系图、VLM→VLA 代码示例（动作 Token 化+词表扩展+Co-fine-tuning）、三种路线全面对比（含计算需求） |
| 2.2 | [动作表示与 Token 化](../02-architecture/02-action-representation) | ⚙️ | 离散 Token（256-bin 量化公式+精度分析）vs 连续回归（MLP 多模态问题图解）vs Diffusion/FM（多峰建模原理）三路线深度对比 |
| 2.3 | [动作解码头设计](../02-architecture/03-action-head) | ⚙️ | 自回归/MLP/Diffusion/FM 四种解码头 PyTorch 伪代码、Action Chunking 时序集成公式、典型 Chunk Size 对比表、推理延迟实测对比 |
| 2.4 | [多模态输入融合](../02-architecture/04-multimodal-fusion) | ⚙️ | 早期 vs 晚期融合详解、FiLM/Cross-Attention/Perceiver Resampler 代码、本体感觉 4 种注入方式对比、历史帧处理策略、8 个模型融合策略一览表、Token 数量 vs 计算量分析 |
| 2.5 | [架构对比表](../02-architecture/05-architecture-comparison) | 🔰⚙️ | 10 模型全维度对比（架构/性能/部署）、Benchmark 成功率表、4 种设计模式详解（VLM 直出/独立解码头/双系统/无 LLM 轻量）、决策树选型指南、4 种部署场景推荐、参数效率分析 |

## 模块三：主流模型详解

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 3.1 | [SayCan](../03-models-zoo/01-saycan) | 🔰⚙️ | Affordance 评分公式（乘法原理）、完整 6 步任务分解示例（含评分表）、内循环 Python 伪代码、551 技能库分析、vs CaP/ProgPrompt/Inner Monologue 5 维对比、4 种失败模式 |
| 3.2 | [RT-1](../03-models-zoo/02-rt1) | ⚙️ | 完整架构图（EfficientNet→FiLM→TokenLearner→Transformer）、TokenLearner 机制伪代码（100× 注意力压缩）、训练超参表、700+ 任务分布分析、数据 Scaling 消融、vs BC-Z/Gato 对比 |
| 3.3 | [RT-2](../03-models-zoo/03-rt2) | ⚙️🔬 | PaLI-X(55B) vs PaLM-E(12B) 详细对比、动作 Token 化方案（bin 范围/Token ID 映射公式）、Co-fine-tuning 训练配方（伪代码+超参+数据混合比例影响）、5 类涌现能力详解（含成功率数据）、定量结果表 |
| 3.4 | [Octo](../03-models-zoo/04-octo) | ⚙️ | 完整架构图（Octo-S 27M / Octo-B 93M）、Readout Token 注意力掩码机制详解、Diffusion vs MLP 头对比（含伪代码+实验结果）、微调 API 代码、跨具身体结果表、训练计算详情 |
| 3.5 | [OpenVLA](../03-models-zoo/05-openvla) | ⚙️ | Prismatic 双编码器融合伪代码（SigLIP+DINOv2 互补性验证）、量化/反量化公式、完整训练配方（64×A100, 14 天）、LoRA vs 全量微调对比、HuggingFace 推理+微调代码、跨机器人 benchmark |
| 3.6 | [π₀](../03-models-zoo/06-pi-zero) | ⚙️🔬 | Flow Matching 训练目标数学推导、Action Expert 架构（~500M 额外参数）、两阶段训练配方（256×H100 预训练→微调）、跨具身体数据混合表（7+ 机器人）、ALOHA benchmark 6 任务结果、Diffusion vs FM 速度/质量消融、π₀-FAST 对比、6 种失败模式 |
| 3.7 | [GR 系列](../03-models-zoo/07-gr-series) | ⚙️ | MAE 视频预训练详解（75% 掩码+时空掩码）、完整架构图（GPT-style Transformer 24 层）、联合损失公式（$L_\text{video} + \lambda L_\text{action}$）、GR-2 扩展（3B 参数/FVD 142）、人形 31-DoF 控制详情、vs UniPi/SuSIE 视频预测基线对比 |
| 3.8 | [ALOHA 与 ACT](../03-models-zoo/08-aloha-act) | ⚙️ | 硬件规格表（$21K 成本明细）、ACT CVAE 完整架构图（编码器+解码器）、CVAE 损失公式（L1 重建 + KL）推导、时间集成公式（指数加权）、PyTorch 伪代码、ALOHA 2 改进、5 任务 4 基线成功率对比、消融实验 |
| 3.9 | [Helix](../03-models-zoo/09-helix) | ⚙️🔬 | 频率需求谱分析（为什么需要双系统）、System 1/2 详细规格、系统间通信协议（GoalVector 伪代码）、三级安全机制、vs GR00T N1 10 维对比、Figure 02 硬件规格、BMW 工厂部署案例（性能指标） |
| 3.10 | [GR00T N1](../03-models-zoo/10-groot-n1) | ⚙️🔬 | System 1 Diffusion Transformer 推理伪代码、vs Helix 深度对比（6 维）、Isaac Sim 数据生成管线（域随机化参数表）、仿真规模数据、Sim-to-Real 迁移结果、跨具身体 Adapter 架构、NVIDIA 硬件栈（Jetson Thor） |
| 3.11 | [Gemini Robotics](../03-models-zoo/11-gemini-robotics) | ⚙️🔬 | RT-1→RT-2→Gemini 演化对比、多模态输入架构、推理-动作联合输出示例、vs RT-2 定量对比（6 类任务）、4 层安全机制（含安全拒绝示例）、ER vs NT 对比、灵巧操作演示（拆箱/折纸/工具）、专用 VLA vs 通用大模型路线之争 |
| 3.12 | [其他模型](../03-models-zoo/12-other-models) | ⚙️ | QUAR-VLA（12-DoF 四足架构+运动结果）、RoboFlamingo（Perceiver+Cross-Attention 历史注入+CALVIN benchmark）、LLARVA（2D 视觉轨迹两阶段管线+跨机器人迁移 +30%）、3D-VLA（PointNet++ 点云+3D 世界模型）、SpatialVLA、RoboVLM |
| 3.13 | [演化关系图](../03-models-zoo/13-evolution-map) | 🔰⚙️ | Mermaid 全景图（带会议/年份标注）、里程碑时间线表（机构+贡献）、三条技术主线详解（动作表示/架构/数据）、技术谱系（代码/数据继承关系）、学术 vs 工业路线（含融资数据）、开放问题与收敛趋势 |

## 模块四：训练技术

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 4.1 | [大规模数据集](../04-training/01-datasets) | ⚙️ | Open X-Embodiment（~100 万轨迹/22 机器人）、DROID、RH20T、Bridge V2 详细规格、RLDS 数据格式、数据 Scaling 策略 |
| 4.2 | [数据收集](../04-training/02-data-collection) | ⚙️ | 遥操作 5 种方式对比、ALOHA 双臂系统案例（成本/数据格式）、VR 控制器、手持示教、人类视频动作提取、数据质量管线 |
| 4.3 | [预训练→微调](../04-training/03-pretrain-finetune) | ⚙️ | VLM 预训练→机器人微调两阶段、Co-fine-tuning 配方（数据混合比例影响）、冻结策略选择、超参参考表 |
| 4.4 | [跨具身体迁移](../04-training/04-cross-embodiment) | ⚙️🔬 | 统一动作空间设计、具身体 Adapter 架构、Octo/RT-2-X/π₀ 跨具身体实验分析 |
| 4.5 | [Sim-to-Real](../04-training/05-sim-to-real) | ⚙️ | 域随机化参数表、Isaac Sim/MuJoCo/SAPIEN 3 大仿真器对比、VLM 作为域桥接 |
| 4.6 | [高效微调](../04-training/06-efficient-finetuning) | ⚙️ | LoRA/QLoRA 配置、Adapter 层、Prompt Tuning、参数效率对比表、选型决策树 |
| 4.7 | [开源工具链](../04-training/07-toolchains) | ⚙️ | LeRobot 架构/数据格式（LeRobotDataset）/支持策略（ACT/Diffusion Policy/TDMPC）/训练流程/硬件支持、robomimic、DROID 工具链、快速上手指南 |

## 模块五：机器人应用

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 5.1 | [桌面操作](../05-robotics/01-tabletop) | ⚙️ | 3 维泛化分析（新物体/新指令/新场景）、RT-1/RT-2/Octo/OpenVLA 定量对比、传统流水线 vs VLA 混合架构趋势 |
| 5.2 | [灵巧手操作](../05-robotics/02-dexterous) | ⚙️🔬 | 灵巧手平台对比（Shadow/Allegro/LEAP）、20+ DoF 挑战、触觉 VLA 融合架构、π₀ 折衣/装配突破、π₀-FAST、数据收集 5 种方案对比 |
| 5.3 | [移动操作](../05-robotics/03-mobile) | ⚙️ | 导航+操作复合空间（9-37 DoF）、SayCan Say×Can 评分公式、NaVILA 统一 VLA、Mobile ALOHA Co-training |
| 5.4 | [人形机器人](../05-robotics/04-humanoid) | ⚙️🔬 | 30-50 DoF 全身控制、三种路线（分层/统一/双系统）、Helix System 1+2 vs GR00T N1 System 1+2 详解、GR 系列演化、Figure AI/NVIDIA/1X/Tesla/Unitree 等厂商对比 |
| 5.5 | [长时域规划](../05-robotics/05-long-horizon) | ⚙️ | 三层规划架构（任务/技能/动作）、6 种 LLM-as-Planner 框架对比、Inner Monologue 闭环反馈机制（对话示例）、泛化挑战 4 类分析、前沿方向（VLM 直接规划/世界模型/自动技能发现） |

## 模块六：自动驾驶

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 6.1 | [VLA 驾驶范式](../06-autonomous-driving/01-vla-driving-paradigm) | ⚙️ | 模块化(Apollo) vs 端到端(UniAD/VAD) vs VLA(DriveVLM) 三路线架构图+延迟分析、nuScenes 规划 benchmark、5 大 VLA 独特能力+场景实例、多相机 BEV vs 单图输入对比、RSS 安全分层架构、评测基准总览 |
| 6.2 | [DriveVLM](../06-autonomous-driving/02-drivevlm) | ⚙️ | InternVL-26B 主干详情、输入输出 XML 格式、CoT 三阶段深度解析、3 个完整推理示例（高速障碍/路口行人/夜间雨天汇入）、DriveVLM-Dual 融合架构图、nuScenes 定量结果、长尾场景评估（+24.8%）、延迟分解 |
| 6.3 | [其他驾驶 VLA](../06-autonomous-driving/03-other-driving-vla) | ⚙️ | 6 模型详解：LMDrive（CARLA 闭环 DS=76.1）、DriveGPT4（视频处理管线+解释质量指标）、GPT-Driver（文本化场景表示格式）、Dolphins（多粒度架构 CIDEr=95.7）、DriveLM（443K QA 对）、EMMA（Gemini-based Waymo 结果）；7 模型对比表 |
| 6.4 | [场景理解+决策](../06-autonomous-driving/04-scene-understanding) | ⚙️🔬 | 4 级能力矩阵（L1-L4）、3 个完整场景示例（施工/学校/事故）、解释质量评测方法表（自动/因果/人类）、人类研究结果（50 人 200 场景）、幻觉 5 类分类+发生率、对抗鲁棒性分析、不确定性量化方法、监管法规景观（EU AI Act/UNECE/ISO）、信任校准研究 |
| 6.5 | [局限与展望](../06-autonomous-driving/05-limitations) | 🔬 | 9 配置延迟 benchmark、安全响应时间预算、加速技术路线表、VLA vs 传统规划器安全保证 6 维对比、失效率量化分析、Sim-to-Real gap 4 维分析、监管时间线（2023-2027+）、双系统驾驶架构提案（含架构图）、5 阶段技术路线图 |

## 模块七：大模型融合

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 7.1 | [LLM 推理引擎](../07-foundation-models/01-llm-reasoning) | ⚙️ | 多层级任务分解示例、Code-as-Policies 代码示例、6 类常识推理对比表、CoT 3 种形式（自然语言/结构化/Inner Monologue 闭环对话）、延迟 benchmark（7B-70B×多硬件）、4 种缓解策略详解（异步/缓存/蒸馏/双系统）、5 种局限分析 |
| 7.2 | [VLM 感知骨干](../07-foundation-models/02-vlm-backbone) | ⚙️ | CLIP/SigLIP/DINOv2/EVA-CLIP 特性对比、语义 vs 空间权衡分析、OpenVLA 双编码器实验验证（成功率表）、分辨率 vs Token 数 vs 计算量分析、TokenLearner/Perceiver 压缩代码、多视角 3 种处理策略、选型决策树 |
| 7.3 | [Diffusion 动作生成](../07-foundation-models/03-diffusion-action) | ⚙️🔬 | 多模态动作问题图解、Diffusion Policy 架构（CNN vs Transformer）、训练/推理 PyTorch 伪代码、robomimic 5 任务 benchmark、Flow Matching 训练目标公式、π₀ Action Expert 交织 Transformer 设计、Diffusion vs FM vs 自回归延迟实测对比 |
| 7.4 | [具身 Agent](../07-foundation-models/04-embodied-agent) | 🔬 | 完整感知-推理-行动循环架构（含时序分析表）、三类记忆详细设计（工作/情景 RAG/语义+技能库）、3 个工具使用场景（物理测量/API 调用/异常重规划）、4 种多 Agent 任务分配策略、SayCan/Voyager/DEPS/RoboAgent 框架分析、VLA 6 层全栈定位 |
| 7.5 | [世界模型](../07-foundation-models/05-world-model) | 🔬 | 状态转移模型形式化定义、视频预测架构与损失函数、UniSim 统一动作接口（3 粒度）、Genie 3 组件架构（11B/200K hr）、MCTS 类比想象规划（含计算分析）、GR 双任务架构、6 方法对比表、计算成本分析、5 类失败模式 |
| 7.6 | [未来展望](../07-foundation-models/06-future) | 🔬 | 7 模型 Scaling Laws 实证数据表（RT-1 35M→π₀-FAST 3B）、4 个关键发现、3 阶段通用基础模型路线图、实时推理 4 条加速路线（投机解码/蒸馏/量化/双系统+具体数据）、三层安全架构+RLHF for Robots 设计、数据飞轮机制+Tesla FSD 对比、9 家公司商业格局（融资数据）、3 阶段商业化时间线 |

---

## 推荐阅读路径

### 🔰 入门路径（~12 篇，建立核心直觉）
1.1 → 1.2 → 1.3 → 1.4 → 2.1 → 2.5 → 3.1 → 3.3 → 3.5 → 3.8 → 3.13 → 7.6

### 🤖 机器人工程师路径（~30 篇，掌握实用技术）
入门路径 + 2.2 → 2.3 → 2.4 → 3.4 → 3.6 → 3.8 → 4.1-4.7 → 5.1-5.5 → 7.3

### 🚗 自动驾驶路径（~15 篇）
1.1 → 1.3 → 1.4 → 2.1 → 2.5 → 3.3 → 6.1-6.5 → 7.1 → 7.5

### 🔬 研究者路径（全部 49 篇）
按顺序阅读全部内容

### 🏭 人形机器人路径（~18 篇）
1.1-1.5 → 2.1-2.5 → 3.7 → 3.8 → 3.9 → 3.10 → 4.4 → 4.5 → 5.4 → 7.5 → 7.6

### ⚡ π₀ 深度理解路径（~10 篇）
1.4 → 2.2 → 2.3 → 3.5 → 3.6 → 3.8 → 4.4 → 5.2 → 7.3

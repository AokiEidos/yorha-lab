---
title: "术语表 (Glossary)"
date: 2026-04-20T17:04:13.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# 术语表 (Glossary)

> 按首字母排序，包含全系列出现的专业术语。格式：**中文名（English Name）** — 解释

---

## A
- **ACT（Action Chunking with Transformers）** — ALOHA 配套策略，用 CVAE+Transformer 一次预测 100 步动作序列
- **Action Chunking（动作分块）** — 一次预测未来多步动作序列而非单步，减少推理频率并保持时间一致性。典型 chunk size：ACT 100 步、π₀ 50 步、Diffusion Policy 16 步
- **Action Expert** — π₀ 中与 VLM Transformer 层交织的独立动作生成网络（~500M 参数），负责 Flow Matching 去噪
- **Action Head（动作解码头）** — VLA 中将 VLM 特征转化为机器人动作的模块，主要有自回归 Token、MLP 回归、Diffusion、Flow Matching 四种
- **Action Space（动作空间）** — 机器人能执行的所有动作的集合
- **adaLN-Zero** — Diffusion Transformer 中的条件注入机制，用条件信息调制 LayerNorm 的 scale/shift，gate 初始化为零
- **Affordance Function（可行性函数）** — SayCan 中评估"当前环境下某技能成功执行概率"的函数
- **ALOHA** — Stanford/Google DeepMind 的低成本（~$21K）双臂遥操作系统，4 个 ViperX 300 机械臂 + 4 相机
- **ASIL（Automotive Safety Integrity Level）** — ISO 26262 中的汽车安全完整性等级（A-D）

## B
- **BC（Behavior Cloning，行为克隆）** — 从专家演示中直接学习观测→动作映射的监督学习方法。连续动作用 MSE 损失，离散 Token 用交叉熵损失
- **BC-RNN** — 带循环神经网络的行为克隆变体，用 LSTM/GRU 维护时序隐状态
- **Bellman 方程** — RL 的核心递推关系：$V^\pi(s) = \mathbb{E}_{a \sim \pi}[R(s,a) + \gamma V^\pi(s')]$
- **BEV（Bird's Eye View，鸟瞰图）** — 从上方俯视的 2D 表示，自动驾驶中的核心空间表示
- **BioTac** — SynTouch 的仿生触觉传感器（液压+电极阵列，19 通道，~$5000）

## C
- **Cartesian Space（笛卡尔空间/任务空间）** — 以末端执行器的位置(xyz)和朝向(roll/pitch/yaw)来表示动作，VLA 主流选择
- **CLIP** — OpenAI 的对比语言-图像预训练模型，通过图文对比学习对齐视觉和语言空间
- **Co-fine-tuning（共同微调）** — 同时在机器人数据和原始 VLM 数据上微调以防灾难性遗忘。RT-2 使用 50:50 混合比例
- **Code-as-Policies (CaP)** — 让 LLM 直接生成 Python 控制代码来操作机器人的范式
- **CoT（Chain-of-Thought，思维链）** — 模型在输出动作前先生成中间推理步骤。DriveVLM 使用三阶段 CoT：场景描述→分析→规划
- **Cross-Embodiment Transfer（跨具身体迁移）** — 在不同类型机器人间迁移学习到的技能。Octo/π₀ 验证了跨具身体联合训练的有效性
- **CVAE（Conditional VAE，条件变分自编码器）** — 用潜变量 z 编码动作多模态性的生成模型，ACT 使用

## D
- **DAgger（Dataset Aggregation，数据集聚合）** — 通过迭代数据收集缓解分布偏移的算法，将误差上界从 $O(T^2\epsilon)$ 降至 $O(T\epsilon)$
- **Dexterous Hand（灵巧手）** — 具有多个手指和关节（16-24 DoF）的仿人手型末端执行器（如 Shadow Hand、LEAP Hand）
- **DIGIT** — Meta AI 的低成本弹性凝胶触觉传感器（~$150，240×320 RGB 输出，60fps）
- **Diffusion Policy** — Chi et al. 2023，用条件扩散模型（1D U-Net 或 Transformer）从噪声中去噪生成动作序列
- **Diffusion Transformer (DiT)** — 用 Transformer 替代 U-Net 作为扩散模型的去噪骨干。GR00T N1 的 System 1 使用
- **DINOv2** — Meta 的自监督视觉编码器（ViT 架构），擅长空间细节（边界、深度线索），OpenVLA 双编码器之一
- **Distribution Shift（分布偏移）** — BC 的核心缺陷：推理时策略的微小误差导致实际状态偏离训练分布，误差以 $O(T^2)$ 速率累积
- **Domain Randomization（域随机化）** — 在仿真中随机变化视觉/物理参数以提升 Sim-to-Real 泛化
- **DriveLM** — QA 图结构的驾驶推理数据集（443K QA 对），支持训练可解释驾驶 VLA
- **DriveVLM** — Li et al. 2024，用 InternVL-26B 进行三阶段 CoT 驾驶推理（场景描述→分析→规划）
- **Dual-System Architecture（双系统架构）** — 受认知科学启发的快-慢分离设计：System 1（快速运动控制 30-200Hz）+ System 2（慢速 VLM 推理 1-5Hz）。Helix 和 GR00T N1 采用

## E
- **EBM（Energy-Based Model，能量模型）** — 通过能量函数定义概率分布的模型，IBC 使用此方法建模多模态动作
- **Embodied AI（具身智能）** — 在物理环境中具有身体并能行动的 AI 系统
- **EMMA** — Google 基于 Gemini 的端到端多模态驾驶模型
- **End-Effector（末端执行器）** — 机械臂末端的手/工具/夹爪
- **EVA-CLIP** — 大规模增强版 CLIP（ViT-G 1B 参数），视觉特征更强

## F
- **FAST（Fast Action STructure）** — π₀-FAST 中用于加速动作 Token 生成的方法
- **FiLM（Feature-wise Linear Modulation，特征级线性调制）** — 用条件特征的 γ/β 调制目标特征，RT-1 和 Octo 使用
- **Flow Matching（流匹配）** — Diffusion 的替代方案，用直线路径连接噪声和数据。训练目标预测速度 $v=\epsilon-x_0$，采样比 Diffusion 更快。π₀ 使用
- **F/T Sensor（力/力矩传感器）** — 测量机器人末端受到的 6 维力和力矩（如 ATI Mini45: ±580N/±10Nm）

## G
- **GelSight** — MIT 的弹性凝胶触觉传感器（光度立体法重建 3D 接触面，~25μm 分辨率）
- **Gemini Robotics** — Google DeepMind 2025，将 Gemini 2.0 直接扩展到机器人控制，含 ER（推理）和 NT（导航）两子系统
- **Genie** — Google DeepMind 的可控世界模型（11B 参数，从 200K hr 视频中无监督学习 8 个离散潜动作）
- **GR00T N1** — NVIDIA 2025 的人形机器人基础模型，System 1 用 Diffusion Transformer，System 2 用 VLM，深度绑定 Isaac Sim
- **GR 系列（GR-1/GR-2）** — 字节跳动的人形机器人模型。GR-1: 330M 参数 MAE 视频预训练；GR-2: 3B 参数，潜空间预测，FVD=142

## H
- **Helix** — Figure AI 2025 的人形机器人 VLA 系统，双系统架构，已在 BMW 工厂试点部署

## I
- **IBC（Implicit Behavior Cloning，隐式行为克隆）** — 用能量函数 $E_\theta(o,a)$ 替代显式策略网络的 BC 变体，天然处理多峰动作分布
- **IK（Inverse Kinematics，逆运动学）** — 从末端位姿计算关节角度的求解过程（可能多解或无解）
- **IL（Imitation Learning，模仿学习）** — 从演示中学习的方法总称，VLA 的主训练范式
- **Inner Monologue（内部独白）** — Huang et al. 2022，让机器人在执行过程中持续自我对话并根据反馈调整计划的闭环推理机制
- **Isaac Sim** — NVIDIA 的 GPU 加速机器人仿真器，支持 1024+ 并行环境、Ray Tracing 渲染、PhysX 5.0 物理引擎

## J
- **Joint Space（关节空间）** — 以各关节角度/角速度来表示动作。Franka 7-DoF 关节范围约 ±166°

## K
- **Kinesthetic Teaching（手持示教）** — 人类直接移动机器人手臂来录制演示

## L
- **LeRobot** — HuggingFace 的开源机器人学习框架，支持 ACT/Diffusion Policy/TDMPC 策略和 LeRobotDataset 格式
- **LLARVA** — 用 2D 视觉轨迹作为中间表示的 VLA：VLM 预测图像空间轨迹→几何转换为 3D 动作
- **LLM（Large Language Model，大语言模型）** — 大规模预训练的语言模型（如 GPT-4、Llama、Gemini）
- **LMDrive** — 语言条件闭环驾驶系统（LLaMA-7B + CLIP + PointPillar），CARLA Driving Score=76.1
- **LoRA（Low-Rank Adaptation，低秩适配）** — 用低秩矩阵 $\Delta W = BA$（$r \ll d$）近似权重更新的高效微调方法

## M
- **MAE（Masked Autoencoder，掩码自编码器）** — 自监督预训练方法，掩码 75% 的 patch 后重建。GR-1 用视频 MAE 预训练
- **MDP（Markov Decision Process，马尔可夫决策过程）** — RL 的数学框架：$(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$
- **Mobile Manipulation（移动操作）** — 结合导航和操作的任务，需要底盘+机械臂复合动作空间
- **Mode Averaging（模式平均化）** — MLP 回归输出多峰分布均值的问题——两种合理动作的平均值可能不合理

## N
- **NaVILA** — 将导航和操作统一在 VLA 框架中的移动操作方法

## O
- **Observation Space（观测空间）** — 机器人能感知到的所有信息的集合（图像+本体感觉+语言+深度+力等）
- **Octo** — UC Berkeley 2024 的开源通用机器人策略（93M 参数），核心创新为 Readout Token 机制和可替换 Action Head
- **Open X-Embodiment** — Google DeepMind 联合 21 机构的跨机器人数据集（~100 万轨迹、22 种机器人、160K+ 任务）
- **OpenVLA** — Stanford/Berkeley 2024 的开源 7B VLA（SigLIP+DINOv2+Llama2），HuggingFace 下载量 100K+

## P
- **PaliGemma** — Google 的 VLM（SigLIP + Gemma 2B），π₀ 的基座模型
- **Perceiver Resampler** — 用可学习查询 Token 通过交叉注意力压缩视觉 Token 序列的模块（Flamingo 使用）
- **Policy（策略）** — 从观测到动作的映射函数 $\pi(a|o)$
- **PPO（Proximal Policy Optimization）** — 常用的策略梯度 RL 算法，Isaac Gym 中机器人训练的主流选择
- **Prismatic VLM** — OpenVLA 的基座 VLM，核心创新为 SigLIP+DINOv2 双视觉编码器互补
- **Proprioception（本体感觉）** — 机器人对自身关节角度/速度/力矩/末端位姿的感知

## Q
- **QUAR-VLA** — MiLAB 的四足机器人 VLA，将 VLA 范式从操作扩展到 12-DoF 四足运动控制

## R
- **RAG（Retrieval-Augmented Generation，检索增强生成）** — 具身 Agent 中情景记忆的实现方式：将经验编码为向量→存入数据库→按相似度检索
- **Readout Token** — Octo 中的可学习特殊 Token，通过全局注意力聚合任务+视觉+本体感觉信息后输出给 Action Head
- **REINFORCE** — 基础策略梯度算法：$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot G_t]$
- **Reward Hacking（奖励黑客）** — RL 中模型找到获取高奖励但不符合期望的"捷径"（如通过摔倒向前滑行来最大化前进距离）
- **RL（Reinforcement Learning，强化学习）** — 通过环境交互和奖励信号学习最优策略的方法
- **RLHF（Reinforcement Learning from Human Feedback）** — 用人类偏好反馈微调模型的方法，正在探索应用于 VLA 安全对齐
- **RLDS** — Google 的机器人学习数据集标准格式
- **RSS（Responsibility-Sensitive Safety，责任敏感安全）** — Mobileye 提出的驾驶安全数学模型，提供形式化安全距离保证
- **RT-1** — Google 2022，35M 参数 Robotics Transformer，EfficientNet+TokenLearner+Transformer 架构，130K 真机轨迹训练
- **RT-2** — Google DeepMind 2023，VLA 范式开创者，将动作编码为文本 Token 让 VLM(PaLI-X 55B)直接输出机器人动作

## S
- **SayCan** — Google 2022 的 LLM+机器人框架：$\pi(l_i) = p_{\text{LLM}}(l_i) \times \text{Affordance}(l_i)$
- **SigLIP** — Google 的改进 CLIP，使用 Sigmoid 损失（逐对独立，不需全局 batch 归一化）
- **Sim-to-Real** — 从仿真环境到真实环境的迁移，典型性能衰减：仿真 90% → 真机 60-80%
- **SOTIF（Safety Of The Intended Functionality）** — ISO 21448 预期功能安全标准，VLA 幻觉属于 SOTIF 范畴
- **Speculative Decoding（投机解码）** — 用小模型快速猜测多个 Token→大模型并行验证，加速自回归生成 2-3×
- **SpatialVLA** — 增强 VLA 空间理解能力的模型，引入空间 Token 设计

## T
- **Tabletop Manipulation（桌面操作）** — 在固定桌面上进行的抓取/放置/推/开关等操作任务
- **Tactile Sensing（触觉传感）** — 通过接触获取压力分布等信息（GelSight/DIGIT/BioTac）
- **Teleoperation（遥操作）** — 人类通过控制设备远程操控机器人来收集演示数据（VR 手柄/主从臂/手持示教等）
- **Temporal Ensemble（时间集成）** — ACT 中多次预测的重叠区域取指数加权平均 $a_t = \sum w_i a_t^{(i)}$，平滑动作
- **3D-VLA** — 用点云（PointNet++）替代/增强 2D 图像输入并引入 3D 世界模型的 VLA
- **TokenLearner** — RT-1 中将 81 个空间 Token 压缩为 8 个的可学习模块（~100× 注意力计算量削减）

## U
- **UniSim** — Google DeepMind 的通用交互式世界模拟器，支持语言/连续动作/相机轨迹三种粒度的动作输入

## V
- **VLA（Vision-Language-Action Model，视觉-语言-动作模型）** — 从视觉和语言输入生成机器人动作的模型，LLM→VLM→VLA 的演化产物
- **VLM（Vision-Language Model，视觉-语言模型）** — 同时理解图像和文本的大模型（如 LLaVA、PaliGemma、InternVL）
- **Voyager** — NVIDIA 2023 在 Minecraft 中实现的自我驱动探索 Agent，具有自动课程学习+代码生成+技能库积累
- **VQ-VAE（Vector Quantized VAE）** — 将连续表示离散化为码本中向量的 VAE 变体，Genie 用于视觉 Token 化

## W
- **World Model（世界模型）** — 预测"执行动作 $a$ 后世界状态如何变化"的模型：$\hat{s}_{t+1} = f_\theta(s_t, a_t)$

## Z
- **Zero Convolution（零卷积）** — ControlNet 中权重初始化为零的 1×1 卷积，确保训练初期不破坏预训练权重

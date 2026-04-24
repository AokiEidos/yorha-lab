---
title: "Diffusion 技术文档系列 · 目录索引"
date: 2026-04-20T17:36:41.000+08:00
draft: false
tags: ["diffusion"]
hidden: true
---

# Diffusion 技术文档系列 · 目录索引

## 模块一：基础理论

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 1.1 | [扩散模型核心直觉](../1-core-intuition) | 🔰 | 墨水扩散类比、加噪→去噪直觉、GAN/VAE/Flow 对比表、雕塑家类比、本系列路线图 |
| 1.2 | [前向过程详解](../2-forward-process) | 🔰⚙️ | 马尔可夫链定义、逐步加噪公式、封闭解 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 推导、$\alpha_t/\bar\alpha_t$ 参数表、SNR 定义、PyTorch 伪代码 |
| 1.3 | [反向过程与训练目标](../3-reverse-process) | ⚙️ | 贝叶斯后验推导、三种参数化（$\epsilon/x_0/v$-预测）对比表、简化 MSE 损失推导、ELBO 分解、方差选择、训练伪代码 |
| 1.4 | [采样算法](../4-sampling-algorithms) | ⚙️ | DDPM 采样逐行解读、祖先采样 vs 确定性采样、$\sigma_t$ 选择、随机性作用分析、速度局限（~1000 步）、FID/IS/CLIP Score 评估 |
| 1.5 | [噪声调度](../5-noise-schedule) | ⚙️ | 线性 vs 余弦调度（含 $\bar\alpha_t$ 曲线对比图）、SNR 统一视角、Min-SNR 加权、高分辨率偏移噪声、Zero Terminal SNR、连续时间调度 |
| 1.6 | [得分函数与得分匹配](../6-score-matching) | ⚙️🔬 | 得分函数几何直觉（地形图类比）、DSM 推导、与 DDPM 等价性 $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$、NCSN 多尺度设计、退火朗之万动力学伪代码 |
| 1.7 | [SDE/ODE 统一视角](../7-sde-ode) | 🔬 | VP-SDE/VE-SDE/sub-VP 定义与对比、反向 SDE（Anderson 1982）、概率流 ODE 推导、Euler-Maruyama 离散化 = DDPM、ODE = DDIM、高阶求解器预告 |

## 模块二：主流模型架构

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 2.1 | [DDPM 详解](../1-ddpm) | ⚙️ | 完整 U-Net 架构图（ResBlock+Attention+时间步嵌入）、GroupNorm/SiLU 选择、完整训练/采样 PyTorch 伪代码、EMA、Improved DDPM、与 GAN/VAE FID 对比 |
| 2.2 | [DDIM 详解](../2-ddim) | ⚙️ | 非马尔可夫前向过程构造、$\eta$ 参数控制随机性、采样公式三项解读、子序列采样伪代码、步数-FID 对比表、DDIM Inversion 伪代码、与概率流 ODE 联系 |
| 2.3 | [Score-based Models](../3-score-based) | 🔬 | NCSN 多尺度设计、退火朗之万伪代码、Song et al. SDE 统一（DDPM=VP-SDE, NCSN=VE-SDE）、PC 采样器、两条路线交汇图 |
| 2.4 | [Latent Diffusion & Stable Diffusion](../4-latent-diffusion) | ⚙️ | 像素→潜空间压缩比分析、VAE+条件 U-Net+文本编码器三组件、Cross-Attention 代码、SD 1.x→2.x→XL→3.x 演化表（含 MM-DiT/Flow Matching） |
| 2.5 | [Consistency Models](../5-consistency-models) | ⚙️🔬 | 自一致性约束公式、CD（教师 ODE 一步）和 CT（无教师）训练伪代码、多步采样算法、LCM 连接、Pseudo-Huber 损失、iCT lognormal 时间分布、Scaling Laws（FID vs 模型规模 152M-1.5B）、与渐进蒸馏对比 |
| 2.6 | [Flow Matching](../6-flow-matching) | ⚙️🔬 | CNF 框架、CFM 训练目标推导与代码、Rectified Flow + Reflow 伪代码、OT 路径数学、SD3 logit-normal 时间采样公式、FM vs DDPM vs DDIM 定量对比表（FID/步数/速度）、与 VLA π₀ 交叉引用、Stochastic Interpolants |
| 2.7 | [模型演化关系图](../7-evolution-map) | 🔰 | Mermaid 全景图、DDPM→DDIM→Score SDE→LDM→SD→Consistency→FM 技术脉络、三条主线（架构/训练/加速）、里程碑时间线 |

## 模块三：条件生成技术

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 3.1 | [条件生成概述](../1-overview) | 🔰⚙️ | 贝叶斯分解→两大范式、完整分类树（引导/注入/微调 3 大类 12 种方法）、4 种技术注入方式代码（加法/Cross-Attn/拼接/自适应归一化）、条件信号类型表（含 VLA 动作生成）、训练 vs 推理对比、历史演进时间线（2020-2025） |
| 3.2 | [Classifier Guidance](../2-classifier-guidance) | ⚙️ | 完整贝叶斯→得分函数→噪声预测推导、温度缩放 $p(y\|x_t)^s$ 数学、噪声分类器训练代码、ADM 架构详情（554M 参数）、完整采样伪代码、ImageNet 256 多引导强度 benchmark（FID/IS/Precision/Recall 7 级）、Precision-Recall 曲线图、局限分析表 |
| 3.3 | [Classifier-Free Guidance](../3-cfg) | ⚙️🔬 | 隐式分类器等价性证明、Guidance Scale 效果表（含 FID/CLIP Score）、动态引导策略代码（线性/余弦衰减）、CFG++ $x_0$ 空间引导、负提示机制公式、批处理优化代码（内存/速度权衡表）、不同模型 CFG 配置表（SD1.5/SDXL/SD3/FLUX/Imagen） |
| 3.4 | [ControlNet](../4-controlnet) | ⚙️ | 零卷积梯度流数学分析（为什么零初始化有效）、完整架构图（含各层通道维度 320/640/1280）、训练配方表（lr/batch/GPU/数据量）、预处理代码（Canny/MiDaS/OpenPose）、Multi-ControlNet 权重平衡策略、ControlNet-XS（85% 参数削减） |
| 3.5 | [IP-Adapter](../5-ip-adapter) | ⚙️ | 解耦交叉注意力完整架构图（含 tensor 维度）、前向传播 PyTorch 代码、基础版 vs Plus 版（CLS vs Patch tokens）对比、训练详情（10M 数据/8×V100/1 周）、Scale $\lambda$ 调优指南表、ControlNet 组合架构图、FaceID 变体（InsightFace） |
| 3.6 | [适配器与微调方法](../6-adapters) | ⚙️ | T2I-Adapter vs ControlNet 对比、LoRA 数学（$W'=W+BA$）+ Rank 选择分析（rank 1-128 表）+ $\alpha$ 缩放 + SDXL 特殊性 + 合并代码、Textual Inversion 优化伪代码、DreamBooth 先验保持损失推导 + 训练配方、选型决策树 |
| 3.7 | [方法对比](../7-comparison) | 🔰 | 3 张对比表（核心维度/训练成本/质量精度）、方法组合兼容性矩阵（12 种组合+注意事项）、4 种部署场景推荐（在线服务/创作工具/角色动画/数据增强）、完整演进时间线（2020-2025）、按控制维度和预算的选型速查卡 |

## 模块四：加速与部署

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 4.1 | [采样加速概述](../01-overview) | ⚙️ | 推理开销数学分析（NFE 概念）、三方向分类树（减步数/加速单步/减计算量）、综合对比表（方法→步数→延迟→FID）、方法可组合性矩阵、6 款硬件延迟对比（H100→Jetson）、生产成本模型（100 万张/日年成本 $17K vs $864K）、边缘设备可行性、技术时间线 |
| 4.2 | [高阶 ODE 求解器](../02-ode-solvers) | ⚙️🔬 | 半线性 ODE 结构发现、指数积分器推导（常数变易法）、DPM-Solver 1/2/3 阶公式、DPM-Solver-2 PyTorch 伪代码、DPM-Solver++ 数据预测重参数化（CFG 友好性分析）、Heun 预测-校正伪代码、Karras 噪声调度公式、UniPC 统一框架、CIFAR-10 + ImageNet FID 对比表、选型指南 |
| 4.3 | [蒸馏方法](../03-distillation) | ⚙️ | 蒸馏分类树（轨迹/一致性/对抗）、渐进蒸馏算法（数学目标+伪代码+训练成本表）、引导蒸馏（CFG 烘焙公式）、LCM 增广 PF-ODE 公式+训练伪代码+LCM-LoRA 使用代码、ADD 对抗损失公式+DINOv2 判别器、DMD2 分布匹配、7 方法对比表（FID/步数/训练成本/LoRA 兼容） |
| 4.4 | [模型量化](../04-quantization) | ⚙️🔬 | 量化公式（均匀量化+NF4）、精度层级表（FP32→INT4）、3 大扩散专属挑战（累积误差雅可比公式+时间步激活分布图+注意力 softmax 敏感性）、PTQ 流程伪代码、Q-Diffusion/PTQD 方法、时间步感知混合精度算法伪代码、层级敏感度经验表、QAT STE 公式+训练代码、SD 1.5/SDXL benchmark 表（延迟/FID/显存）、TRT 部署代码 |
| 4.5 | [推理优化实践](../05-inference-optimization) | ⚙️ | Flash Attention 分块机制（HBM vs SRAM + 伪代码 + 复杂度对比）、xFormers/SDPA、Token Merging 算法（合并比例-FID 表）、DeepCache 缓存策略图+伪代码+间隔分析表、VAE Tiling、CPU Offload 3 级策略、TensorRT/torch.compile/CUDA Graph 代码、3 种端到端优化方案（快速/高质量/低显存+实测数据） |

## 模块五：大模型时代应用

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 5.1 | [文生图技术栈](../01-text-to-image) | ⚙️ | 完整 T2I 管线图、6 种文本编码器对比表（CLIP-L→T5-XXL 含维度/参数/训练）、多编码器融合代码（SD3 三编码器）、U-Net vs DiT 对比（归纳偏置/Scaling/硬件效率）、分辨率策略对比（级联/直接/潜空间+Imagen 级联流程图）、8 模型全面对比表（含 FID/CLIP Score）、训练数据/框架演进 |
| 5.2 | [DiT 架构解析](../02-dit) | ⚙️🔬 | Patchification 数学、adaLN-Zero 完整代码（6 参数+零初始化原理）、DiT Scaling 表（S/B/L/XL 含 GFLOPs/FID）+ GFLOPs-FID 曲线图、MM-DiT 联合注意力架构图+数学（独立 QKV→concat→split）、SD3 Rectified Flow + logit-normal 时间采样、DiT vs U-Net vs MM-DiT 对比 |
| 5.3 | [文生视频](../03-text-to-video) | ⚙️ | 3D U-Net 架构图、时空分离注意力完整 PyTorch 代码（含 einops）、零初始化膨胀技巧、计算量分析（全 3D vs 分离 ~16× 节省）、Sora 3D Patchification 原理+设计分析表、视频 VAE 时空压缩对比表、运动控制方法表、8 模型对比表（含时长/分辨率/开源）、评估指标（FVD/CLIPSIM） |
| 5.4 | [图像编辑](../04-image-editing) | ⚙️🔬 | SDEdit 噪声强度-编辑强度量化表（LPIPS/SSIM）、DDIM Inversion 数学推导+误差累积分析、Null-text Inversion 伪代码、Prompt-to-Prompt 三种操作伪代码（word_swap/refine/reweight）、InstructPix2Pix 训练数据管线图（GPT-3→P2P→配对数据）、Inpainting 架构（9 通道输入）、7 种编辑方法对比表 |
| 5.5 | [超分辨率与图像修复](../05-super-resolution) | ⚙️ | 回归模糊原因图解、条件扩散 SR 数学、SR3 架构+实验（人类混淆率 50.2%!）、PSNR vs LPIPS 权衡曲线+Set14 benchmark、SD Upscaler 管线、Imagen 级联 SR、盲 SR 统一退化模型（Real-ESRGAN 两阶段退化）、统一修复框架数学（6 种任务统一为 $y=\mathcal{A}(x)+n$）、无训练后验采样伪代码 |
| 5.6 | [3D 生成](../06-3d-generation) | 🔬 | DreamFusion 管线图、SDS 梯度完整数学推导 $\nabla_\theta L = E[w(t)(\epsilon_\phi-\epsilon)\partial x/\partial\theta]$、Janus 问题图解+5 种解决方案、VSD 变分改进（LoRA 基线 vs 随机噪声对比表）、Zero-1-to-3 视角条件、多视角扩散对比表（MVDream/SyncDreamer/Wonder3D/Zero123++）、3DGS+Diffusion（DreamGaussian）、综合方法对比表 |
| 5.7 | [多模态融合](../07-multimodal) | 🔬 | 架构 A（LLM+Diffusion 管线，DALL-E 3 Prompt 重写机制）、架构 B（统一 Token 空间，离散 VQ-VAE 代码 vs 连续 VAE 对比表，Chameleon/Show-o）、架构 C（Diffusion Head on LLM，Transfusion 训练目标伪代码+实验）、3 架构对比表、Agent 中的 Diffusion、未来趋势（AR+扩散混合） |

## 模块六：自动驾驶应用

| # | 文档 | 难度 | 核心内容 |
|---|------|------|---------|
| 6.1 | [全景概述](../vla/autonomous-driving-01-vla-driving-paradigm) | 🔰⚙️ | 完整 AD 技术栈架构图（含 Diffusion 切入点标注）、三层分类法（数据/算法/系统）、4 大契合点数学分析（多模态分布公式+不确定性量化+条件控制+数据生成）、长尾场景频率表、Diffusion vs 传统方法 6 任务对比、代表性工作时间线（2022-2025） |
| 6.2 | [驾驶场景生成](../vla/autonomous-driving-02-drivevlm) | ⚙️ | BEVGen 架构图+生成管线、MagicDrive 3D Bbox+HD Map 条件机制架构图、跨视角一致性 4 层解决方案、DriveDreamer 两阶段管线图、7 维可控生成表、FID/FVD 质量指标对比、vs DriveGAN 对比 |
| 6.3 | [轨迹预测](../vla/autonomous-driving-03-other-driving-vla) | ⚙️ | 多模态未来图解、条件去噪数学建模（$p(\tau\|c)$ 公式）、MID 架构图（Transformer 编码+Diffusion 解码）、CTG 联合多智能体机制、Diffusion-ES 进化搜索框架图、3 种约束引导公式（$J_{collision}/J_{lane}/J_{kin}$）、引导强度调优表、ETH-UCY + nuScenes 定量对比 |
| 6.4 | [运动规划](../vla/autonomous-driving-04-scene-understanding) | ⚙️🔬 | Diffuser 完整架构图（训练+推理两阶段）、轨迹级去噪数学、离线 RL 数据集表、Classifier Guidance 奖励引导公式表（5 种约束）、Decision Diffuser CFG 公式、vs MPC/RL 全面对比表（10 维度）、安全性深度分析（概率模型 vs 硬约束）、SafeDiffuser、工业混合架构 |
| 6.5 | [数据增强与仿真](../vla/autonomous-driving-05-limitations) | ⚙️ | 采集成本结构表、长尾覆盖分析（nuScenes 场景分布）、LiDAR Range Image 转换管线图+数学、天气/光照增强管线（SDEdit/WeatherDiffusion）、Sim-to-Real 域迁移管线图（CARLA→nuScenes mAP +8-15%）、闭环仿真完整架构图（场景+行为+渲染+评估）、真实 vs 合成成本经济学、质量验证流程 |
| 6.6 | [世界模型](../vla/autonomous-driving-04-scene-understanding) | 🔬 | 状态转移模型形式定义（$\hat{s}_{t+1}=f_\theta(s_t,a_t)$）+概率形式、GAIA-1 架构图（9B 参数/三路编码/Video Diffusion 解码）、DriveDreamer-2 LLM 场景描述管线、GenAD 潜空间统一架构（vs Diffuser 对比表）、视频质量指标表（FVD/FID/LPIPS）、物理一致性 7 类失败案例表、长期预测误差衰减、当前局限量化表+发展路线图、VLA 交叉引用 |

---

## 推荐阅读路径

### 🔰 入门路径（~10 篇，建立直觉）
1.1 → 1.2 → 1.3 → 1.4 → 2.1 → 2.2 → 2.4 → 3.1 → 3.3 → 5.1

### ⚙️ 工程师路径（~25 篇，掌握实用技术）
入门路径 + 1.5 → 2.5 → 2.6 → 3.4 → 3.5 → 3.6 → 3.7 → 4.1-4.5 → 5.2 → 5.4

### 🔬 研究者路径（全部 42 篇）
按顺序阅读全部内容

### 🚗 自动驾驶路径（~15 篇）
1.1-1.4 → 2.1 → 2.4 → 3.3 → 4.1 → 6.1-6.6

### ⚡ 部署优化路径（~10 篇）
1.4 → 2.2 → 2.5 → 2.6 → 3.3 → 4.1-4.5

### 🎨 创作应用路径（~12 篇）
1.1-1.3 → 2.4 → 3.1 → 3.3-3.7 → 5.1 → 5.4

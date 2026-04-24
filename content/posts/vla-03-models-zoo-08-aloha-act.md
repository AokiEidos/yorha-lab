---
title: "ALOHA 与 ACT"
date: 2026-04-20T16:40:06.000+08:00
draft: false
tags: ["VLA"]
hidden: true
---

# ALOHA 与 ACT

> ⚙️ 进阶 | 前置知识：[动作解码头设计](../02-architecture/03-action-head.md)

## ALOHA：低成本双臂遥操作

**ALOHA（A Low-cost Open-source Hardware System for Bimanual Teleoperation）**（Stanford / Google DeepMind, 2023，发表于 *RSS 2023*）是一套低成本双臂遥操作系统，配合 **ACT（Action Chunking with Transformers）** 策略，在 VLA 生态中扮演了**数据收集基础设施**和 **Action Chunking 技术推动者**的双重角色。

ALOHA 的影响远超其论文本身——它成为了社区中最流行的双臂操作数据收集平台之一。π₀、OpenVLA 等后续工作都使用 ALOHA 采集的数据进行训练和评测。

## ALOHA 硬件系统

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALOHA 硬件系统                                 │
│                                                                 │
│  操作员端 (Leader):              机器人端 (Follower):              │
│  ┌──────────┐ ┌──────────┐    ┌──────────┐ ┌──────────┐       │
│  │ 主臂 L   │ │ 主臂 R   │    │ 从臂 L   │ │ 从臂 R   │       │
│  │ ViperX   │ │ ViperX   │    │ ViperX   │ │ ViperX   │       │
│  │ 300 Pro  │ │ 300 Pro  │    │ 300 Pro  │ │ 300 Pro  │       │
│  └────┬─────┘ └────┬─────┘    └────┬─────┘ └────┬─────┘       │
│       │             │               │             │             │
│       └──────┬──────┘               └──────┬──────┘             │
│              │  主从映射                     │                    │
│              │  (关节角直接复制)              │                    │
│              └────────────────────────→─────┘                    │
│                                                                 │
│  相机系统:                                                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
│  │ Top  │ │Front │ │Left  │ │Right │                          │
│  │ cam  │ │ cam  │ │wrist │ │wrist │                          │
│  └──────┘ └──────┘ └──────┘ └──────┘                          │
│  (640×480, 30 FPS × 4 视角)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 详细硬件规格表

| 组件 | 型号/规格 | 数量 | 单价 (USD) | 小计 |
|------|---------|------|-----------|------|
| **机械臂** | Trossen ViperX 300 Pro | 4 (2主+2从) | ~$4,500 | $18,000 |
| **夹爪** | ViperX 标准平行夹爪 | 4 | 含在臂中 | - |
| **俯视相机** | Logitech C920 | 1 | ~$70 | $70 |
| **前视相机** | Logitech C920 | 1 | ~$70 | $70 |
| **腕部相机** | Intel RealSense D405 | 2 | ~$300 | $600 |
| **控制电脑** | 标准 PC + GPU | 1 | ~$2,000 | $2,000 |
| **安装框架** | 8020 铝型材 | 1 套 | ~$300 | $300 |
| **线缆/电源** | 各种 | - | ~$200 | $200 |
| **总计** | | | | **~$21,240** |

对比：工业级双臂遥操作系统（如 Shadow Hand + 双 Kuka/Franka）成本 **$200K+**，ALOHA 降低了约 **10 倍**。

### 关键硬件参数

| 参数 | 值 |
|------|-----|
| 臂类型 | 6-DOF + 1 夹爪 = 7 DOF per arm |
| 关节驱动 | Dynamixel XM/XL 系列舵机 |
| 最大负载 | ~750g per arm |
| 重复精度 | ~1mm |
| 工作空间 | 约 30×30×20 cm (桌面) |
| 控制频率 | 50 Hz (关节角指令) |
| 通信方式 | USB (Dynamixel U2D2) |
| 每条轨迹 | ~10-60 秒 |
| 数据录制 | 50 Hz 关节角 + 30 FPS 图像 (4 视角) |

### 数据格式

```python
# ALOHA 数据格式 (单条轨迹)
trajectory = {
    'observations': {
        'images': {
            'top': np.array(shape=(T, 480, 640, 3)),      # 俯视相机
            'front': np.array(shape=(T, 480, 640, 3)),     # 前视相机
            'left_wrist': np.array(shape=(T, 480, 640, 3)),  # 左腕相机
            'right_wrist': np.array(shape=(T, 480, 640, 3)), # 右腕相机
        },
        'qpos': np.array(shape=(T, 14)),  # 14 维关节角 (7L + 7R)
        'qvel': np.array(shape=(T, 14)),  # 14 维关节角速度
    },
    'actions': np.array(shape=(T, 14)),   # 14 维目标关节角
    'language_instruction': "pick up the cube and place it in the bowl",
}
# T ≈ 500-3000 步 (10-60 秒 @50Hz)
```

## ACT（Action Chunking with Transformers）详解

### 核心创新

ACT 是 ALOHA 配套的策略算法，有两个核心创新：

1. **Action Chunking**：一次预测 $H=100$ 步的完整动作序列（2 秒 @50Hz）
2. **CVAE 框架**：用**条件变分自编码器（Conditional Variational Autoencoder, CVAE）** 处理动作的多模态性

### 完整架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACT Architecture                              │
│                                                                 │
│  ═══════════ 训练时 (有专家动作) ═══════════                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  CVAE 编码器 (仅训练时使用)                             │       │
│  │                                                      │       │
│  │  专家动作序列 a* (100步×14维)                           │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  [动作 Tokenizer] → 100 个动作 Token                  │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  [CVAE Encoder Transformer]                          │       │
│  │    4 层, d=512, 8 头                                  │       │
│  │    输入: [CLS] + 动作 Token + 观测 Token              │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  CLS 输出 → [Linear] → μ, σ²  (潜变量参数)            │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  z = μ + σ · ε,  ε~N(0,I)  (重参数化采样)             │       │
│  │  z ∈ R^d_z  (d_z = 32)                               │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│  ═══════════ 推理时 (无专家动作) ═══════════                      │
│                         │                                       │
│  z ~ N(0, I)  ──────────┤ (推理时从先验采样)                      │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  CVAE 解码器 (训练和推理时都使用)                        │       │
│  │                                                      │       │
│  │  观测编码:                                             │       │
│  │    图像 (4 视角) → [ResNet-18] → 各 512-d 特征         │       │
│  │    关节角 (14维) → [MLP] → 512-d                      │       │
│  │                                                      │       │
│  │  输入拼接:                                             │       │
│  │    [z (32-d → 投影到 512-d)] +                        │       │
│  │    [图像特征 ×4] + [关节角特征]                         │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  [CVAE Decoder Transformer]                          │       │
│  │    7 层, d=512, 8 头                                  │       │
│  │    Query: 100 个可学习位置 Token (对应 100 步)          │       │
│  │    Key/Value: 观测 + z                                │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  [Linear 投影] → 100 × 14 维 = 1400 维               │       │
│  │       │                                              │       │
│  │       ▼                                              │       │
│  │  预测动作序列: (100 步 × 14 维关节角)                   │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### CVAE 损失公式

ACT 的训练损失包含两部分：**重建损失** + **KL 散度**：

$$\mathcal{L}_{\text{ACT}} = \underbrace{\frac{1}{H \cdot D} \sum_{t=1}^{H} \| \hat{a}_t - a_t^* \|_1}_{\text{L1 重建损失}} + \beta \cdot \underbrace{D_{\text{KL}}(q_\phi(z | a^*, o) \| p(z))}_{\text{KL 正则化}}$$

其中：
- $H = 100$: chunk size（预测步数）
- $D = 14$: 动作维度（双臂 7×2）
- $\hat{a}_t$: 解码器预测的第 $t$ 步动作
- $a_t^*$: 专家的第 $t$ 步动作
- $q_\phi(z | a^*, o)$: 后验分布（编码器输出，训练时）
- $p(z) = \mathcal{N}(0, I)$: 先验分布（推理时采样源）
- $\beta$: KL 权重，论文中 $\beta = 10$

**为什么使用 L1 而非 L2 重建损失？** L1 损失对异常值更鲁棒。在遥操作数据中，偶尔会有操作员的手抖动或误操作，L1 损失对这些异常点的惩罚更温和。

**KL 散度的直觉**：KL 项迫使后验分布 $q_\phi(z|a^*,o)$ 接近标准高斯先验 $p(z)$。如果没有 KL 项，编码器可能学到任意复杂的潜变量分布，但推理时我们只能从标准高斯采样——两者不匹配会导致生成质量下降。

### CVAE 处理多模态动作

同一观测下可能有多种合理动作（如左手先动 vs 右手先动）。CVAE 通过潜变量 $z$ 对不同"模式"进行编码：

```
观测: 桌上有个杯子，需要双手协作拿起

模式 1: 左手先移向杯子 → z ≈ [-1.2, 0.3, ...]
模式 2: 右手先移向杯子 → z ≈ [0.8, -0.5, ...]
模式 3: 双手同时移动   → z ≈ [0.1, 0.1, ...]

训练时: z 从后验 q(z|a*,o) 采样 → 知道专家选了哪种模式
推理时: z 从先验 p(z) = N(0,I) 采样 → 随机选择一种合理模式
```

### 时间集成（Temporal Ensemble）详细推导

执行时不是等 100 步全执行完再推理，而是每执行 $k$ 步就重新推理，对重叠部分做**指数加权平均**。

**问题**：在时刻 $t$，可能有多次推理（在时刻 $t-k, t-2k, ...$ 触发的）对 $t$ 的动作有预测。如何融合这些预测？

**解法**：指数加权平均，越近的推理权重越高。

设在时刻 $\tau_i$ 进行了第 $i$ 次推理，它对时刻 $t$ 的预测是 $a_t^{(\tau_i)}$（其中 $t - \tau_i$ 是 chunk 内的相对位置）。则实际执行的动作为：

$$a_t^{\text{exec}} = \frac{\sum_{i} w_i \cdot a_t^{(\tau_i)}}{\sum_{i} w_i}$$

权重公式：

$$w_i = \exp(-m \cdot (t - \tau_i))$$

其中 $m \geq 0$ 控制衰减速度：
- $m = 0$：所有预测等权（简单平均）
- $m \to \infty$：只用最新一次推理的预测（无集成）
- 论文推荐 $m = 0.01$

**推导直觉**：

```
时刻 t=50, k=10 (每10步推理一次):

推理 @t=40: 预测了 [a_40, a_41, ..., a_139]
  → 对 t=50 的预测: a_50^(40), 距离=10, w=exp(-0.01×10)=0.905
  
推理 @t=50: 预测了 [a_50, a_51, ..., a_149]
  → 对 t=50 的预测: a_50^(50), 距离=0,  w=exp(-0.01×0)=1.000

融合: a_50^exec = (1.000 × a_50^(50) + 0.905 × a_50^(40)) / (1.000 + 0.905)
              ≈ 0.525 × a_50^(50) + 0.475 × a_50^(40)
```

这种集成的好处是**平滑动作**（减少抖动）同时**偏向最新预测**（最新观测包含最新信息）。

### ACT 训练伪代码

```python
# ACT 完整训练循环
class ACT(nn.Module):
    def __init__(self, action_dim=14, chunk_size=100, latent_dim=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        # 视觉编码器 (4 个相机共享权重)
        self.image_encoder = ResNet18(pretrained=True, output_dim=512)
        
        # 本体感觉编码器
        self.proprio_encoder = nn.Linear(action_dim, 512)
        
        # CVAE 编码器 (仅训练时)
        self.cvae_encoder = TransformerEncoder(
            num_layers=4, d_model=512, nhead=8
        )
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        
        # CVAE 解码器
        self.z_proj = nn.Linear(latent_dim, 512)
        self.position_queries = nn.Parameter(
            torch.randn(chunk_size, 512)  # 100 个位置查询
        )
        self.cvae_decoder = TransformerDecoder(
            num_layers=7, d_model=512, nhead=8
        )
        self.action_head = nn.Linear(512, action_dim)
    
    def encode(self, images, proprio, expert_actions):
        """CVAE 编码器: 从专家动作中提取潜变量"""
        # 编码观测
        img_feats = [self.image_encoder(img) for img in images]  # 4 × (B, 512)
        proprio_feat = self.proprio_encoder(proprio)              # (B, 512)
        
        # 编码专家动作
        action_tokens = self.action_tokenizer(expert_actions)     # (B, 100, 512)
        
        # Transformer 编码
        cls_token = self.cls_token.expand(B, 1, 512)
        encoder_input = torch.cat([cls_token, action_tokens, 
                                    *[f.unsqueeze(1) for f in img_feats],
                                    proprio_feat.unsqueeze(1)], dim=1)
        encoded = self.cvae_encoder(encoder_input)
        cls_output = encoded[:, 0]
        
        mu = self.mu_head(cls_output)
        logvar = self.logvar_head(cls_output)
        return mu, logvar
    
    def decode(self, images, proprio, z):
        """CVAE 解码器: 从潜变量生成动作序列"""
        img_feats = [self.image_encoder(img) for img in images]
        proprio_feat = self.proprio_encoder(proprio)
        z_feat = self.z_proj(z)
        
        # 构造条件 (Key/Value)
        condition = torch.cat([z_feat.unsqueeze(1),
                               *[f.unsqueeze(1) for f in img_feats],
                               proprio_feat.unsqueeze(1)], dim=1)
        
        # 位置查询 (Query) → 解码 100 步
        queries = self.position_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.cvae_decoder(queries, condition)
        
        actions = self.action_head(decoded)  # (B, 100, 14)
        return actions
    
    def loss(self, images, proprio, expert_actions, beta=10.0):
        # CVAE 编码
        mu, logvar = self.encode(images, proprio, expert_actions)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        
        # CVAE 解码
        pred_actions = self.decode(images, proprio, z)
        
        # L1 重建损失
        recon_loss = F.l1_loss(pred_actions, expert_actions)
        
        # KL 散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (B * self.chunk_size * 14)  # 归一化
        
        return recon_loss + beta * kl_loss
```

## ACT 实验结果

### ALOHA 原始论文任务（Stanford, 50 条演示/任务）

| 任务 | ACT | BC-ConvMLP | VINN | IBC |
|------|-----|-----------|------|-----|
| 插入电池 | **92%** | 24% | 36% | 18% |
| 转移方块 | **96%** | 52% | 64% | 38% |
| 打开杯盖 | **88%** | 32% | 44% | 22% |
| 穿鞋带 | **68%** | 8% | 12% | 4% |
| 涂抹黄油 | **76%** | 28% | 36% | 16% |
| 平均 | **84%** | 29% | 38% | 20% |

### 消融实验

| 配置 | 平均成功率 |
|------|-----------|
| 完整 ACT (CVAE + Chunk=100 + Temporal Ensemble) | **84%** |
| 无 CVAE (z=0, 确定性) | 68% |
| 无 Temporal Ensemble | 74% |
| Chunk=1 (每步推理) | 42% |
| Chunk=10 | 62% |
| Chunk=50 | 78% |
| Chunk=100 (默认) | **84%** |
| Chunk=200 | 80% |
| L2 损失替代 L1 | 76% |

**关键发现**：
1. **CVAE 贡献 ~16%**：确定性策略（z=0）在多模态任务上显著下降
2. **Chunk Size 100 最优**：太小（1-10）时时间不一致，太大（200）时末端预测不准
3. **Temporal Ensemble 贡献 ~10%**：平滑了 chunk 交界处的动作跳变

## ALOHA 2 改进

Google DeepMind 在 2024 年发布了 **ALOHA 2**，主要改进：

| 维度 | ALOHA (v1) | ALOHA 2 |
|------|-----------|---------|
| 机械臂 | ViperX 300 Pro | **ViperX 300 S** (更高精度) |
| 夹爪 | 标准平行夹爪 | **可更换夹爪系统** |
| 安装方式 | 固定铝型材 | **可调节工作台** |
| 相机 | Logitech C920 + RealSense | **全 RealSense D405** |
| 数据质量 | 中 | **高（更稳定的机械结构）** |
| 力反馈 | 无 | **被动力反馈（弹簧机构）** |
| 控制延迟 | ~20ms | ~10ms |
| 成本 | ~$21K | ~$25K |

### ALOHA 2 的配套策略改进

ALOHA 2 不仅改进硬件，还验证了与更强策略的集成：

| 策略 | ALOHA 1 成功率 | ALOHA 2 成功率 |
|------|--------------|---------------|
| ACT | 84% | 89% |
| Diffusion Policy | 78% | **91%** |
| ACT + 数据增强 | 88% | 92% |

## ACT 与 Diffusion Policy 的集成

ACT 和 **Diffusion Policy（扩散策略）** 代表了两种不同的多模态动作建模方法。社区中有越来越多的工作尝试结合两者：

| 维度 | ACT (CVAE) | Diffusion Policy |
|------|-----------|-----------------|
| **多模态机制** | 潜变量 z 编码模式 | 去噪随机性 |
| **Chunk 大小** | 100 步 | 16 步 |
| **推理速度** | 快（单次前向） | 慢（多步去噪） |
| **动作质量** | 好 | 更好（复杂任务） |
| **训练难度** | 中（KL 调参） | 中（噪声调度） |
| **代码复杂度** | 中 | 中 |

### 结合方式

```
ACT 的 CVAE 编码器
     +
Diffusion Policy 的去噪解码
     ↓
更好的多模态建模 + 更长的 Action Chunk
```

π₀ 本质上就是这种结合的体现：VLM 提供条件编码 + Flow Matching 提供多模态动作生成 + 长 Action Chunk (50步)。

## 对 VLA 生态的全面推动

| 贡献维度 | 具体影响 |
|---------|---------|
| **硬件平台** | $21K 的双臂系统降低了进入门槛，30+ 实验室采用 |
| **数据收集** | 50Hz 高质量双臂数据，成为 Open X-Embodiment 重要来源 |
| **Action Chunking** | 证明了一次预测多步的有效性，直接启发 π₀ 的 50 步 chunk |
| **CVAE 多模态** | 提供了除 Diffusion 外的多模态动作建模方案 |
| **时间集成** | 指数加权平均成为后续 chunk 执行的标准技术 |
| **开源精神** | 硬件设计 + 代码 + 数据全部开源，推动社区协作 |
| **双臂研究** | 推动了双臂协作操作从小众走向主流 |

## 局限性

1. **负载有限**：ViperX 最大负载 ~750g，无法操作大型/重型物体
2. **精度上限**：舵机驱动的精度 (~1mm) 不如工业级（~0.02mm）
3. **无力反馈**（ALOHA 1）：操作员无法感受力，导致柔软物体操作数据质量较低
4. **ACT 的 CVAE 调参**：$\beta$ 和 $d_z$ 对性能影响大，需要任务特定调参
5. **固定工作空间**：桌面固定安装，不支持移动操作

## 小结

| 概念 | 要点 |
|------|------|
| ALOHA 硬件 | 4× ViperX 300 (2主+2从), 4 相机, ~$21K, 50Hz, 开源 |
| ACT 核心 | CVAE (z∈R³²) + Transformer + 100 步 Action Chunking |
| CVAE 损失 | $\mathcal{L} = \text{L1}(\hat{a}, a^*) + 10 \cdot D_{\text{KL}}(q \| p)$ |
| 时间集成 | $a_t = \sum w_i a_t^{(i)} / \sum w_i$, $w_i = \exp(-0.01 \cdot \Delta t)$ |
| 关键结果 | 50 条演示平均 84%，远超 BC/VINN/IBC 基线 |
| ALOHA 2 | 更高精度硬件 + 被动力反馈 + 配合 Diffusion Policy 达 91% |
| 生态贡献 | 降低数据门槛 + 推广 Action Chunking + 推动双臂研究主流化 |

---

> **下一篇**：[Helix（Figure AI）详解](./09-helix.md)

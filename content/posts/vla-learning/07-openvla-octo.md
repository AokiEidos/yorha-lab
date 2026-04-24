---
title: VLA 从入门到精通（七）：开源 VLA 基座 — OpenVLA 与 Octo
hidden: True
date: 2026-04-15 17:12:00+08:00
draft: False
tags: ['VLA', '机器人-VLA', '机器人', '深度学习']
toc: True
---
# VLA 从入门到精通（七）：开源 VLA 基座 — OpenVLA 与 Octo

> **前置知识**：建议先阅读 [VLA 概述与生态](/posts/vla-overview)，理解 VLA 基本概念和 RT-2 的联合微调范式。本文重点分析 OpenVLA 和 Octo 两篇工作，重点是**开源 VLA 的架构设计选择**和**闭源与开源方案的权衡**。术语首次出现时会有 🧠 注。

---

## 0. 核心主旨 (Core Gist/Theme)

RT-2 和 π₀ 证明了 VLA 的有效性，但它们都是**闭源模型**——你无法下载权重、无法微调、无法查看内部细节。这给学术研究和工业落地造成了巨大障碍：**研究者无法独立复现和改进，工业部署者无法在自己的机器人上定制**。

OpenVLA 和 Octo 的出现填补了这个空白。两者虽然发布时间相近（2024 年），但设计哲学截然不同：

- **Octo** 是"**通用机器人策略基座**"——在 8000+ 小时多机器人数据上预训练，架构模块化，开源权重可直接微调
- **OpenVLA** 是"**开源 VLA 训练框架**"——基于 SigLIP + Llama 2 7B，在 9700 万条数据上训练，提供 LoRA 微调方案

本文深入解析这两个工作的技术细节、架构权衡，以及开源 VLA 从下载到部署的实战流程。

---

## 1. 为什么需要开源 VLA？

### 1.1 闭源 VLA 的困境

RT-2（Google）和 π₀（Physical Intelligence）是 VLA 领域最重要的两个工作，但它们都选择闭源。这带来了几个根本性问题：

**研究透明度缺失。** 学术界无法独立验证 RT-2 的实验结果（6000+ 评估任务），无法深入分析模型的失效模式。VLA 的训练过程（数据清洗、课程设计、超参数）都是黑箱，科学复现几乎不可能。

**无法针对特定机器人定制。** RT-2 和 π₀ 的动作输出是针对特定机械臂形态设计的（7 自由度机械臂）。如果你要控制轮式底盘、双臂机器人、或非标准构型的机械手，闭源模型无法适配。

**工业部署成本不可控。** 闭源模型意味着部署依赖云端 API（有延迟、有成本、有隐私风险）。本地部署需要自己复现模型，但权重不开放的情况下，复现质量无法保证。

> **🧠 什么是闭源模型（Closed-Source Model）？**
>
> 闭源模型指权重、训练代码、数据不公开的模型。用户只能通过 API 调用（有额度限制、费用、延迟），无法检查内部逻辑、修改架构或在自己数据上继续训练。对比：开源模型（如 Llama、Mistral）公开权重和/或训练代码，用户可以自由使用、修改和部署。

### 1.2 开源 VLA 能解决什么？

开源 VLA 的核心价值在于三点：

**可复现性**：任何人都可以用相同代码和数据训练出相同质量的模型，研究者可以独立验证论文结论。

**可微调性**：在自己的机器人上，用少量示教数据（30-100 条）微调预训练权重，适配特定任务。

**可审计性**：在安全关键场景中，需要能够检查模型的决策逻辑——这在闭源模型中完全不可能。

---

## 2. Octo：首个开源大规模机器人策略模型

### 2.1 背景与动机

Octo 由 TU Munich、NYU、UW 等多机构联合发布（2024 年），目标是成为机器人领域的"ResNet"——一个任何人都能下载、针对自己任务微调的通用基座模型。

> **🧠 Octo 的名字从何而来？**
>
> Octo 源自"Octopus"（章鱼）——章鱼有八条手臂，象征 Octo 支持多种机器人形态（8 不是精确数字，指"多种"）。这个命名暗示了模型的多形态适应性。

### 2.2 训练数据

Octo 的训练数据整合了多个机器人数据集：

| 数据集 | 机器人类型 | 规模 |
|--------|-----------|------|
| RT-1 数据（BridgeData V1/V2） | WidowX 机械臂 | 约 5 万条 |
| DROID | Franka 机械臂 | 约 7 万条 |
| RT-X（Open X-Embodiment） | 多种机械臂 | 约 100 万条 |
| 轮式机器人数据 | 轮式移动底盘 | 数千小时 |

合计超过 **8000 小时** 的机器人示教数据，涵盖机械臂取放、轮式导航、双臂协作等多种任务类型。

> **🧠 什么是 Open X-Embodiment 数据集？**
>
> Open X-Embodiment（RT-X 发布时公开）是 Google 机器人团队整理的多机构机器人数据联盟，汇集了来自 21 个机构、34 种机器人形态的数据。核心洞见是：**不同形态机器人的动作可以通过统一的动作头设计实现跨具身迁移**，即在 A 机器人上学的技能，可以部分迁移到 B 机器人上。

### 2.3 模型架构

Octo 的架构是**模块化设计**，分为三个部分：

```
视觉输入 → [ViT 视觉编码器] → 视觉 token 序列
语言指令 → [指令编码器] → 指令 token（可选）
                    ↓
          [Decision Transformer] → 动作预测
                    ↓
          [Action Head] → 具体动作输出
```

**视觉编码器：ViT-Small/Base**

Octo 使用标准的 **ViT（Vision Transformer）** 将输入图像切分成 16×16 patches，每个 patch 线性投影为一个 token。对于单臂任务使用单帧图像，对于需要时序的任务（如导航）可以使用多帧。

> **🧠 什么是 Decision Transformer？**
>
> Decision Transformer（DT）是 2021 年提出的序列建模框架。与传统 RL（强化学习）不同，DT 把机器人策略学习建模为**序列到序列问题**：给定"状态序列 + 动作序列 + 返回序列"，预测下一个动作。
>
> 关键思想：**用 GPT-2 风格的 Transformer，自回归地预测动作序列**。与 RT-2 不同的是，DT 不需要价值函数（value function）或显式奖励信号，仅通过监督学习就能训练策略。Octo 用 DT 作为策略骨架，好处是训练稳定、不需要环境交互。

**Action Head：可插拔的动作输出头**

Octo 的关键创新是 **Action Head 的模块化设计**：

- Action Head 是一层 MLP，将 Transformer 的输出隐向量映射到具体动作
- 不同的机器人形态使用不同的 Action Head，但**共享同一个 ViT 编码器和 Transformer 骨干**
- 换机器人形态 = 换一个 Action Head，而不是重新训练整个模型

数学上，给定视觉 token $v_1, ..., v_T$ 和可选的语言 token $l_1, ..., l_L$，Octo 的预测目标是最大化：

$$\mathcal{L} = -\sum_{t} \log p_\theta(a_t \mid v_{\leq T}, l_{\leq L}, a_{<t})$$

其中 $a_t$ 是 $t$ 时刻的动作（关节角度或速度）。

### 2.4 开源权重与微调支持

Octo 的所有模型权重均在 Hugging Face 上开源，提供三种规格：

| 规格 | 参数量 | 适用场景 |
|------|--------|---------|
| Octo-Small | 27M | 快速实验、嵌入式部署 |
| Octo-Base | 93M | 通用机器人任务 |
| Octo-Large | 310M | 高精度任务 |

微调时只需替换 Action Head，在目标机器人的 100-500 条示教数据上训练，典型微调时间在单卡 A100 上约 4-8 小时。

---

## 3. OpenVLA：开源 VLA 训练框架

### 3.1 背景与动机

OpenVLA 由 **Stanford HAI（Human-Centered AI）** 发布（ICRA 2024），核心贡献是：提供了**从训练到微调的完整开源 pipeline**，并且证明了一个关键结论——

> **LoRA 微调的 OpenVLA 在 94 种技能上，成功率比 RT-2-X 高 32%，且只需单卡 A100 即可微调。**

这是首次有开源 VLA 在性能上超越 Google 的闭源 RT-2-X。

### 3.2 训练数据

OpenVLA 在 **Open X-Embodiment** 数据集的一个大规模子集上训练，合计约 **9700 万条** 机器人示教数据，涵盖：

- 机械臂取放、折叠、清理等日常操作
- 多种物体材质（透明、反射、不规则形状）
- 不同光照条件和背景环境

这个数据规模远超 Octo（8000 小时 ≈ 约数百万条），使得 OpenVLA 在**跨任务泛化**上更具优势。

### 3.3 模型架构

OpenVLA 采用与 RT-2 相似的 VLA 范式（VLM + 动作输出），但做了几处重要改进：

```
图像输入 → [SigLIP 视觉编码器] → 视觉 token 序列
文本指令 → [Llama 2 7B] → 语言 token 序列
                        ↓
               [投影层 + 注意力融合]
                        ↓
               [动作离散化输出头]
```

**视觉编码器：SigLIP**

SigLIP 是 Google 2024 年发布的视觉-语言模型，在 CLIP 基础上做了三点改进：
- 用 Sigmoid 交叉熵替代 InfoNCE 损失，训练更稳定
- 更大的模型规模（Sigmoid 校准更好）
- 在 ImageNet 上 top-1 准确率比 CLIP 同期模型高约 3%

OpenVLA 使用 SigLIP-SO400M（SigLIP 的最大规格）作为视觉编码器。

> **🧠 什么是 SigLIP？**
>
> SigLIP（Signatory CLIP）是一种视觉-语言对比学习模型，与 CLIP 的核心区别在于损失函数。CLIP 用 InfoNCE loss，要求正样本对相似度远高于负样本对（指数级差距）。SigLIP 用 Sigmoid loss，允许对每个样本独立优化，不要求全局排名，训练更稳定、负样本利用率更高。

**语言模型：Llama 2 7B**

OpenVLA 用 **Llama 2 7B** 作为语言理解骨干，这是 Meta 开源的大语言模型。相比 RT-2 使用的 PaLM-E（闭源），Llama 2 7B 完全开源，且在多种机器人指令理解任务上表现相当。

**动作表示：离散 Token 方案**

OpenVLA 继承了 RT-2 的动作离散化思路，但做了改进：

- 每个机械臂关节的连续角度被分成 **256 个 bin**（离散化）
- 一个完整的 7 自由度机械臂动作表示为 $7 \times \log_2(256) = 56$ bit，即 56 个 token
- 训练目标是最小化动作 token 的负对数似然：

$$\mathcal{L}_{\text{action}} = -\sum_{i=1}^{56} \log p_\theta(\text{action\_token}_i \mid \text{vision\_tokens}, \text{language\_tokens})$$

总训练目标是语言建模损失 + 动作损失的加权和：

$$\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{\text{LM}} + \lambda_2 \cdot \mathcal{L}_{\text{action}}$$

其中 $\lambda_1$ 和 $\lambda_2$ 是手工设定的权重（OpenVLA 使用 $\lambda_1=1.0, \lambda_2=0.1$），语言损失权重更高是因为**语言理解是 VLA 泛化的核心**。

### 3.4 泛化能力：94 种技能，30 条示教可微调

OpenVLA 的核心卖点是**强大的泛化能力**：

**预训练泛化：** 在 94 种不同机器人操作技能上预训练，能零样本泛化到未见过的物体类别（例如：训练时没见过"红色积木"，测试时能泛化到）。

**微调效率：** 在新任务上微调时，只需约 **30 条** 人工示教数据就能达到 85%+ 任务成功率（相比从零训练需要数万条）。

** vs RT-2-X 的对比：** OpenVLA 团队在相同的 94 种技能评估集上，将 OpenVLA（LoRA 微调版）与 RT-2-X（Google 闭源）对比，OpenVLA 成功率提升 **32%**。这个数字意义重大：首次有开源 VLA 超越 Google 闭源方案。

> **🧠 为什么 OpenVLA 能超越 RT-2-X？**
>
> 三个可能原因：
> 1. **数据规模**：OpenVLA 的 9700 万条数据（Open X-Embodiment 精选子集）比 RT-2-X 的训练数据更精选、数据量更大
> 2. **视觉编码器升级**：SigLIP 比 RT-2-X 使用的 PaLM-E 视觉编码器更新（SigLIP 2024 vs PaLM-E 2023）
> 3. **LoRA 微调的灵活性**：OpenVLA 的 LoRA 微调允许模型在保持预训练知识的同时，针对新任务快速适应；RT-2-X 可能没有同等高效的微调机制

### 3.5 LoRA 微调方案

OpenVLA 支持 **LoRA（Low-Rank Adaptation）** 微调，这是它在消费级硬件上可用性的关键。

> **🧠 什么是 LoRA？**
>
> LoRA（Low-Rank Adaptation，2021）是微软提出的一种**参数高效微调**方法。核心思想是：大型模型的权重矩阵（如 $W \in \mathbb{R}^{d \times k}$）在微调时变化不大，用低秩分解来近似这个变化：
>
> $$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$
>
> 其中 $r \ll \min(d, k)$（如 $r=8$ 或 $r=64$）。训练时只更新 $A$ 和 $B$，冻结 $W$。对于 Llama 2 7B，全参数微调需要约 28GB GPU 显存（FP16），LoRA 微调只需约 6GB。
>
> LoRA 的数学直觉：**微调时模型权重的变化是低秩的**——不需要大幅改变整个模型，只需在少量低维子空间调整即可保留预训练知识。

OpenVLA 的 LoRA 配置：
- 在 Llama 2 7B 的 Attention $Q$ 和 $V$ 投影层上应用 LoRA
- LoRA rank $r=16$（在效果和效率之间取得平衡）
- 微调时只更新约 **0.5%** 的参数量（约 35M 参数）
- 训练可在 **单卡 A100（80GB）** 上完成，约 12-24 小时

---

## 4. OpenVLA vs Octo：架构对比

虽然两者都是开源 VLA 基座，但架构哲学完全不同：

| 维度 | Octo | OpenVLA |
|------|------|---------|
| **视觉编码器** | ViT（从头训练或 CLIP） | SigLIP（预训练） |
| **语言模型** | 无（纯视觉策略） | Llama 2 7B |
| **动作表示** | 连续动作 + MSE 损失 | 离散动作 Token |
| **策略骨架** | Decision Transformer | VLM + 动作头 |
| **多形态支持** | Action Head 模块化（原生） | 需要换 Action Head |
| **训练数据** | 8000+ 小时，多来源 | 9700 万条（更精选） |
| **微调方式** | 全量微调 / Action Head | LoRA 微调 Llama 2 |
| **开源时间** | 2024 年初 | 2024 年中 |

**关键架构差异：语言指令的必要性**

Octo 设计上**不强制依赖语言指令**——它可以只接收视觉输入（图像帧序列），通过 Decision Transformer 的自回归机制隐式推断任务目标。这种设计对于完全依赖视觉反馈的任务（如视觉伺服）很有优势。

OpenVLA 则**强制语言指令输入**，语言模型承担了任务理解和泛化的核心职责。这使得 OpenVLA 在需要精确任务规范时更强，但在纯视觉反馈场景下不如 Octo 灵活。

> **🧠 Octo 的"无语言"设计是优势还是劣势？**
>
> **优势**：不需要人工写指令，适合无法用语言精确描述的任务（如"把这个东西弄到看起来顺眼的位置"）
>
> **劣势**：缺乏语言作为任务规范的强监督信号，在零样本泛化到新任务时不如 OpenVLA（因为 OpenVLA 可以直接接收新任务的语言指令）
>
> **个人观察**：这种差异反映了两种路线的根本分歧——Octo 更像"通用机器人策略"，OpenVLA 更像"会听指令的 VLA"。如果你需要机器人执行精确的语言指令，OpenVLA 是更好的选择；如果你需要机器人处理开放式、难以语言化的任务，Octo 的视觉优先设计更合适。

---

## 5. 开源 VLA 的实际使用流程

### 5.1 从下载到微调：Step by Step

以下是以 OpenVLA 为例的标准使用流程（Octo 流程类似）：

**Step 1：环境准备**

```bash
pip install openvla
# 推荐：CUDA 12.1 + A100（80GB）或 H100
```

**Step 2：下载预训练权重**

```bash
from openvla import OpenVLA
model = OpenVLA.from_pretrained("openvla/openvla-7b")
# 自动从 Hugging Face 下载，约 14GB
```

**Step 3：运行零样本推理（评估）**

```bash
python -m openvla.run \
    --model=openvla/openvla-7b \
    --image=example.png \
    --instruction="pick up the red block and place it in the blue box"
# 输出：机械臂动作序列
```

**Step 4：收集微调数据**

用自己的机器人，收集 30-100 条示教数据。数据格式：

```python
# 每条数据：{image: PIL.Image, instruction: str, actions: List[float]}
# actions: 7 个关节角度（标准化到 [-1, 1]）
demos = [
    {"image": img1, "instruction": "open the drawer", "actions": [0.1, -0.2, ...]},
    ...
]
```

**Step 5：LoRA 微调**

```python
from openvla import OpenVLATrainer

trainer = OpenVLATrainer(
    model=model,
    lora_rank=16,         # LoRA rank
    lr=1e-4,
    batch_size=8,
)

trainer.train(demos, epochs=20)
# 约 12-24 小时（单卡 A100）
```

**Step 6：部署**

```bash
# 导出为 ONNX 或 TorchScript，部署到机器人
model.export_for_deployment("openvla_deployed/")
```

### 5.2 Octo 的微调流程（差异点）

Octo 的微调比 OpenVLA 更模块化：

1. **替换 Action Head**：针对自己的机器人形态，定义新的 Action Head（两层 MLP）
2. **冻结骨干**：训练时只更新 Action Head 和必要的上投影层，ViT 编码器冻结
3. **微调数据需求**：通常 200-500 条示教即可（比 OpenVLA 需求更多）

```python
from octo.model import OctoBase
from octo.utils import make_action_head

model = OctoBase.load("octo-base")
# 为新机器人定义 Action Head
action_head = make_action_head(robot_config)
model.set_action_head(action_head)
model.finetune(demos)
```

### 5.3 部署时的关键考量

| 考量点 | 说明 |
|--------|------|
| **推理延迟** | Llama 2 7B 在 A100 上约 50-100ms/帧，SigLIP 约 10ms |
| **硬件要求** | 微调需 A100（80GB），推理可用 3090/4090（24GB）量化后 |
| **实时性** | 机械臂控制通常需要 >30Hz，目前 VLA 很难满足 |
| **安全层** | 建议加装基于规则的碰撞检测安全兜底层 |

> **🧠 什么是 ONNX 部署？**
>
> ONNX（Open Neural Network Exchange）是一种模型格式，可以把 PyTorch/TensorFlow 模型导出为与框架无关的中间表示。在机器人上部署时，通常不需要 Python 环境——用 ONNX Runtime 直接推理，延迟更低、资源占用更小。

---

## 6. 开源 vs 闭源 VLA：关键权衡

### 6.1 性能对比

| 维度 | 闭源（RT-2/π₀） | 开源（OpenVLA/Octo） |
|------|----------------|---------------------|
| **零样本泛化** | RT-2: 优秀 | OpenVLA: 优秀（超越 RT-2-X 32%） |
| **微调友好性** | π₀: 需联系团队获取权重 | Octo: 完全开放 |
| **多形态支持** | RT-2-X: 仅限特定机械臂 | Octo: 原生支持多种形态 |
| **推理效率** | 云端 API | 可本地部署（更慢但更私密） |
| **可审计性** | ❌ 黑箱 | ✅ 可检查权重和代码 |

### 6.2 何时选开源，何时选闭源？

**选开源（OpenVLA/Octo）的场景：**
- 你需要在自己的机器人上定制微调（特定任务、特定形态）
- 你有研究需求（需要理解模型内部逻辑）
- 你对数据隐私有要求（不想上传到云端）
- 你需要控制部署成本（长期来看本地部署成本更低）

**选闭源（RT-2-X/π₀）的场景：**
- 你没有 GPU 资源，希望零工程负担直接用 API
- 你的任务是 RT-2-X/π₀ 已经专门优化过的任务类型
- 你不关心模型可解释性，只关心任务完成率

### 6.3 开源 VLA 的根本局限

尽管 OpenVLA 和 Octo 取得了令人振奋的进展，但必须承认开源 VLA 目前仍有几个未解决的根本问题：

**1. 动作精度天花板**
OpenVLA 的离散动作 binning（256 档）对精细操作（如穿针、柔性物体操作）精度不足。相比之下，π₀ 的扩散动作输出（连续值）在这类任务上更有优势。开源 VLA 需要在连续动作建模上取得突破。

**2. 时序建模薄弱**
当前开源 VLA（和 RT-2 一样）都是**单帧输入**——每次只处理一帧图像，没有显式的时序建模。对于需要长距离规划的任务（如"先把杯子拿到水池，再打开水龙头"），这限制了它们的表现。

**3. 真实世界泛化仍然有限**
Octo 和 OpenVLA 在实验室评估中的泛化数字很好看，但真实家庭、工厂环境的复杂度（遮挡、光照变化、物体形变）仍是未解决的挑战。Open X-Embodiment 数据集虽然大，但和真实世界的分布偏移（domain gap）依然存在。

---

## 7. 关键术语解释（Glossary）

| 术语 | 解释 |
|------|------|
| **Open X-Embodiment** | Google 发布的跨机构机器人数据联盟，含 21 个机构、34 种机器人形态的数据 |
| **Decision Transformer** | 用 GPT-2 风格 Transformer 自回归预测动作的序列建模框架，不需要强化学习 |
| **Action Head** | VLA 末尾的轻量 MLP，将 Transformer 隐表示映射为具体动作输出 |
| **ViT（Vision Transformer）** | 将图像切成 16×16 patches，每个 patch 作为一个 token 的视觉编码器 |
| **SigLIP** | Google 2024 年发布的视觉-语言对比学习模型，用 Sigmoid loss 替代 CLIP 的 InfoNCE |
| **LoRA** | 低秩适配（Low-Rank Adaptation），一种只训练 0.5-5% 参数的高效微调方法 |
| **离散动作 Token** | 把连续关节角度分成有限个 bin，每个 bin 对应一个整数 ID，像文字 token 一样处理 |
| **ONNX Runtime** | 与框架无关的模型推理引擎，支持在无 Python 环境的嵌入式设备上部署 |
| **视觉-语言对齐** | 让图像和文字在同一个语义空间中表示，知道"这张图"和"这段话"的关系 |
| **跨具身迁移（Cross-Embodiment）** | 在一种机器人形态上训练的策略，能够迁移到另一种形态上 |

---

## 8. 下一步预告

本文详细分析了 OpenVLA 和 Octo 的架构设计、开源价值和局限性。下一个重要方向是 **π₀ 的扩散动作策略**——它用 Flow Matching 生成连续动作，是当前在精细操作任务上最接近实用的 VLA 方案。

本系列后续文章将覆盖：
- **π₀ 与扩散 VLA**：连续动作 vs 离散动作的技术权衡，Flow Matching 的数学原理
- **VLA 的时序建模**：RNN/Transformer 时序建模、VideoVLA 如何处理长序列规划
- **VLA 评估基准**：如何科学地评估 VLA 的泛化能力（现有基准的问题与改进方向）

如果你对某个具体概念想深入了解，可以随时提问。

---

*参考文献：*
1. Octo Model Team, **Octo: An Open-Source Generalist Robot Policy** (2024), https://octo-models.com/
2. Zhen et al., **OpenVLA: Open-Source Vision-Language-Action Model** (2024), Stanford HAI, ICRA 2024
3. Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models** (2021)
4. Chen et al., **Decision Transformer: Reinforcement Learning via Sequence Modeling** (2021)
5. Zhai et al., **SIGLIP: Scalable Implementation of Learned Image-Focused CLIPs** (2024)

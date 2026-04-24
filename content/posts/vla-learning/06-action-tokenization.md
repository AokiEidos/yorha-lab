---
title: "VLA 从入门到精通（六）：Action Tokenization — 离散 vs 连续动作"
hidden: true
date: 2026-04-15T17:10:00.000+08:00
draft: false
tags: ["VLA", "动作分词", "动作分词", "扩散模型", "机器人"]
toc: true
---

# VLA 从入门到精通（六）：Action Tokenization — 离散 vs 连续动作

> **前置知识**：了解 VLA 基础概念（前五篇）。本文聚焦 VLA 的核心技术差异之一：**如何表示和输出动作**，重点对比 RT-2 的离散 token 方案和 π₀ 的连续扩散方案。

---

## 0. 核心主旨 (Core Gist/Theme)

VLA 面临一个独特的挑战：视觉输入和语言输入都是序列（图像 patches、文字 tokens），但**动作输出是连续值**（关节角度、速度、力矩）。如何把连续动作融入 Transformer 的离散 token 框架，是 VLA 架构设计的核心分歧点。

**两条技术路线：**
- **RT-2 路线**：强制离散化 → 把连续动作变成整数 token → 用交叉熵损失监督
- **π₀ 路线**：保持连续性 → 用扩散模型/Flow Matching 生成连续动作分布 → 用 MSE 损失监督

本文深入分析两条路线的数学本质、工程实现和各自优劣。

---

## 1. 动作输出的形式化

### 1.1 连续动作空间的数学表示

机器人的动作空间通常是连续的：

$$a_t \in \mathbb{R}^D$$

其中 $D$ 是动作维度（机械臂：4-7 个关节；人形机器人：30+ 自由度）。每个维度是实数，取值范围通常有物理约束：

$$a_t^d \in [a_{\min}^d, a_{\max}^d]$$

> **🧠 为什么动作空间有约束？**
>
> 关节角度有物理限制（不能超过 180°），电机输出力矩有上限，机械臂末端速度不能超过安全阈值。这些约束保证了机器人在物理世界中的安全性。
>
> VLA 输出动作时，如果输出的是无约束的实数，需要一个**动作映射层（Action Head）**将其投影到合法区间。

### 1.2 动作序列的时序结构

VLA 的动作输出不是单帧决策，而是**时序序列**：

$$\{a_t, a_{t+1}, a_{t+2}, ..., a_{t+H-1}\}$$

其中 $H$ 是**预测视野（Prediction Horizon）**，通常取 8-16 步。一次性预测多步动作有几个好处：

- 减少推理频率（一次推理管多步）
- 规划时考虑动作之间的连贯性
- 为下层控制器提供平滑的参考轨迹

---

## 2. 路线一：离散化 Action Tokenization（RT-2 采用）

### 2.1 动作离散化

RT-2 把每个动作维度 $a_t^d$ 强制离散化成 $K$ 个 bin：

$$a_t^d \approx \text{bin}(a_t^d) = \left\lfloor \frac{a_t^d - a_{\min}^d}{a_{\max}^d - a_{\min}^d} \cdot (K-1) \right\rfloor$$

RT-2 使用 $K = 256$，即每个动作维度被分成 256 档。

> **🧠 256 bin 的精度损失有多大？**
>
> 以机械臂关节角度范围 0°~360° 为例：
>
> - 精度：360/255 ≈ 1.4° 每档
> - 对于精细操作（如穿针、叠衣服），1.4° 的误差可能导致任务失败
>
> 对于粗动作（拿起一个杯子），1.4° 误差可以通过视觉反馈和末端修正弥补，影响不大。

### 2.2 动作 token 序列

RT-2 把 $D$ 个动作维度编码成 $D$ 个 token：

$$\text{ActionTokens} = [\text{action}_0^1, \text{action}_0^2, ..., \text{action}_0^D, \text{action}_1^1, ..., \text{action}_H^D]$$

这和文字 token 序列的形式完全一致，可以用标准的 Transformer 自注意力处理。

### 2.3 交叉熵损失函数

RT-2 的训练损失是标准的**交叉熵（Cross-Entropy）**：

$$\mathcal{L}_{\text{RT-2}} = -\sum_{h=0}^{H-1} \sum_{d=1}^{D} \log \pi_\theta\left(a_t^d = \text{bin}(a_{\text{gt}}^d) \mid s_t, \text{language}\right)$$

其中 $\pi_\theta$ 是 VLM 的输出分布，$\text{bin}(a_{\text{gt}}^d)$ 是真实动作的离散 bin ID。

> **🧠 交叉熵损失 vs MSE 损失**
>
> - **交叉熵（Cross-Entropy）**：衡量两个**概率分布**之间的差异，适用于分类/离散问题
> - **MSE（Mean Squared Error）**：衡量两个**数值**之间的差异，适用于回归/连续问题
>
> RT-2 把动作预测变成了一个"分类问题"——从 256 个 bin 中选一个正确的。交叉熵适合这种离散化设置。
>
> 但交叉熵的代价是：**它忽略了 bin 之间的相对距离**。如果真实动作 bin=128，预测 bin=127（MSE 误差只有 1），交叉熵和 bin=100 的误差是一样的。连续动作中"差一点"的预测被过度惩罚。

---

## 3. 路线二：连续扩散（π₀ 采用）

### 3.1 扩散模型的动作输出

π₀ 没有把动作离散化，而是用 **Flow Matching** 直接建模连续动作分布 $p(a_{0:H} | s, \text{language})$。

**条件向量场定义：**

给定初始噪声动作 $a_1 \sim \mathcal{N}(0, I)$ 和条件 $(s, \text{language})$，π₀ 学习一个向量场 $v_\theta(a_t, t, s, c)$，将噪声在 $T$ 步内逐渐推向真实动作：

$$a_{t-1} = a_t + (a_1 - a_0) \cdot \frac{dt}{T} + v_\theta(a_t, t, s, c) \cdot dt$$

其中 $a_0$ 是真实动作，$a_1$ 是纯噪声，$\frac{dt}{T}$ 是步长。

### 3.2 Flow Matching 目标函数

π₀ 的训练目标是最小化向量场的 MSE：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,T],\; a_0 \sim p_{\text{data}},\; a_1 \sim \mathcal{N}(0,I)} \left\| v_\theta(a_t, t, s, c) - (a_1 - a_0) \right\|^2$$

> **🧠 直观理解 Flow Matching 的目标**
>
> $(a_1 - a_0)$ 是从数据点 $a_0$ 到噪声点 $a_1$ 的向量。Flow Matching 的目标是：让神经网络 $v_\theta$ 在任意中间时刻 $t$、任意中间状态 $a_t$ 下，都能准确预测这个向量的方向。
>
> 换句话说：$v_\theta$ 学习的是"在任意时刻，把任意中间状态直接拉向终点的最短路径"。这就是最优传输（Optimal Transport）的思想——所有粒子都走直线，速度恒定。

### 3.3 推理：Euler 方法采样

π₀ 的推理使用 **Euler 前向积分**：

```python
def euler_sample(v_theta, a_cond, T=50):
    a_T = torch.randn_like(a_cond)  # 纯噪声
    dt = 1.0 / T
    for t in reversed(range(T)):
        a_t = a_t - v_theta(a_t, t*dt, **a_cond) * dt
    return a_0  # 去噪后的动作
```

> **🧠 为什么叫"Euler"方法？**
>
> Euler 方法是最简单的微分方程数值积分：一阶近似 $\frac{da}{dt} = f(a, t)$，则 $a_{t+1} = a_t + f(a_t, t) \cdot \Delta t$。
>
> 这里 $f(a_t, t) = -v_\theta(a_t, t)$（因为是从 $t$ 向 $t-1$ 逆向积分），$\Delta t = 1/T$ 是固定步长。

---

## 4. 两条路线的系统对比

| | RT-2（离散） | π₀（连续扩散） |
|---|---|---|
| **动作表示** | 256 bin 离散整数 | 连续实数 |
| **损失函数** | 交叉熵（分类） | MSE（回归） |
| **推理速度** | 快（单次前向） | 中等（50步扩散） |
| **动作精度** | 受 bin 数量限制 | 无限精度 |
| **分布建模** | 单峰（softmax） | 多峰（扩散过程） |
| **跨具身迁移** | 较差（绝对角度差异大） | 较好（相对变化更稳定） |
| **训练稳定性** | 良好 | 中等（需要噪声调度） |

### 4.1 精度 vs 速度的权衡

**RT-2 的速度优势**：单次 Transformer 前向传播即可输出所有动作 token，推理延迟约 **10-50ms**（取决于模型规模）。

**π₀ 的精度优势**：连续动作理论上可以表示任意精度的动作值，不受离散 bin 的量化误差限制。对于精细操作（穿针、柔性物体操作），这可能是决定性的。

### 4.2 多峰分布的建模

真实世界中，一个状态可能对应多个合理的动作选择。例如：机械臂伸手去抓一个物体，既可以从左侧抓，也可以从右侧抓——两种动作都能完成任务。

- **RT-2 的 softmax 输出**：只能输出**一个最可能的动作**，无法表达多峰分布
- **π₀ 的扩散输出**：扩散过程可以自然地表达多峰分布（一次采样可能落在不同的峰上）

> **🧠 多峰分布在实际中有用吗？**
>
> 有。在需要"策略多样性"的场景中（如人机协作，机器人需要变换策略适应人类的不可预测行为），多峰输出能增加机器人的行为多样性。
>
> 但对于固定任务（如装配线），多峰反而可能导致不必要的策略变化。所以这是场景需求，不是绝对的优劣。

---

## 5. Action Head：动作输出的最后一层

无论离散还是连续，最终都需要一个 **Action Head** 把 Transformer 的隐表示映射成动作：

### 5.1 MLP Action Head（π₀ 采用）

```python
class ActionHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, horizon):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * horizon)
        )

    def forward(self, x):
        # x: [batch, hidden_dim]
        return self.mlp(x).view(-1, self.horizon, self.action_dim)
```

### 5.2 动作约束的处理

实际机器人的动作有物理约束（关节限位、速度上限）。处理方式有两种：

**方式 A：TanH 映射（π₀ 采用）**

```python
action = action_range_min + (action_range_max - action_range_min) * (torch.tanh(raw_action) + 1) / 2
```

TanH 输出在 $[-1, 1]$，线性映射到动作范围。

**方式 B：Sigmoid + 线性映射**

```python
action = action_range_min + (action_range_max - action_range_min) * torch.sigmoid(raw_action)
```

两种方式效果相近，TanH 更常用（梯度更稳定）。

---

## 6. 关键术语解释

| 术语 | 解释 |
|------|------|
| **动作维度 $D$** | 机器人动作空间的自由度数量 |
| **预测视野 $H$** | 一次性预测的动作步数 |
| **动作离散化** | 把连续动作值强制映射为有限个整数 bin |
| **256 bin** | RT-2 将每个动作维度分成 256 档 |
| **交叉熵损失** | 衡量两个概率分布差异的损失函数 |
| **MSE 损失** | 均方误差，衡量数值差异 |
| **Flow Matching** | 通过学习向量场来生成数据的方法 |
| **Euler 积分** | 最简单的微分方程数值积分方法 |
| **多峰分布** | 同一个输入对应多个可能输出的分布 |
| **Action Head** | VLA 模型末尾将隐表示映射为动作的 MLP 层 |
| **TanH 映射** | 用 TanH 将无约束输出映射到有界动作区间 |

---

## 7. 下一步预告

本系列已过半。接下来将进入：
- **第七篇**：VLA 的时序建模（单帧 → 多帧，RNN vs Transformer）
- **第八篇**：自动驾驶专篇 — DriveVLA、端到端规划
- **第九篇**：技术挑战 — 实时性、安全性、长尾场景

---

*参考文献：*
1. Brohan et al., **RT-2: Vision-Language-Action Models** (2023)
2. Black et al., **π₀: A Vision-Language-Action Flow Model** (2024)
3. Lipman et al., **Flow Matching for Generative Modeling** (2023)
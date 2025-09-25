---
title: "Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies"
draft: false
date: 2025-09-05
tags:
  - VLA
  - 论文阅读
  - Diffusion
---

# 论文信息

论文地址：https://arxiv.org/pdf/2508.20072

穆尧团队的工作。将离散扩散引入到VLA领域。让我疑惑的一点是模型是在经典Prismatic-7B上改造的，原本是个自回归模型因果注意力，是怎么改造成离散扩散这种双向注意力的。在模型图里并没有展示出对应的改造方法，论文里面对这块没有解释，目前代码也还没有开源。

# 研究内容

现在的VLA方法主要是两种范式：（1）自回归方法，顺序预测动作token：[[OpenVLA]]、 [[FAST|π0-FAST]]。（2）连续扩散方法，把整个动作轨迹当成连续信号来迭代去噪： [[Pi-0|π0]]、[[SmolVLA]]。连续扩散模型比自回归模型更好地建模多模态动作，但是和VLM骨干解耦。

![[Discrete Diffusion VLA overview.png]]

提出了Discrete Diffusion VLA，统一了视觉、语言、动作到一个Transformer中，保留了较强的动作建模能力。对于机器人操作任务，与连续扩散解码器不同，将动作生成保留在统一的Transformer中，采用与VLMs相同的训练目标（即交叉熵损失）。保留了大部分预训练的视觉和语言能力，类似于扩展词汇到新语言，同时提供了一种继承统一Transformer的缩放行为中的潜在路径，为大规模VLA研究打下基础。此外，离散扩散VLA打破了自回归解码器从左到右的瓶颈。动作片段在少量固定的步骤中并行解码，不确定的token可以通过迭代重新掩码，利用完整的跨模态上下文（包括动作间依赖）进行优化。

贡献如下：

- 提出了首个用于 VLA 的离散扩散动作头，将动作生成与视觉语言统一到一个Transformer中，同时保持高性能。
- 开发了一种先简单，后复杂的自适应解码策略，通过迭代重掩码，实现并行动作token解码和错误修正，提高准确性。
- 在 LIBERO 的 Franka Panda 上以及在 SimplerEnv 的 Google Robot 和 WidowX Arm 上验证了模型，在LIBERO上达到 96.3% 的平均成功率，在 SimplerEnv 中分别达到了 64.1% 和 49.3% 的成功率，优于自回归和连续扩散baseline。

# 模型

## 概览

离散扩散 VLA 是一个单一的Transformer视觉-语言-行动（VLA）策略，从多模态上下文预测固定长度的未来动作片段。给定图像观察（单视图或多视图）和语言指令，将每个连续控制维度离散化为token，并将其组装成固定长度的动作序列。一个单一的Transformer处理所有模态：来自预训练语言模型的语言token和来自预训练视觉编码器的视觉token，以及动作token。将动作生成视为在同时编码视觉和语言的同一个Transformer内部进行的掩码 token去噪。

在训练过程中，每一步对动作位置采样一个掩码比例，用掩码 token `[MASK]`替换选定的token，并优化交叉熵目标，以恢复掩码掉的token，使动作学习与离散扩散风格的破坏和去噪对齐。在推理过程中，将所有动作位置初始化为`[MASK]`，并运行少量并行优化迭代。采用自适应重新掩码策略进行多步降噪和优化，同时使用二次重新掩码策略进一步增强迭代间的一致性。

## 动作token的离散扩散

定义一个动作块为长度为 $L$ 的token序列 $\mathbf{a}_0=(a_{0,1},\dots,a_{0,L})$ ，其中每一个 $a_{0,i}\in\{1,\dots,K\}$ 是通过将连续控制（位置、朝向、夹爪）分成连续区间获得。在动作词汇中额外增加一个掩码 token M（代表`[MASK]`），使得一共有 $V=K+1$ 个符号和one-hot基 $\{\mathbf{e}_1,\dots,\mathbf{e}_K,\mathbf{e}_K\}$ 。

离散扩散的前向过程是一个马尔可夫链 $\{\mathbf{a}_t\}^T_{t=0}$ ，每步的转移矩阵 $\mathbf{Q}_t\in\mathbb{R}^{V\times V}$ 独立地映射每个token到M（概率为 $\beta_t$ ）或者是不变（概率为 $1-\beta_t$ ）。正式地来说，对于token $a_{t,i}$ 的one-hot向量 $\mathbf{e}_{a_{t,1}}$ ：

$$
\mathbf{Q}_t\mathbf{e}_{a_{t,i}}=(1-\beta_t)\mathbf{e}_{a_{t,i}}+\beta_t\mathbf{e}_M
$$

将转移矩阵组合起来得到 $\bar{\mathbf{Q}_t}=\mathbf{Q}_t\dots\mathbf{Q}_t$ ，在时间 $t$ 被破坏的分布可以按位置分解为：

$$
q(\mathbf{a}_t \mid \mathbf{a}_0) = \prod_{i=1}^{L} \text{Categorical}(a_{t,i} \mid \bar{\mathbf{Q}}_t \mathbf{e}_{a_0, i})
$$

其中 $L$ 是动作块token的长度。

反向过程定义了在多模态上下文 $\mathbf{c}$ 条件下的 $p_\theta(\mathbf{a}_{t-1}\mid\mathbf{a}_t,\mathbf{c})$ 。根据贝叶斯定律，对于每一个位置 $i$ ：

$$
p_{\theta}(a_{t-1,i} \mid a_{t,i}, \mathbf{c}) \propto q(a_{t,i} \mid a_{t-1,i}) p_{\theta}(a_{t-1,i} \mid \mathbf{c})
$$

其中在掩码破坏公式下，可以推出：

$$
p_{\theta}(a_{t-1,i} \mid a_{t,i}, \mathbf{c}) = \begin{cases}
\delta(a_{t-1,i} = a_{t,i}), & a_{t,i} \neq \text{M} \\
\text{Categorical}(a_{t-1,i} \mid \pi_{\theta}(i \mid \mathbf{c})), & a_{t,i} = \text{M}
\end{cases}
$$

其中 $\pi_\theta(i\mid\mathbf{c})$ 是模型基于目前掩码后的序列 $\tilde{\mathbf{a}}_t$ 和 $c$ 的预测分布。在每一步，离散扩散VLA只恢复一部分被掩码的位置，保留剩余的被掩码，慢慢降低掩码率，最终到达 $\mathbf{a}_0$ 。

在实现过程中，采用掩码扩散/填充方法，并将多步骤链合并为一个掩码后token的预测目标。设置一个与时间 $t$ 相关的掩码比例 $\gamma_t\in(0,1]$ ，将选定动作位置替换为特殊token`[MASK]`，从而得到 $\tilde{\mathbf{a}}_t$ ，训练一个Transformer $f_\theta$ 来预测原始token，在被掩码的位置利用交叉熵计算损失：

$$
\mathcal{L}_{\text{CE}}(\theta) = -\sum_{i \in \mathcal{M}_{\gamma_t}} \log p_{\theta}(a_{0,i} \mid \tilde{\mathbf{a}}_t, \mathbf{c}), \quad p_{\theta}(\cdot) = \text{softmax}(W f_{\theta}(\tilde{\mathbf{a}}_t, \mathbf{c}))
$$

其中 $\mathcal{M}_{\gamma_t}$ 是被掩码的集合， $W$ 将动作位置的隐藏状态映射到 $K$ 路动作词汇表中。训练目标保留了扩散的破坏-去噪特性，同时使用了一个简单的最大似然替代函数。在推理阶段，通过并行解码和自适应重掩码来实现反向过程：（1）将所有 $L$ 个动作位置初始化为 `[MASK]`；（2）并行预测所有位置的token分布；（3）保留高置信度的预测并重掩码不确定部分；（4）重复进行少量迭代。

Discrete Diffusion VLA 通过增加训练时间复杂度（即解决指数数量的填充任务）来获得测试时的任意顺序解码，并自适应选择推理顺序（例如，通过置信度或置信差距），这与 BERT 风格的掩码语言模型不同，后者在一次运行中使用固定的小掩码比，缺乏有原则的生成逆链。

## 统一的VLA架构

![[Discrete Diffusion VLA model.png]]

### 基本架构

利用OpenVLA作为表示骨干（Prismatic-7B，SiGLIP+DINOv2作为视觉编码器，Llama2作为语言模型骨干）。和原始的自回归动作头不一样，离散扩散VLA在处理视觉和语言的同一Transformer中嵌入离散扩散，从而形成一个统一的Transformer作为 VLA。

### 分词和动作分块

和OpenVLA一样，采用了每个维度256块百分比（1%-99%）方案，把夹爪作为一个二进制token。单个时间步的动作包含 $D_{act}=7$ 个token：3个位移，3个旋转，1个夹爪。对于动作分块，将未来 $H$个时间步的token排列成固定的布局，一共有 $L=H\times D_{act}$ 动作位置。还在动作词汇表中添加了一个特殊的`[MASK]` token用于离散扩散。

视觉输入包括强制的第三人称摄像头和可选的左右腕部摄像头。每个视角由SigLIP和DINOv2 ViT编码，特征投影到语言模型嵌入空间；语言指令由语言模型直接分词。可选的本体状态（如末端执行器位置和关节角度）通过一个小MLP嵌入并连接。最终的输入序列为`[视觉token；语言token；动作token]`，其中在训练过程中，在 $L$ 个动作位置中随机选择一部分被替换为`[MASK]`。

### 统一的Transformer和头

所有token都通过单个Transformer进行处理，且在动作位置上仅预测数值。对于动作token，使用双向注意力掩码，这意味着没有因果约束，每个动作位置可以关注所有视觉-语言token。这种设计无需定制适配器，即可自然实现全模态融合。动作位置的隐藏状态通过共享分类头投影到256维的值。视觉和语言使用标准VLM掩码和头（若使用的话）。

与之前的两个动作头设计相比，这种统一的骨干保留了语言模型的表示能力，可无缝扩展到任意模型大小，并支持并行解码。在推理阶段，自适应重新掩码进一步优化不确定token，将Transformer的全局上下文优势与扩散模型的迭代去噪特性相结合。

### 算法流程

- 训练流程

在训练过程中，针对每个minibatch，首先从一个模拟扩散时间的调度（例如线性或余弦）中随机抽取一个掩码比例 $\gamma \in(0,1]$ 。然后用特殊token `[MASK]`替换 $\gamma L$ 个动作位置，并根据最小化这些动作索引上的掩码交叉熵。采用硬标注监督，将掩码索引处的真实值表示为one-hot向量。视觉和语言token保持不变（可应用标准的语言模型掩码/打包方法处理）。这保留了预训练的VLM能力，因为目标与语言模型训练兼容，同时通过掩码调度将动作学习与离散扩散相结合。由于模型预测的是长度为 $L=H\times D$ 的固定长度的动作片段，因此所有序列的长度均保持统一，无需`[EOS]`填充。通过对 $L$ 个位置进行联合优化，可以自然地一次性训练动作片段。

- 推理流程

在测试时，将所有 $L$ 个动作位置初始化为 `[MASK]`，并执行少量并行优化迭代。在每个迭代 $t$ 时，模型预测当前所有掩码位置的token后验。然后，通过多项式采样，从预测的数值中为每个位置抽取候选token。接着，根据预设的调度策略，将 $\gamma_t$ 设置为当前掩码比例，并用它来确定剩余的掩码位置数量。通过数据相关的评分（例如，最大置信度或置信差距）对掩码位置进行排序，将前 $(1 − \gamma_t)$ 的采样的token变更，并将剩余的 $\gamma_t$ 比例的继续掩码。这种调度方法使得解码顺序适应每个实例，而非固定不变。一个轻量级的二次重掩码机制进一步对先前取消掩码的token进行阈值和一致性检查，以防止错误传播。

### 自适应解码和二次重掩码

- 自适应解码机制

推理流程从一个完全掩码的动作块 $\mathbf{a}_1=M^L$ ，其中掩码率 $\gamma_1=1$ 。接着经过 $T$ 个改善步，利用单调调度策略 $\gamma_{t+1}<\gamma_t$ 。这里用余弦调度策略：

$$
\gamma_t = \cos\left(\frac{\pi}{2} t\right), \ t \in [0, 1)
$$

在 $t$ 步，模型得到每个位置的后验 $p_\theta(\mathbf{a}_{t-1}\mid\mathbf{a}_t,\mathbf{c})$ 。利用其中一个自适应指标来对每个位置 $i$ 进行打分：

$$
s_{t,i} = \max_{k} p_{\theta}(k \mid \mathbf{a}_t, \mathbf{c}) \ (\text{Max Confidence})
$$

$$
g_{t,i} = p_{\theta}(k_{(1)} \mid \cdot) - p_{\theta}(k_{(2)} \mid \cdot) \ (\text{Confidence Gap})
$$

其中 $k_{(1)},k_{(2)}$ 是对应了最高和第二高的概率的token。令 $m_{t,i}\in\{s_{t,i},g_{t,i}\}$ ，保留前 $(1-\gamma_{t+1})L$ 个位置：

$$
\mathcal{K}_t = \underset{(1 - \gamma_{t+1})L}{\text{argtop}} m_{t,i}
$$

并且将这些位置的token通过调整温度的Gumbel采样更新，以鼓励探索。

$$
a_{t+1,i} \sim \text{Categorical}\left( \text{softmax}\left( \frac{\log p_{\theta}(\cdot \mid \mathbf{a}_t, \mathbf{c}) + \varepsilon}{\tau_t} \right) \right), \ i \in \mathcal{K}_t
$$

其中 $\varepsilon$ 有 $\text{Gumbel}(0,1)$ 独立同分布成分（等价为Gumbel-Max)， $\tau_t$ 是随 $\gamma_t$ 衰减的温度。不在 $\mathcal{K}_t$ 中的位置设为 M，这个过程持续迭代直到 $\gamma_T=0$ 或者收敛。

- 二次重掩码

除了达到目标比例 $\gamma_{t+1}$ ，对先前变更的token进行两次轻量级检查，以防止早期错误持续存在。未能通过检查的token会被重新屏蔽，这一步骤称为二次重掩码。

1. 阈值检查

如果当前的置信度降至一个单调递增的步骤阈值 $\eta_t^{abs}$ 以下，则会重新对该token进行掩码处理。

$$
\mathcal{R}_t^{\text{abs}} = \left\{ i \in \mathcal{K}_t : s_{t,i} < \eta_t^{\text{abs}} \right\}
$$

1. 残值下降检查

令 $t_i^*$ 表示第 $i$ 步时保留的位置，缓存参考置信分数 $s^{ref}_i := s_{t_i^*,i}$ 。计算置信残值 $\Delta_{t,i} = s^{ref}_i − s_{t,i}$ 。置信度降低大的token要么通过阈值规则，要么通过top-Q 规则重新掩码：

$$
\mathcal{R}_t^{\text{drop}} = \left\{ i \in \mathcal{K}_t : \Delta_{t,i} > \eta_t^{\text{drop}} \right\} \ \text{or} \ \mathcal{R}_t^{\text{drop}} = \underset{Q}{\text{argtop}} \Delta_{t,i}
$$

二次重掩码的集合为 $\mathcal{R}_t=\mathcal{R}_t^{\text{abs}} \cup \mathcal{R}_t^{\text{drop}}$。对于 $i\in\mathcal{R}_t$，在 $t+1$步之前将 $a_{t+1,i}=\text{M}$。

# 实验

## 评估基准和基线

在以下环境评估：

- Franka Panda in LIBERO
- Google Robot in SimperEnv-Fractal
- WidowX Robot in SimplerEnv-Bridge

![[Discrete Diffusion VLA result1.png]]

对比以下模型：

- 自回归动作解码器：RT-1-X/RT-2-X、OpenVLA、[[Octo]]-Small/Octo-Base、HPT、TraceVLA、[[SpatialVLA]]、[[OpenVLA-OFT]] (Discrete)和 $\pi_0$+FAST
- 连续扩散/流匹配：[[Diffusion Policy]] 、MDT、 DiT Policy (Dita) 、 RoboVLM 、 $\pi_0$ 、 OpenVLA-OFT (Cont.-Diffusion) 和 GR00T-N1

LIBERO评估结果

![[Discrete Diffusion VLA result2.png]]

SimplerENv Google Robot评估结果

![[Discrete Diffusion VLA result3.png]]

SimplerENv WidowX Robot评估结果

![[Discrete Diffusion VLA result4.png]]

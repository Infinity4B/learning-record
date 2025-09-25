---
title: "WorldVLA: Towards Autoregressive Action World Model"
draft: false
date: 2025-09-19
tags:
  - VLA
  - 论文阅读
---

# 论文信息

论文地址：https://arxiv.org/pdf/2506.21539

项目地址：https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA

代码地址：hhttps://github.com/alibaba-damo-academy/WorldVLA

阿里达摩院。应该是首个将世界模型和动作模型结合的VLA，兼具理解和生成能力。

# 研究内容

![[WorldVLA overview.png]]

VLA模型在机器人任务中有很强的泛化性，但是对于动作理解不深。世界模型能够根据当前观察和动作预测未来状态，对于视觉信息和行为动力学有着很深的理解。不过世界模型不能直接输出动作，限制了应用。

提出了WorldVLA，将动作和图片理解和生成结合到一起。主要贡献如下：

- 提出WorldVLA，一个统一动作和图像理解与生成的自回归动作世界模型
- 在自回归模型中引入了一种动作注意力掩码策略，用于动作块生成任务，解决在生成多个动作序列时动作错误累积的问题
- 实验表明WorldVLA优于独立的动作和世界模型，突出显示了世界模型和动作模型之间的相互增强。此外，动作注意力掩码策略解决了生成动作片段时的性能下降，并显著提高了抓取性能。

# 模型

## 问题定义

有两个重要组件：动作模型（或者叫策略模型） $\pi_\theta$ 和一个世界模型 $f_\phi$ 。动作模型 $\pi_\theta$ 用于根据历史图片观察 $\{o_{t - h}, o_{t - h + 1}, \dots, o_t\}$ 和语言指令 $l$ 生成动作 $a_t$ ：

$$
a_t = \pi_{\theta}(a_t \mid o_{t-h:t}, l)
$$

同时，世界模型 $f_\phi$ 基于历史观察 $\{o_{t - h}, o_{t - h + 1}, \dots, o_{t-1}\}$ 和对应的动作序列 $\{a_{t - h}, a_{t - h + 1}, \dots, a_{t-1}\}$ 预测下一帧 $o_t$ ：  

$$
o_t = f_{\phi}(o_t \mid o_{t-h:t-1}, a_{t-h:t-1})
$$

目标是构建一个集成的动作世界模型 $M_{\psi}$ 来统一这两个功能。模型需要能够同时作为策略模型预测动作也能作为世界模型预测未来状态：

$$
M_{\psi}: \begin{cases}
a_t = M_{\psi}^{\text{policy}}(a_t \mid o_{t-h:t}, l), \\
o_t = M_{\psi}^{\text{world}}(o_t \mid o_{t-h:t-1}, a_{t-h:t-1})
\end{cases}
$$

其中 $M_{\psi}^{\text{policy}}$ 代表了动作生成部分， $M_{\psi}^{\text{world}}$ 代表世界状态预测部分。

## 架构

![[WorldVLA model.png]]

模型从Chameleon初始化，这是一个统一图片理解和生成模型。模型有三个分词器：

- 图片分词器：VQ-GAN模型，额外有对特殊图片区域的感知损失。压缩比是16，码表长度是8192。根据256x256图片生成256个token，根据512x512图片生成1024个token
- 动作分词器：离散化每一个动作维度，分为256个区间。共用7个token表示动作，3个代表相对位置，3个代表相对角度和一个夹爪状态
- 文本分词器：训练好的BPE分词器，词汇表大小为65536，包含8192和图片token和256个动作token

## 训练方法

将动作模型数据和世界模型数据混合在一起，用于训练WorldVLA。

### 动作模型数据

输入文本为："What action should the robot take to + task instruction + ?”，整个token序列为：

$$
\texttt{[BOS]\{text\}}\underbrace{\texttt{[BOI]\{image\}\dots\{image\}[EOI]}}_{\times M}\texttt{[EOS]}\overbrace{\underbrace{\texttt{[BOA]\{action\}\dots\{action\}[EOA]}}_{\times K}\texttt{[EOS]}}^{\mathcal{L}_{action}}
$$

输入包含 $M$ 张图片，输出包含 $K$ 个动作。只计算动作token $\mathcal{L}_{action}$ 的损失。

### 世界模型数据

世界模型根据当前图像观察和动作生成下一个图像帧。它不需要任务指令，因为动作本身就完全决定了下一个状态。输入文本为：”Generate the next frame based on the current image and the action."，整个token序列为：

$$
\texttt{[BOS]\{text\}}
\underbrace{ % Outer brace starts here
  \texttt{[BOI]\{image\}[EOI][BOA]\{action\}[EOA][EOS]} % First part inside outer brace % Inner brace starts here
    \overbrace{\texttt{[BOI]\{image\}[EOI][EOS]} % Content for inner brace
   % Label for inner brace
}^{\mathcal{L}_{world}}}_{\times N}
$$

基于动作的下一帧预测重复 $N$ 次，只计算生成的图片token $\mathcal{L}_{world}$ 的损失。

### 注意力掩码

![[WorldVLA attention.png]]

(a)展示了典型的自回归模型中的标准注意力机制采用的是因果注意力掩码，该掩码仅允许当前token访问前面token的信息，而排除了后续token。然而，这种常规配置在生成动作片段（即连续多项动作）方面表现不佳。虽然基础MLLM由于在多样化数据集上进行了大规模预训练，在图像和文本领域表现出强大的泛化能力，但在动作领域有效泛化的能力相对有限。因此，先前的动作错误会在默认注意力掩码下传播到后续的动作，导致性能下降。为解决这一限制，引入了一种专为动作生成设计的替代注意力掩码，如(b)所示。这种修改后的掩码确保当前的动作仅依赖于文本和视觉输入，同时禁止访问先前的动作。这种设计使自回归框架能够并行生成多个动作。世界模型部分采用的是传统的因果注意力掩码，如图(c)所示。

### 训练目标

损失函数为：

$$
\mathcal{L}=\mathcal{L}_{action}+\alpha\mathcal{L}_{world}
$$

其中两个都是交叉熵损失函数。

# 实验

## 训练设定

动作模型采用默认的输入图像数量 $M = 2$ 。在LIBERO Long任务中，动作块的规模设为 $K = 10$ ；在其余三个LIBERO任务中， $K = 5$ 。为最大限度地减少计算开销，世界模型在默认配置下仅运行一次 $N = 1$ 。实验设置中，参数 $\alpha$ 固定为0.04。

## 评估指标

- 对于动作模型评估，每个任务在不同初始状态下进行50次实验，并记录成功率（SR）
- 对于世界模型评估，使用验证集并记录FVD、PSNR、SSIM和LPIPS值

## 评估结果

LIBERO评估结果

![[WorldVLA result1.png]]

动作模型消融实验

![[WorldVLA result2.png]]

世界模型消融实验

![[WorldVLA result3.png]]

动作块长度消融实验

![[WorldVLA result4.png]]

动作世界模型与动作视频预测模型的对比

![[WorldVLA result5.png]]

历史图片输入长度消融实验

![[WorldVLA result6.png]]

世界模型预训练消融实验

![[WorldVLA result7.png]]

---
title: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model"
draft: false
date: 2025-08-29
tags:
  - VLA
  - 论文阅读
---

# 论文信息

论文地址：https://arxiv.org/pdf/2501.15830

项目地址：https://spatialvla.github.io/

代码地址：https://github.com/SpatialVLA/SpatialVLA

RSS 2025。一篇比较早研究3D空间感知理解的VLA论文。

# 研究内容

目前的VLA基本都是基于2D观察输入，缺乏3D的感知理解。想要研究如何让VLA具备一些3D物理世界的空间理解能力。亮点在于建模在三维空间，从而将自回归方式所需输出的token减少，提高推理速度。

难点：

- 不同机器人具身观察3D不对齐
- 不同的机器人有不同的动作特性

提出了一个通用机器人策略SpatialVLA，通过自我为中心的3D位置编码集成3D空间信息和语义信息。对于机器人动作，利用自适应动作网格整合动作空间，离散化连续机器人动作，根据统计数据分布生成空间网格。

![[SpatialVLA overview.png]]

# 模型

![[SpatialVLA model.png]]

SpatialVLA接收图片观察 $\mathbf{o}_t=\{\mathbf{I}_t^1,\dots,\mathbf{I}_t^n\}$ 和一个自然语言任务描述 $\mathbf{L}$ 作为输入，学习一个映射函数 $\tau(\cdot)$ 来生成一系列机器人动作 $\mathbf{A}_t=[\mathbf{a}_t,\mathbf{a}_{t+1},\dots,\mathbf{a}_{t+H-1}]$ ，即 $\mathbf{A}_t=\mathcal{F}(\mathbf{o}_t,L)$ 。为了增强模型对3D空间的理解，用机器人特定的3D感知输入和输入来增强VLM骨干，即自我为中心的3D位置编码和自适应动作网格。自我为中心的3D位置编码表示 $\mathbf{O_{3d}}$ 目标是通过整合3D空间信息和2D语义信息捕捉3D场景结构。自适应动作矩阵想要通过一系列空间动作token $\mathfrak{a}=\{\mathfrak{a}^1,\dots,\mathfrak{a}^V\}$ 来表示连续机器人动作 $\mathbf{a}$ 。在训练时，SpatialVLA训练接收自我3D位置编码 $\mathbf{O_{3d}}$ 以及自然语言任务描述 $\mathbf{L}$ 作为输入，自回归生成空间动作token $\tilde{a}_t$ ，损失函数是交叉熵：

$$
\mathfrak{L}(\theta)=\mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t)}\mathcal{L}(\mathfrak{a}_t,\tilde{\mathfrak{a}}_t))
$$

其中预测出的动作token $\tilde{\mathfrak{a}}_t=\tau({\mathbf{O_{3d}},\mathbf{L}})$ 转换为连续动作信号 $\mathbf{a}_t$ 用于机器人控制。

## 模型设计

### 自我3D位置编码

提出的自我3D位置编码从相机帧中整合了深度信息和图片像素来构建一个自我为中心的3D坐标系统，避免了机器人与相机的外参矫正，并且适用于各种机器人设定。用ZoeDepath来计算深度图 $D$，并且通过反投影 $\pi^{-1}$ 获取自我为中心的3D坐标系统的像素的3D位置 $\mathbf{p}=\{x,y,z\}$ 。首先用SigLip视觉编码器提取2D语义视觉特征 $\mathbf{X}\in\mathbb{R}^{d\times h\times w}$ 来继承视觉和语言之间的对齐，并且在自我为中心的3D坐标系统中计算对应的3D位置 $\mathbf{P}\in\mathbb{R}^{3\times h\times w}$ 。自我为中心的3D位置 $\mathbf{P}$ 通过一个正弦函数 $\gamma(\cdot)$ 以及一个可学习的MLP后被编码成3D位置嵌入 $\mathbf{P'}\in\mathbb{R}^{d\times h\times w}$ 。自我为中心的3D空间表示 $\mathbf{O_{3d}}\in\mathbb{R}^{d\times h\times w}$ 是通过把3D位置嵌入 $\mathbf{P'}$ 和2D路径视觉token $\mathbf{X}$ 相加得到的：

$$
\mathbf{O_{3d}}=\mathbf{X}+\mathbf{P'}=\mathbf{X}+\text{MLP}(\gamma(\mathbf{P}))
$$

### 自适应动作网格

设计自适应动作网格把连续机器人动作转换为分词后的类别用于预测。对于单臂机器人，7个维度动作 $\mathbf{a}=\{\text{x},\text{y},\text{z},\text{roll},\text{pitch},\text{yaw},\text{grip}\}$ ，分为三个部分：

$$
\mathbf{a}=\{\mathbf{a_{trans}},\mathbf{a_{rot}},\mathbf{a_{grip}}\}
$$

其中 $\mathbf{a_{trans}}=\{\text{x},\text{y},\text{z}\}$表示了位移动作 $\Delta\mathbf{T}$ ， $\mathbf{a_{rot}}=\{\text{roll},\text{pitch},\text{yaw}\}$ 表示了旋转动作 $\Delta\mathbf{R}$ ， $\mathbf{a_{grip}}=\{\text{grip}\}$ 表示了两个离散token代表了打开和关闭夹爪的动作。此外，将位移动作 $(\text{x},\text{y},\text{z})$ 转成极坐标 $(\phi,\theta,r)$ 从而分离运动方向 $(\phi,\theta)$ 和距离 $r$ 。

![[SpatialVLA model2.png]]

对每个机器人环境，将每个动作变量归一化到 $[-1,1]$ ，统计位移和旋转 $\Delta\mathbf{R}=\{\text{roll},\text{pitch},\text{yaw}\}$ 以及 $\Delta\mathbf{T}=\{\phi,\theta,r\}$ ，接着用参数化高斯分布拟合 $\mathcal{N}(\mu^a,\Sigma^a)$ 。连续的动作分成 $M$ 个间隔 $\mathbf{G}_{i=1,\dots,M}=\{[a_1=-1,a_2],\dots,[a_{M-1},a_M=1]\}$ ，在每个归一化的动作变量中概率相等都是 $1/M$ ，即：

$$
a_2,\dots,a_M=\argmin_{a_2,\dots,a_M}\int_{a_i}^{a_{i+1}}f(x)dx-1/M,i=1,\dots,M
$$

其中 $f(x)$ 是高斯分布的概率密度函数。在 $\{\phi,\theta\}$ 中分配了更多的网格而不是在移动距离 $r$ 上，因为需要捕捉精细的动作方向。假设 $M_\phi,M_\theta,M_r$ 是 $(\phi,\theta,r)$ 的离散区间的数量，那么位移空间共有 $M_{trans}=M_\phi\cdot M_\theta\cdot M_r$ 个离散空间网格 $\mathfrak{a}_{trans}=\{\mathfrak{a}^1,\dots,\mathfrak{a}^{M_{trans}}\}$ 。相似地，在3D空间中有 $M_{rot}=M_{roll}\cdot M_{pitch}\cdot M_{yaw}$ 个3D离散空间网格 $\mathfrak{a}_{rot}=\{\mathfrak{a}^1,\dots,\mathfrak{a}^{M_{rot}}\}$ 。那么相关的可学习空间动作token嵌入可以定义为：

$$
\mathbf{E}_\mathfrak{a}=\{\mathbf{E}_{trans},\mathbf{E}_{rot},\mathbf{E}_{grip}\}
$$

其中 $\mathbf{E}_{trans}\in\mathbb{R}^{d\times M_{trans}}$ ， $\mathbf{E}_{rot}\in\mathbb{R}^{d\times M_{rot}}$ ， $\mathbf{E}_{grip}\in\mathbb{R}^{d\times 2}$ 表示了位移、旋转和夹爪动作，总的动作token共 $V=M_{trans}+M_{rot}+2$ 个。在训练后，这些可学习的空间动作token捕捉了通用机器人动作知识，可以用于新机器人具身适应。值得注意的是，该模型仅需为单步机器人动作生成3个token，而[[RT-1]] 、[[RT-2]] 和[[OpenVLA]] 则需生成7个token，从而实现了快速模型推理速度。

## 预训练和后训练方案

### 预训练步骤

预训练数据包括OXE的子集和RH20T数据集，参考OpenVLA改变了混合权重。在预训练开始时空间动作token嵌入 $\mathbf{E}_\mathfrak{a}$ 和MLP参数都是随机初始化，在训练过程中和Paligemma2的视觉编码器和LLM骨干一起优化。每个训练步中在打乱后的示例 $\{\zeta_i,\dots,\zeta_j\}$ 中从随机时间步 $t_1,\dots,t_B$ 中抽取一批动作对 $[(\mathbf{o}_{t_1},\mathbf{A}_{t_1},\mathbf{L}_i),\dots,(\mathbf{o}_{t_B},\mathbf{A}_{t_B},\mathbf{L}_j)]$ 。SpatialVLA训练目标是下一token预测的自回归目标。为了保留预训练VLM的世界知识，文字token $\mathbf{E}_{text}$ 的嵌入被冻结。

### 后训练设计

以往工作采用全参数或者LoRA微调。这篇文章主要研究空间嵌入适应，提出了一种新的有效的后训练方法。具体来说，对每个后训练数据集的动作变量拟合一个新的高斯分布 $\mathcal{N}(\mu_{new},\Sigma_{new})$ ，并且在位移和旋转移动创建离散空间动作网格 $\mathbf{G}_{new}$ 从而构建动作网格 $\mathbf{G}_{new}$ 和token $\mathfrak{a}_{new}$ ，其中新的空间动作token $\mathbf{E}_{\mathfrak{a}_{new}}$ 是根据预训练动作token $\mathbf{E}_\mathfrak{a}$ 进行三线性插值初始化的。这些动作token嵌入 $\mathbf{E}_{\mathfrak{a}_{new}}$ 和模型参数通过下一token预测目标优化。

对于新的空间动作网格 $\mathbf{G}_{new}$ ，假设在质心为 $(\phi_{new}^i,\theta_{new}^i,r_{new}^i)$ 的位移空间 $\mathbf{a}_{trans}^{new}$ 第 $i$ 个3D网格 $\mathbf{G}_{new}^i$ 以及预训练动作网格中其相邻的动作网格 $\mathbf{G}^{adj}=\{\mathbf{G}^1,\dots,\mathbf{G}^K\}$ 。第 $i$ 个动作token $\mathbf{e}^i_{\mathfrak{a}_{new}}$ 的嵌入通过基于 $\mathbf{G}^{ad}$ 的三线性插值初始化：

$$
\mathbf{e}^i_{\mathfrak{a}_{new}}=\sum_{j=1}^Kw_j\mathbf{e}^j
$$

其中 $\mathbf{e}^j\in\mathbb{R}^d$ 是预训练动作网格的嵌入， $w_j$ 是通过计算质心和相邻质心的归一化距离得出。新的旋转动作token $\mathbf{E}_{\mathfrak{a}_{rot}}$ 也是用这样的方式初始化。

# 实验

评估了SpatialVLA在7种不同机器人学习场景中的能力，共包括24个真实机器人任务和3个模拟环境。首先，在SimplerEnv 模拟环境和真实世界WidowX机器人平台（BridgeV2 设置）中评估了SpatialVLA，测试了其在不同机器人上的即插即用控制能力，这些设置与预训练数据集一致。其次，评估了本方法在模拟和真实世界设置中的微调效果，包括LIBERO 和新的Franka机器人设置，以适应新的机器人环境和任务。然后，设计了4个特殊任务，这些任务需要在2种不同的真实世界机器人环境中进行精确的空间理解，以测试SpatialVLA的空间表示的有效性。最后，对Fractal和BridgeV2的数据集进行了全面的消融实验，以验证在SpatialVLA中的设计决策。

## 实现细节

SpatialVLA 模型基于 110 万个真实机器人演示数据预训练而成，这些数据来自 OXE 和 RH20T 数据集，在 64 台 A100 GPU 的集群上进行了 10 天的训练，批大小为 2048。使用AdamW，学习率为2e-5，线性调度，无权重衰减，预热比例为0.005，ZeRO1。训练分为两个阶段。第一阶段在完整数据集上训练模型16万步，准确率达90%。第二阶段去除DROID数据集后，模型进行了4万步的微调，准确率超过95%。

![[SpatialVLA result1.png]]

对于输入的机器人观测数据，SpatialVLA 策略仅基于一个第三人称摄像头进行训练，并通过一张图像构建以自我为中心的三维空间表示。对于输出的机器人动作，SpatialVLA 预测 $T = 4$ 个未来动作（总共 $V=8194$ 个动作 token 中的 12 个空间动作 token），并在预测下一个动作之前执行这些动作。

在推理过程中，SpatialVLA 需要 8.5GB 的 GPU 内存，在单个 NVIDIA RTX 4090 GPU 上运行速度约为 20Hz，以在模拟和现实环境中进行评估。

## 零样本机器人控制

### 真机评估

在BridgeData V2评估中的真实世界WidowX机器人平台上进行了实验。设计了七个任务套件，涵盖语言基础、语义理解（未知背景和姿势）以及运动干扰（手动移动物体）。所有通用操作策略，包括[[Octo]]、RT-1-X、OpenVLA和RoboVLM，在七个任务套件中进行了评估，每个任务包含11次尝试，共进行了77次尝试。

![[SpatialVLA result2.png]]

### 模拟环境评估

SimplerEnv的Google Robot评估结果

![[SpatialVLA result3.png]]

SimplerEnv的WidowX评估结果

![[SpatialVLA result4.png]]

## 适应新机器人设定评估

在LIBERO上进行SpatialVLA的评估。和OpenVLA一样，在四个任务集上进行了实验，每个任务集包含10个任务，每个任务有50个遥操演示。这些任务集评估了模型对空间关系的理解能力（LIBERO-Spatial）、物体类型（LIBERO-Object）、任务导向行为（LIBERO-Goal）以及对包含多样化物体、布局和目标的长期任务的泛化能力（LIBERO-Long）。将SpatialVLA与几种通用操作策略方法进行了比较，包括[[Diffusion Policy]]、Octo、OpenVLA和TraceVLA。SpatialVLA在对应数据集上进行了200轮微调，使用LoRA（ $r=32,\alpha=32$ ）。

为了进行更全面的评估，设计了13个Franka任务，以验证模型的操纵性能。评估分为三个设置：**单任务**，包括四个基本任务：选择、放置、推和关闭；**指令跟随**，涉及根据语言指令在同一场景中操作不同物体；**多任务**，涉及将所有四个单任务数据混合训练，并在此类任务上进行测试。将SpatialVLA与主流策略（包括Diffusion Policy、Octo和OpenVLA）进行比较。

![[SpatialVLA model3.png]]

LIBERO评估结果

![[SpatialVLA result5.png]]

## 空间理解能力评估

通过三个机器人设定来评估SpatialVLA的空间理解能力：Franka机器人微调设定，WidowX机器人零样本设定，以及Lifero-Spatial微调设定。这些任务具有不同的空间复杂度，其中Franka任务涉及提示理解，WidowX任务涉及显式高度变化，以及Lifero-Spatial任务涉及物体布局变化。七种主流策略，即Diffusion Policy、Octo、RT-1-X、OpenVLA、TraceVLA和RoboVLM，用于比较。

![[SpatialVLA result6.png]]

## 消融实验

### 混合数据集预训练

![[SpatialVLA result7.png]]

在一个结合了Google Fractal和BridgeData V2的混合数据集上进行预训练。所有模型都是从头开始在8个A100 GPU上进行训练的，批大小为128，步数为12万。分别在SimplerEnv选取了2个任务，蓝色为Google Robot，黄色为WidowX Robot。

### 领域数据集后训练

![[SpatialVLA result8.png]]

分别在Google Fractal和BridgeData V2上进行微调，并比较在小规模LIBERO数据集上进行全参数微调和LoRA微调的效果。空间嵌入调整是指从新数据集中分割出高斯分布的空间网格，并用这些网格更新空间特征嵌入。

---
title: "CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation"
draft: false
date: 2025-08-29
tags:
  - VLA
  - 论文阅读
  - Diffusion
---

# 论文信息

论文地址：https://arxiv.org/pdf/2411.19650

项目地址：https://cogact.github.io/

代码地址：https://github.com/microsoft/CogACT

一个VLM+动作头的模型，将VLM出来的特征当作动作模型的条件进行动作生成。对[[ACT]]的动作块进行了改进。

![[CogACT overview.png]]

# 研究内容

目前的VLA用VLM预测动作方法比较简单，影响了任务性能。[[RT-2]]和[[OpenVLA]]将连续动作映射到离散区间，限制了动作精度。RoboFlamingo引入了额外的LSTM动作头，把VLM输入转换为动作，但是忽视了动作的概率和多模态性质。

提出一个新的VLA架构，没有将预训练的VLM用于动作预测，而是利用VLM提取的认知信息来指导专用动作模块的动作预测过程，利用DiT作为动作模块。

想法是解耦认知和动作能力，通过端到端的训练或微调，协同提升认知和行动能力。因此，提出的方法称为CogACT。

# 模型

## 问题定义

给定语言指令 $l$ 和时间 $t$ 下的视觉观察 $\mathbf{o}_t$ ，模型 $\pi$ 预测时序动作序列 $(a_{t},a_{t+1},\dots,a_{t+N})$ 来执行目标任务：

$$
\pi:(l,o_t)\to(a_{t},a_{t+1},\dots,a_{t+N})
$$

虽然通常来说 $a_t$可以描述不同的机器人动作，这里考虑7自由度带夹爪的动作空间：

$$
a_t=[\Delta x,\Delta y,\Delta z, \Delta\phi,\Delta\theta,\Delta\psi,g]
$$

其中 $\Delta x ,\Delta y ,\Delta z$ 是末端执行器的相对偏移量， $\Delta\phi ,\Delta\theta ,\Delta\psi$ 表示旋转变化， $g\in\{0,1\}$ 代表夹爪打开和关闭的状态。

![[CogACT model.png]]

## VLM

和OpenVLA一样，VLM利用Prismatic-7B。

### 视觉模块

在时间步 $t$ ，图片观察 $o_t$ 传入DINOv2和SigLIP，分别得到 $f_t^{DINO} $和 $f_t^{Sig}$ 。两个特征在channel维度concat后传入线性层，序列化成一系列的视觉感知token $\mathcal{V}=\{v_1,v_2,\dots,v_{N_v}\}$ ，默认的 $N_v=256$ 。

### 语言模块

用LLAMA-2作为骨干，用分词器把语言指令 $l$ 转化成语言token $\mathcal{T}=\{l_1,l_2,\dots,l_{N_\mathcal{T}}\}$ ，然后和视觉token $\mathcal{V}$ 以及一个额外的可学习的认知token $c$ concat到一起，之后利用模型的因果注意力进行处理。对应到认知token的输出结果特征 $f_t^c$ 编码了集成信息，决定了当前任务所需要的动作。

## 扩散动作模块

扩散动作模块接收认知特征作为输入条件，生成一系列动作。利用DiT作为动作解码过程的骨干网络。

动作模块接收认知特征 $f_t^c$ 以及一系列噪声动作 $(a_t^i,a_{t+1}^i,\dots,a_{t+N}^i)$ 作为输入，其中 $i$ 代表去噪步骤，通过多步去噪过程预测最终的动作 $(a_t,a_{t+1},\dots,a_{t+N})$ 。认知特征和去噪动作作为Transformer块的输入，其中时间步信息 $i$ 通过正弦编码后加到认知特征里。实验中 $N=15$ ，因此共有17个上下文长度。

## 训练目标

最小化预测噪音的MSE损失：

$$
\mathcal{L}_{MSE}=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,1),i}||\hat{\epsilon}^i-\epsilon||_2
$$

其中 $\hat{\epsilon}^i$ 是噪声动作序列在第 $i$ 步预测出的噪音， $\epsilon$ 是真实噪音。

## 自适应动作融合

![[CogACT model2.png]]

ACT的时序融合方法只是简单的汇总，但是可行的动作可能属于不同模式，简单的生成可能会导致动作不属于任何模式。

提出了自适应的融合策略，避免了不合理的动作融合。 $a_t|o_t$ 代表在时间步 $t$ 根据观察 $o_t$ 预测出的动作。 $\{a_t|o_{t-K},\dots,a_t|o_{t-1}\}$ 表示基于历史观察 $\{o_{t-K},\dots,o_{t-1}\}$ 对应的动作预测。推导出最终在时间步 $t$ 执行的动作 $\hat{a}_t$ 为：

$$
\hat{a}_t=\sum_{k=0}^Kw_k^{ada}\cdot a_t|o_{t-k}
$$

其中 $w_k^{ada}$ 是自适应权重标量，对于与目前预测 $a_t|o_t$ 更相似的历史预测赋予更高的权重：

$$
w_k^{ada}=\exp(\alpha\cdot<a_t|o_t,a_t|o_{t-k}>)
$$

其中 $<\cdot,\cdot>$ 计算两个动作的余弦相似度， $\alpha$ 是超参数，在实验中设置为0.1。

# 实验

## 训练集和实验细节

训练集用[[Octo]]和OpenVLA的OXE训练集，有22.5M帧。

训练batchsize为256，每个样本8个扩散步，端到端训练视觉模块、语言模块和动作模块，学习率为 $2e-5$ ，135K次迭代。利用FSDP框架在16个A100训练5天。动作模型选的是DiT-Base。融合窗口 $K$设置为与每帧距离成反比，这可以从机器人移动动作和观察频率得出。在利用Google Robot的RT-1数据集 $K$ 设置为2，利用WidowX机器人的BridgeDataV2数据集 $K$ 设置为7。

## 模拟评估

在SIMPLER评估模型。这个模拟平台通过模拟真实环境，为Google机器人和WidowX机器人等机器人提供真实的控制和视觉体验，从而弥合现实与模拟之间的控制和视觉差距。

### 评估设置

有两个评估设定：Visual Matching，通过最小化模拟环境与真实环境之间的差异，严格复制现实世界任务；Variant Aggregations，通过修改背景、照明、干扰物和桌子纹理等元素，为 Visual Matching 引入变化。对于 Google 机器人，SIMPLER 提供了两个评估设置，每个设置包含相同的四个任务：1.捡起可乐罐；2. 移动到附近；3.打开/关闭抽屉；4.打开顶部抽屉并放置苹果。对于 WidowX 机器人，SIMPLER 只提供 Visual Matching 设置，包含四个任务：1.把勺子放在毛巾上；2. 把胡萝卜放在盘子上；3. 把绿色方块堆在黄色方块上；4. 把茄子放在黄色篮子里。以成功率作为评估指标。

### Google机器人实验

![[CogACT result1.png]]

模型在两种设置中均达到了最高的平均成功率，在Visual Matching中为74.8%，在Variant Aggregations中为61.3%。

### WidowX机器人实验

![[CogACT result2.png]]

模型平均成功率达51.3%，远超其他模型，表现出色。

## 真实世界Realman机器人实验

### 机器人参数

RM-75-B，7自由度，以及一个1自由度的夹爪。

![[CogACT result3.png]]

### 任务定义

设计了三种任务：

- 捡：捡起 Object 并且放到颜色是 Color 的盘子上。其中Object ∈ {Banana, Lemon, Avocado}，并且 Color ∈ {White, Blue, Yellow}
- 堆：把颜色是 Color 的 Object 堆到颜色是 Color 的 Object 上，其中 Object ∈ {Cup, Bowl} 并且 Color ∈ {Pink, White, Blue, Yellow}
- 放：把颜色是 Color 的方块放到颜色是 Color 的方块上，其中 Color ∈ {Red, Orange, Blue, Green, Yellow}

### 微调数据

分别给3个任务收集了48、67和79个示例，并额外收集了197个不同任务的示例，例如“拿起橙色罐子，并将其放入篮子里”。总共收集了391个示例用于微调。和评估任务一样的示例很少。

### 评估方法

- 捡：每个物品8次尝试。评估每个物体的成功率，以及三个物体的平均成功率。
- 堆：每个物体24次尝试。评估每个物体的成功率，以及两个物体的平均成功率。
- 放：一共24次尝试。任务分为两步，第一步捡起方块，第二步把方块放到其他方块上。记录每一步的成功率，再计算两步的平均成功率。

在评估时随机选择颜色，放置物品以及最少3个随机位置的干扰物品。桌子高度在机器人可以够到的范围里选择多个，并且随机设置机器人的位置。

### 结果

![[CogACT result4.png]]

模型取得了显著的改进，在成功率上超过 OpenVLA 的 59.1%。

### 泛化性评估

评估了微调模型在未包含在微调数据集中的环境、干扰因素、桌子和物品上的泛化能力。将Octo-Base排除在本次评估之外，因为它在可见任务中的成功率已经接近于零。

- 没见过的桌子以及没见过的干扰物品：将桌子颜色更改为与背景和地板颜色相近，增加视觉复杂性。此外，引入三个新的没见过的物体作为干扰。

![[CogACT result5.png]]

- 没见过的颜色、形状和类别：

评估了“捡”任务中未见过的颜色表现，使用了已见过的物体（Banana, Avocado）和新的颜色（Green, Pink）的盘子。每个物体颜色组合被评估两次，共进行8次试验。

为了评估未见过的形状，定义了四个新任务：(1-2) “捡起形状是 Shape 的方块并且把它放进篮子里”，其中Shape ∈ {Triangular, Arched}; (3) “捡起一个小罐子并且把它扔到篮子里”; (4) “把一个圆柱体块堆在另一个圆柱体块之上” 每个任务进行四次评估，共进行16次试验。

为了评估未见过的类别，引入了新的评估方法：捡起 Object 并且放到篮子里，其中 Object ∈ {Eggplant, Hammer}。此任务共评估8次，每个物体4次。结果表明模型具有较强的泛化能力，尤其是在处理未见过的颜色和形状时。

![[CogACT result6.png]]

## 真实世界Franka机器人实验

### 机器人参数

Franka Arm，有7个自由度和一个1自由度的夹爪。

![[CogACT result7.png]]

### 任务定义

定义了四个任务进行评估：(1) 关上烤箱门；(2) 打开烤箱门；(3) 捡起绿色刷子；(4) 捡起装有食物的碗。

### 微调数据

针对每个任务，收集了100个演示样本，最终总共使用了400个样本来微调所有模型。

### 结果

![[CogACT result8.png]]

每个任务在11次试验中进行测试。表格展示了模型与Octo-Base和OpenVLA的对比，证明提出的方法显著提高了性能。与Realman机器人和Franka机器人的实际结果一致验证了方法在不同机器人平台上的泛化能力。

## 消融实验

在消融实验中，对Google机器人和WidowX机器人都采用SIMPLER评估。使用GR代表Google机器人，WR代表WidowX机器人，VM代表SIMPLER Visual Matching设置，VA代表SIMPLER Visual Aggregation设置。

### 动作模型架构

对比了分别有3层和7层深度的MLP，以及一系列不同规模的DiT。两个MLP的隐藏状态维度分别设为256和1024。

![[CogACT result9.png]]

表格说明：

- MLP和Trasnformer的成功率都随着模型大小变大而升高
- 相同参数量Transformer效果比MLP好
- DiT-Large平均成功率最高

### 多步动作预测

对比了预测未来不同 $N$步。结果表明15步最好。

![[CogACT result10.png]]

### 自适应动作融合

对比了ACT的动作分块和时序融合。

![[CogACT result11.png]]

平均得分更高，说明提出的自适应融合方法有效。
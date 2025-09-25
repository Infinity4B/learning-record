---
title: "Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression"
draft: false
date: 2025-09-12
tags:
  - VLA
  - 论文阅读
  - Diffusion
---

# 论文信息

论文地址：https://arxiv.org/pdf/2412.03293

项目地址：https://diffusion-vla.github.io/

代码地址：https://github.com/juruobenruo/DexVLA

ICML 2025。美的AI Lab。一样是实验非常丰富，但是没有公式没有问题建模，对于模型交代的很晦涩。此外OpenVLA本身是只支持单臂的，论文

![[DiffusionVLA overview.png]]

# 研究内容

自回归的VLA效果不错，但是有两个缺点：（1）离散动作token在一致性和精度上都很差；（2）下一token在实时动作预测性能不强。

扩散模型，例如[[Diffusion Policy]]已经在机器人领域展现出自己的能力，但是在推理能力很差，无法解决复杂任务。

因此，提出了DiffusionVLA (DiVLA) 试图结合自回归的推理能力和扩散模型的高频动作生成能力的优势，具有以下优点：

- 快速推理速度
- 增强的视觉泛化性
- 推理能力
- 对新指令和上下文能力的泛化性
- 其他本体的泛化性
- 可扩展性

# 模型

## 方法

- 图片通过SigLIP后通过ViT转为图片嵌入。对于多个图像输入，通过共享的骨干提取，然后串接到一起
- 利用Qwen2-VL作为VLM骨干，有3种大小：2B、7B和72B

### 动作解码器

利用LLM输出的token作为输入。架构采用的是Diffusion Policy的架构，随机模型权重。利用一个MLP层预测机器人关节空间。对于不同本体采用不同的MLP层。

### 推理注入模块

将最后推理部分提取出的嵌入过一个FiLM模块输入到策略模型中。

## 模型设计选择

### 视觉适应分词

随着视角变多，token数量也变多，带来很高的开销。作者认为腕部相机的处理比较低效，吃资源，因此解决方法是将腕部相机的token减少20倍，变为16。

### 训练目标

$L=L_{diff}+\alpha L_{ntp}$ ，分别代表扩散模型损失和下一token预测损失。其中 $\alpha$ 是超参数，用于平衡损失。实验中 $\alpha=1$ 。

### 预训练数据

利用Droid预训练DiVLA-2B和DiVLA-7B,利用OXE和Droid预训练DiVLA-72B。利用GPT-4o将数据转化为带推理的形式。

# 实验

用LoRA进行微调，学习率为2e-5。

有四种实验场景：分类、无序抓取、多任务学习和整理桌子。前三个在Franka上进行，最后一个在双臂AgileX上进行。数据集有500个分类的轨迹和580个多任务学习的轨迹。无序抓取定义为零样本学习。对于整理桌子收集了400个轨迹，物品随机摆放并且可能会重叠。

## 真实世界多任务学习

设计了五个任务：选择物体、翻转竖直放置的罐子、将方块放入指定盒子里、将杯子放在盘子上，以及将方块放在盒子里。

![[DiffusionVLA result1.png]]

## 真实机器人端到端分类

![[DiffusionVLA result2.png]]

分为两种难度：简单难度下少于5个物品摆放在桌子上，困难难度有5到11个随机摆放的物体。见过的和没见过的物体混合放置。在杂乱放置的场景下，物体可能会重叠。

![[DiffusionVLA result3.png]]

## 零样本没见过的物体无序抓取

有训练集中没见过的102个独特物体，大小不一。将任务指令改为`将右侧面板上的任何物体移到左侧框里`。

![[DiffusionVLA result4.png]]

## 双臂机器人适应

![[DiffusionVLA result5.png]]

所有餐具应放在左侧的板子上，并且将垃圾倒到右侧的垃圾桶中。

![[DiffusionVLA result6.png]]
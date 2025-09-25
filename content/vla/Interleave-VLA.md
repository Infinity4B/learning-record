---
title: "Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions"
draft: false
date: 2025-08-08
tags:
  - VLA
  - 论文阅读
---

# 论文信息

论文地址：https://arxiv.org/pdf/2505.02152

项目地址：https://interleave-vla.github.io/Interleave-VLA-Anonymous/

代码地址：https://github.com/Interleave-VLA/Interleave-VLA

一篇将VIMA的交织概念带到VLA的工作，核心贡献点在于提出的数据集以及对应的构建数据集的方法。

# 背景

![[Interleave-VLA overview.png]]

LLM和VLM作为基座模型能够在广泛的任务和领域中进行泛化。机器人社区也在广泛地使用机器人基座模型来将相似的泛化性引入物理具身世界。然而，许多机器人策略如今只能接收观察的图片和文字指令，不能像数字基座模型那样处理混合模态序列。举例来说，用户可能会要求机器人“拿起这个形状像这样的物体”，同时用手指着一个不规则或颜色独特的物体。用口头描述目标可能会显得笨拙或模糊不清。相比之下，交错的图像和文本指令提供了一种更直观、精确且可泛化的方法来传达这些目标。

图文交错之前在VLM中也有（Flamingo）。[[VIMA]]在机器人做过，但是重点是在界面设计。而且评测只在仿真里做，机器人操作太简单了。

VIMA是最先探索交错指令机器人控制的，提出的VIMA-Bench来学习2D物品姿态估计的视觉语言规划，但是主要集中于接口统一化而不是更广的泛化性。因此，这一范式还需要进一步探索。

核心贡献如下：

- 提出了一个自动化转换纯文本指令为图片文本交织的指令，创造来首个大规模真实世界交织的数据集，基于Open X-Embodiment拥有210k的episode和13M帧
- 提出了Interleave-VLA，一个简单可泛化，模型无关的适应性框架，首个使得VLA模型能够在最小框架改变下处理交织图片文本指令
- 在SIMPLER、VIMA-Bench和真机下展示出来稳定的领域内提升，得到2-3倍领域外泛化，大大提升了零样本环境下对多样的用户提供的视觉指令理解能力

# 模型和数据集

![[Interleave-VLA example.png]]

## Interleave-VLA

Interleave-VLA把动作分布 $P(A_t|o_t)$ 建模为基于观察 $o_t=(I_t,\mathcal{I},\mathbf{q})$。这里的 $I_t$ 是观察图片， $\mathbf{q}$ 是机器人的本体状态， $\mathcal{I}$ 是图片文本交织指令，具体可以表示为文本 $l_i$ 和图片 $\mathbf{I}_i$ 的混合序列，表示为 $\mathcal{I}=(l_1,\mathbf{I}_1,l_2,\mathbf{I}_2,\dots)$ 。目前的VLA是一种特殊情况，只使用文本描述， $\mathcal{I}=(l)$ 。

Interleave-VLA不用修改模型的核心架构，通过修改输入格式来接收交织的图片和文本token。在两种sota VLA模型[[OpenVLA]]和[[Pi-0|π0]]上进行了测试。

## 构建Open Interleaved X-Embodiment数据集

数据集可以参考[[MOO]]已有的方法。

首先用Qwen2.5提取关键物体。第二步，对于开放词目标检测，用sota的开放词检测器OWLv2基于解析出的关键词来从轨迹帧中定位和裁剪目标物体。最后，引入来数据质量检测来针对OWLv2失败的情况。用Qwen2.5-VL来验证检测出的物品，如果需要的话再提供关键点来让Segment Anything进行更精确的分割。

应用数据集生成流程处理了来自Open X-Embodiment 的11个数据集：RT-1, Berkeley Autolab UR5, IAMLab CMU Pickup Insert, Stanford Hydra, UTAustin Sirius, Bridge , Jaco Play  UCSD Kitchen , BC-Z , Langugae Table 和UTAustin Mutex。精选数据集包含 21 万集和 1300 万帧，涵盖 3500 个独特物体和多种任务类型。

![[Interleave-VLA model.png]]

# 实验

## 任务

利用SIMPLER和VIMA-BENCH进行评估。在配备了SMC夹爪的FANUC LRMate 200iD/7L机械臂上进行了实机的机器人实验。

## 实验结果

SIMPLER

![[Interleave-VLA result1.png]]

VIMA-Bench

![[Interleave-VLA result2.png]]

真机

![[Interleave-VLA result3.png]]
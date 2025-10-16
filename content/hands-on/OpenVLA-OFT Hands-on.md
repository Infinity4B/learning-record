---
title: OpenVLA-OFT是如何将VLM改造成VLA的
draft: true
tags:
  - VLA
  - Hands-on
date: 2025-10-16
---
# 前言

在[[OpenVLA Hands-on]]中已经介绍了OpenVLA的改造。[[OpenVLA-OFT]]我们也介绍过，是在[[OpenVLA]]后对于VLA范式的探索。OpenVLA-OFT仓库是OpenVLA仓库的一个fork，在本文中研究OpenVLA-OFT对于VLM的改造。

# 环境配置

原始仓库地址： https://github.com/moojink/openvla-oft

相比于OpenVLA每个LIBERO任务都有一个模型，这次提供了一个all-in-one的模型 https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 。

首先根据`SETUP.md`安装对应的环境，再根据`LIBERO.md`安装对应的LIBERO环境。之后运行

```bash
pip install numpy==1.26.0
pip install opencv-python==4.6.0.66
```

就可以把环境配置完成了。具体评测运行同样可以参考`LIBERO.md`中的指令。


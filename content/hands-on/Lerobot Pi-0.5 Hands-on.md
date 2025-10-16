---
title: Lerobot Pi-0.5 代码分析
draft: false
tags:
  - VLA
  - Diffusion
  - Hands-on
date: 2025-10-16
---
# 前言

> [!INFO]
> Pi-0.5目前官方开源的实现和原论文有所不同。
> 
> 在阅读本文之前，强烈建议先阅读[[Lerobot Pi-0 Hands-on]]。

参考 https://github.com/Physical-Intelligence/openpi/issues/647 。官方给的回应是：

> Jup that's correct -- we currently only support action decoding in openpi, but as [@Qu3tzal](https://github.com/Qu3tzal) mentioned this should already give you a capable policy!

![[file-20251016162633324.png]]

开源的 $\pi_{0.5}$ 代码没有加入论文中提到的low-level command机制。对比[[Pi-0|π0]]和 [[Pi-0.5|π0.5]]代码，改动很小。具体在以下几个方面：

1. 删掉了本体状态state和对应的MLP，直接没使用
    
2. action_time_emb变为直接使用action_emb作为suffix_emb，time_emb作为adarms_cond条件

其余代码和 $\pi_0$ 代码几乎一模一样。

概览图（可点击↔︎放大，往下拖）：

```mermaid
graph TD

  

    B1["obs.image (1,3,256,256)"] -->|resize+norm| B2["(1,3,224,224)"]

    B3["obs.image2 (1,3,256,256)"] -->|resize+norm| B4["(1,3,224,224)"]

    B5["empty camera"] -->|dummy| B6["(1,3,224,224)"]

  

    L1["language.tokens (1,48)"] --> LANG["language input"]

    L2["language.att_mask (1,48)"] --> LANG

    T1["task string"] -.-> L1

  

    B2 --> IMG["PaliGemma img encoder"]

    B4 --> IMG

    B6 --> IMG

    IMG --> IEMB["3 x img_emb (1,256,2048)"]

  

    LANG --> LEMB["lang_emb (1,48,2048)"]

  

    IEMB --> CATpre

    LEMB --> CATpre

    CATpre["concat img+lang"] --> PREFIX["prefix_embs (1,816,2048)"]

  

    PREFIX -->|forward Paligemma| PKV["past_key_values"]

  

    PKV --> LOOP

    N1["init noise (1,50,32)"] --> LOOP

  

    subgraph LOOP["Diffusion Loop"]

        direction TB

        XT["x_t (1,50,32)"]

        TSTEP["timestep (1,)"]

  

        XT --> DENOISE

        TSTEP --> DENOISE

        PKV2["past_key_values"] --> DENOISE

  

        subgraph DENOISE["denoise_step"]

            direction TB

  

            XT2["x_t (1,50,32)"] -->|action_in_proj| ACT["action_emb (1,50,1024)"]

  

            Tm["timestep (1,)"] -->|sinusoidal| Temb["time_emb (1,1024)"]

            Temb -->|time_mlp_in/out + silu| Tcond["time_emb_cond (1,1024)"]-->EXPERT

  

            ACT --> SUF["suffix_embs = action_emb (1,50,1024)"]

  

            SUF --> EXPERT["Paligemma expert<br/>(adarms_cond = time_emb_cond)"]

            EXPERT --> OUTemb["(1,50,1024)"] -->|action_out_proj| Vt["v_t (1,50,32)"]

        end

  

        DENOISE --> UPDATE["x_{t-1} = x_t + dt * v_t"]

        UPDATE --> XT

    end

  

    LOOP --> FINAL["final actions (1,50,32)"]
```

# 评测结果

| **任务**        | **LIBERO-10** | **LIBERO-goal** | **LIBERO-object** | **LIBERO-spatial** |
| ------------- | ------------- | --------------- | ----------------- | ------------------ |
| **Pi-0.5复现**  | 97.0%         | 97.6%           | 99.4%             | 97.6%              |

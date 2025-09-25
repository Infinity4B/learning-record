---
title: A Survey on Diffusion Language Models
draft: false
date: 2025-09-05
tags:
  - Diffusion
  - 论文阅读
---
# 论文信息

论文地址：http://arxiv.org/pdf/2508.10875

# 背景

自回归语言模型在NLP中拥有统治性地位，通过因果注意力和teacher forcing训练下一个token。自回归模型在不同任务上都表现出了很好的效果，但是推理速度很慢，影响了并行性，限制了计算效率和吞吐量。

扩散模型擅长建模复杂数据分布，在图片和视频生成方面达到了sota。成功的模型有Stable Diffusion、Imagen和Sora，证明了扩散范式的可扩展性和泛化性。扩散模型一次迭代可以生成多个token或者一整个序列，带来更好的推理吞吐。

扩散语言模型需要解决的问题是如何建模离散数据。扩散模型主要是在连续的领域，例如图像中较为成功。连续扩散语言模型把token映射到嵌入空间，在这个连续空间中进行去噪。离散扩散语言模型直接在token空间中进行扩散。

![[file-20250925172843447.png]]
对比自回归模型，扩散模型有以下优点：

- 并行生成：一次迭代去噪过程可以生成多个token
- 双向上下文：利用双向上下文，能带来更丰富的上下文嵌入
- 迭代改进：可以接收高置信度的token，掩码低置信度的token，生成高质量文本
- 可控性：可以基于特定token位置和结构生成，非常适合填充和结构生成任务
- 模态间统一建模：用一个共享的去噪建模框架，可以更自然地支持文本和图像生成任务

## 扩散语言模型范式

### 常用的语言建模方法

1. 掩码语言模型

Masked Language Models (MLMs) 最著名的是BERT，利用的是基于Transformer的仅编码器架构。MLM通过随机掩码token学习双向上下文表示。这种方法遵循一种去噪自动编码框架，输入的某些部分会被掩码，模型会经过训练来重建这些部分：

$$

\mathcal{L}_{\text{MLM}} = \mathbb{E}_{x \sim D} \mathbb{E}_{\mathcal{M} \sim \text{Mask}(x)} \left[ -\sum_{i \in \mathcal{M}} \log P_{\theta}(x_i \mid x_{\setminus \mathcal{M}}) \right] 

$$

其中 $x$ 代表输入序列， $\mathcal{M}$ 代表掩码掉的位置， $x_{\setminus \mathcal{M}}$ 代表没有被掩码的上下文。原始的BERT还引入了下一句子预测 (NSP)目标来建模句子间关系：

$$ 

\mathcal{L}_{\text{NSP}} = \mathbb{E}_{(A,B,y) \sim D} \left[ -\log P_{\theta}(y \mid A, B) \right] 

$$

其中 $(A,B)$ 是一对文本片段， $y\in\{0,1\}$ 代表了 $B$ 在原文中是否在 $A$ 的后面。

后续有RoBERTa、ALBERT、DeBERTa进行了改进。但是MLM主要用于理解任务，不是用于生成任务，因此不修改框架无法进行开放式生成任务。

1. 自回归语言模型

受GPT、Transformer-XL和后续的模型影响，自回归语言模型成为了现代生成AI的骨干。自回归语言模型使用单向自左向右的token生成方式。自回归语言模型把文本序列的联合分布分解为条件概率的乘积：

$$ 

P(x)=\prod_{i=1}^{n} P_{\theta}\left(x_{i} \mid x_{1}, x_{2}, \ldots, x_{i-1}\right) 

$$

给定一个token序列 $X=(x_1,x_2,\dots,x_n)$ ，训练目标的最大化这一分解下序列的对数似然估计：

$$ 

\mathcal{L}_{\text{AR}}=\mathbb{E}_{X \sim \mathcal{D}}\left[-\sum_{i=1}^{n} \log P_{\theta}\left(x_{i} \mid x_{1}, \ldots, x_{i-1}\right)\right] 

$$

通常是用因果注意力掩码和teacher forcing训练的的仅解码器的Transformer架构实现。

1. 其他范式

- Seq2Seq

Sequence-to-sequence模型基于编码器-解码器架构，适合条件文本生成任务，如机器翻译和摘要任务等。T5和BART是较为出名的两个模型。

在Seq2seq架构中，编码器把原文本编码成中间表示，用于后续解码器生成目标文本。原始的Seq2seq是自回归方式生成的，但是框架很灵活。DiffuSeq和SeqDiffuSeq把自回归解码器换成了非自回归扩散解码器，利用了编码器的强大的条件能力来引导生成过程中的去噪。

- 排列语言模型

Permutation Language Models (PLMs)，著名的有XLNet，提供了一种新的方法引入双向注意力到生成框架中。PLM也是训练预测token序列，和其他模型的不同的是，序列顺序是随机排列的顺序而不是自左向右的顺序。目标是最大化分解顺序的所有可能排列方式的期望对数似然估计：

$$

\mathcal{L}_{\text{PLM}} = \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[ -\sum_{t=1}^{N} \log P_{\theta}(x_{z_t} \mid \mathbf{x}_{z_{<t}}) \right] 

$$

其中 $\mathcal{Z}_T$ 代表长度为 $T$ 的序列所有可能的组合， ${z_t}$ 和${z_{<t}}$ 代表给定组合 $\mathbf{z}\in\mathcal{Z}_T$ 的第 $t$ 个和前 $t-1$ 个元素。这样的方式同时结合了双向语言模型和自回归生成的优势。

### 连续扩散语言模型

连续空间扩散语言模型首先映射离散token到线序语言空间。扩散过程建模连续空间的数据分布。扩散过程分为前向过程和反向过程。其中前向过程也就是加噪过程，通过一个固定的马尔可夫链将数据样本 $\mathbf{x}_0$通过 $T$时间步转化为噪音：

$$ 

q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) 

$$

$$ 

q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \mu_t(\mathbf{x}_{t-1}), \Sigma_t) 

$$

其中 $\mu_t$ 和 $\Sigma_t$ 代表了噪音调度。在许多实现例如DDPM和Rectified Flow中，每个时间步的边缘分布有一个闭式解表达形式：

$$ 

\mathbf{x}_t = \alpha_t \mathbf{x}_0 + b_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) 

$$

其中 $\alpha_t$ 和 $b_t$ 是时间 $t$ 有关的确定性函数。

反向过程学习逆向这一破坏，从噪音 $\mathbf{x}_T\in\mathcal{N}(0,\mathbf{I})$ 开始逐渐去噪来恢复一个接近 $\mathbf{x}_0$ 的样本。通常是通过Transformer架构的参数为 $f_\theta(\mathbf{x}_t,t)$ 的神经网络实现，预测与前向过程有关的数值 $\mathbf{z}$（可能是干净数据、噪音或者速度）。训练目标为：

$$ 

\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \mathbf{z}} \left[ \| f_{\theta}(\mathbf{x}_t, t) - \mathbf{z} \|^2 \right] 

$$

其中 $\mathbf{x}_t$ 是给定 $\mathbf{x}_0$ 采样出的， $\mathbf{z}$ 是从 $\mathbf{x}_0$ 和 $t$ 推导出的对应的回归目标。

在训练后，给定噪音 $\mathbf{x}_T\in\mathcal{N}(0,\mathbf{I})$ ，在每个时间步 $t=T,T-1,\dots,1$ 模型定义一个条件分布 $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ ，目标是估计真实的逆向转换 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 。从这些学习到的条件下迭代采样，生成噪声越来越少的潜在状态，直到恢复原始数据 $\mathbf{x}_0$ 的估计值。生成去噪后的嵌入 $\hat{\mathbf{x}}_0$ 后，通过一个rounding的步骤将其映射回离散空间，通常是通过嵌入空间中的最近邻搜索或使用解码器头来完成。

连续扩散过程也可以在数值空间中实现，TESS利用一个k数值的单纯形的token表示实现。之后的TESS2对此进行拓展，将预训练自回归模型适应到扩散语言模型上，提高了指令跟随能力。

### 离散扩散语言模型

离散空间扩散语言模型直接在token词汇表上定义扩散过程，避免在扩散过程中需要一个连续的嵌入空间。D3PM首先通过在离散token中引入一个结构化的扩散过程说明了这一点。前向过程通过一个转移矩阵 $\mathbf{Q}_t$ 破坏一个序列。转移矩阵定义了一个token转换到另一个词汇表里的token的概率。给定初始状态 $\mathbf{x}_0$ ，状态 $\mathbf{x}_t$ 的概率可以通过一个分类分布定义：

$$ 

q(\mathbf{x}_t|\mathbf{x}_0) = \text{Cat}(\mathbf{x}_t; \mathbf{p} = \mathbf{x}_0 \bar{\mathbf{Q}}_t), \text{ where } \bar{\mathbf{Q}}_t = \prod_{i=1}^{t} \mathbf{Q}_i 

$$

一个常见的 $\mathbf{Q}_t$ 选择是一个吸收状态转移，每个token有概率不变或者转移到`[MASK]` token。反向过程学习逆向这些转换，根据被破坏的序列预测原token的概率分布。

随着时间推移，掩码扩散语言模型已成为离散扩散语言模型的代表]。以这类模型中最具代表性的LLaDA模型为例。受先前关于重参数化和简化训练目标的研究启发，LLaDA采用仅在掩码标记上计算的交叉熵损失进行从头训练：

$$ 

\mathcal{L}(\theta) \triangleq -\mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \text{M}] \log p_{\theta}(x_0^i | x_t) \right] 

$$

其中 $x_0$ 是从训练语料库中采样的， $t$ 均匀从 $[0,1]$ 区间采样， $x_t$ 是通过前向过程破坏 $x_0$ 得到的。 $\mathbf{1}[\cdot]$ 保证损失只在被掩码的位置计算。在推理时从一个期望长度的全被掩码的序列开始，在每个迭代步中，模型接收当前的序列，预测一个完整的token序列。根据模型的预测置信度和噪声调度，一定数量的高置信度预测被取消掩码并固定，剩余位置重新掩码。这种优化过程迭代进行，直到所有 `[MASK]` token被解析。这种方法结合了MLM的双向上下文和可控的并行生成过程。LLaDA-8B 表现出强大的可扩展性和指令跟随能力，性能与 LLaMA3-8B 等强大的自回归模型相当。这挑战了自回归模型在大型语言生成中的长期主导地位。

### 混合自回归-扩散语言模型

混合自回归-扩散语言模型试图在全并行化非自回归模型和强因果依赖建模的自回归模型中取得平衡。一种流行的混合AR-扩散建模方法采用块级半自回归生成过程。在这种设置中，模型自回归地生成一块一块的token，而每个块内的token则通过类似扩散的迭代过程并行生成。

生成过程通常由两个嵌套的循环组成。在外部循环中，自回归地生成一块一块的token，每个块的生成依赖于之前生成的块。在每个块内，内部循环通过扩散式迭代去噪过程并行生成token。在BD3-LM中，训练目标定义为：

$$ 

\mathcal{L}_{\text{BD}}(\mathbf{x}, \theta) := -\sum_{b=1}^{B} \mathbb{E}_{t \sim [0, 1]} \mathbb{E}_q \frac{1}{t} \log p_{\theta}(\mathbf{x}^b | \mathbf{x}_t^b, \mathbf{x}^{<b}) 

$$

这种混合策略通过自回归在多个块之间捕捉长期依赖，同时通过并行扩散加速每个块的生成。这样的设计还支持灵活的输出长度和KV-Cache。
![[file-20250925192423490.png]]
# 预训练和后训练方法

这个人很懒，还没有填充内容。

# 推理策略

这个人很懒，还没有填充内容。

# 多模态和统一方法

这个人很懒，还没有填充内容。

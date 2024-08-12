
# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- Paper Link: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
- Author's code: https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py
- https://mp.weixin.qq.com/s/Z8BVt7g6rnuAFzoca1fjfg


# 1- introduce

这个模型通过对输入token进行梯度下降来压缩上下文，这种方法被称为「测试时间训练层（Test-Time-Training layers，TTT）」。
TTT层直接替代了注意力机制，解锁了具有表现力记忆的线性复杂度架构，使我们能够在上下文中训练包含数百万（未来可能是数十亿）个token的LLM。
作者相信，这个研究了一年多的项目，将从根本上改变我们的语言模型方法。

序列越长后面的预测困惑度就会越低，transformer 在32K之后就有明显的表现，Mamba在16K之后有明显的表现。On the Other hand, once context is long enough, existing RNNs such as Mamba struggle to actually take advantage of the extra information being conditioned on.

## TTT-layers
> In this paper, we begin with the observation that self-supervised learning can compress a massive training set into the weights of a model such as LLM, which often exhibits deep understanding about the semantic connections among its training data - exactly what we need from a compression heuristic.

new class of sequence modeling layres where <font color=darkred>hidden state is a model</font> , and update rule is a step of self-supervised learning. 
Because the process of updating hidden state on a test sequence is equivalent to training a model at test time, this new class of layers is called Test-Time Traing(TTT) layer. 

- TTT-Linear: hidden state is a linear model
- TTT-MLP: hidden state is two-layer MLP

## 主要贡献：
1. propose TTT layers. Our perspective that the forward pass of a layer contains a training loop itself opens up a new direction for future reasearch.
2. TTT-Linear, one simple instantiation of TTT layers, outperforms Transformers and Mamba in our evaluations ranging from 125M to 1.3B parameters.
3. We improve the hardware efficiency of TTT layers through mini-batch TTT and the dual form, making TTT-Linear already a practical building block for LLMs

# 2- Method 

所有的序列建模层，都可以从将历史上下文存储到隐藏状态的角度来看待。比如RNN层——如LSTM、RWKV和Mamba层——<font color=darkred>将上下文压缩成一个固定大小的状态，这个状态随时间变化。</font>
这种压缩带来了两种结果：
- 优势是处理效率高，因为每个Token的处理时间是恒定的
- 劣势是在处理上下文是，RNN性能受限于隐藏状态的 表达能力

self-attention也可从上述两个角度去理解, 不同在于隐藏状态为K V缓存，是一个随着t增长的线下list


|Method|Initial state|Update rule| Output rule | Cost|
|-|-|-|-|-|
|Naive RNN| $s_0=\text{vector()}$|$s_t=\sigma (\theta_{ss}s_{t-1} + \theta_{sx}x_t)$|$z_t=\theta_{zs}s_t+\theta_{zx}x_t$|O(1)|
|Self-attention|$s_0=list()$|$s_t=s_{t-1}.append(k_t,v_t)$|$z_t=V_tsoftmax(K^T_t q_t)$|O(t)|
|Naive TTT|$W_0=f.params()$|$W_t=W_{t-1}-\eta \nabla \mathcal{l}(W_{t-1};x_t)$|$z_t=f(x_t;W_t)$|O(1)|

因此，为了在长上下文中既保持效率，又具有表达能力，需要一个更好的 压缩启发式 (`compression heuristic`)方法。具体来说，就需要将数百万个token压缩成一个<b>能有效捕捉其底层结构和关系的隐藏状态。</b>


## 2.1 TTT as updating hidden state

使用自监督学习来将历史上下文$x_1,...,x_t$压缩成一个隐藏状态$s_t$。本文方法是将上下文视为一个无标签数据集，而将状态视为一个模型。

具体来说，隐藏状态$s_t$现在等同于一个模型f的权重$W_t$，这个模型f可以是线性模型、小型神经网络或其他任何形式。输出规则简单表示为:
$$z_t=f(x_t;W_t) ...... (1)$$

直观来说，输出token就是由更新后权重$W_t$的模型f对$x_t$所做的预测。更新规则是在某个自监督损失$l$上进行的一步梯度下降:
$$W_t=W_{t-1} - \eta \nabla \mathcal{l}(W_{t-1};x_t)  ...... (2)$$

从压缩的角度来看，每种启发式方法需要决定记住/忘记那些输入。W会记住那些产生大梯度的输入——直观说，就是那些使W学习很多的输入。

l的一种选择是重构$x_t$本身。
$$l(W;x_t)=||f(\hat{x_t};W)-x_t||^2  ...... (3)$$


Similar to denoising autoencoders, f需要发现$x_t$ 各个维度之间的相关性，以便从部分信息
$\hat{x_t}$ 中重构出 $x_t$， 梯度下降可以减少l，但无法将其降低为零。


## 2.2 Training a network with TTT layers

TTT-layers 和 RNN & self-attention 有一样的API接口


- Learner is not a subclass of nn.Module, state.model is updated manually in the inner loop for each call of state.train.

- outer loop: 训练更大的神经网络
- inner loop: training W within each TTT layer as the inner loop

## 2.3 Learning a self-supervised task for TTT

可以说，TTT最重要的部分是自监督任务，因为它决定了W从测试序列中学习的特征类型。
在这个任务的设计上，研究人员采取了更加端到端的方法——直接优化自监督任务以实现下一个token预测的最终目标。具体来说，研究着将自监督任务的学习，作为外循环的一部分。
将公式3细化：

$$l(W;x_t) = || f(\theta_K x_t; W)  - \theta_V x_t||^2 ....... (4)$$

- traing_view: $\hat{x_t} = \theta_K x_t$; $\theta_K$是可学习的
- label_view: $\theta_V x_t$ instead of $x_t$
  - not all the information in $x_t$ is worth remembering

1. 在内循环中，只有W被优化，因此作为l的参数写出；$\theta$是这个损失函数的超参数。
2. 在外循环中，$\theta_K, \theta_V, \theta_Q$与$\theta_{rest}$一起被优化，而W仅仅是一个隐藏状态，不是参数。

traing view  $\hat{x_t} = \theta_K x_t$ 的维度会比 $x_t$少，公式(1)优化为
$$z_t=f(\theta_Q x_t; W_t) ....... (5)$$

总的来说$\theta_K, \theta_V, \theta_Q$所有可能的选择构成了一系列多视图重构任务，外循环可以被理解为从这个任务组中选择一个具体任务。为了简单起见，研究人员在这里将所有视图设计为线性投影。


## 2.4 Parallelization with mini-batch TTT

$G_t=\nabla l(W_{t^\prime};x_t)$其中$t^\prime = t - mod(t, b)$代表着前一个mini-batch的最后一个时间步,因此可以一次并行b个梯度计算。
- $b=16$ in paper

## 2.5 Dual form 

$G_t=\frac{\partial{loss=MSE(W_0 x_t, x_t)}}{\partial{W_0}}=\nabla l(W_{0};x_t)=2(W_0 x_t - x_t)x_t^T$ 

就可以用上面简化的TTT-Linear情况来演示这些计算，表示X = [x1, . . . , xb]：

$$W_b=W_0-\eta\sum_{t=1}^b G_t=W_0 - 2\eta \sum_{t=1}^b(W_0x_t - x_t)x^T=W_0-2 \eta (W_o X- X)X^T$$

$W_b$可以方便的计算出来
$$z_t=f(x_t;W_t)=W_t x_t=(W_0 - \eta \sum_{s=1}^t G_s)x_t=W_0x_t - 2\eta \sum_{t=1}^b(W_0x_s - x_s)x_s^Tx_s$$
$\delta=\sum_{s=1}^t(W_0x_s - x_s)x_s^T x_s$, $\Delta=[\delta_1, \delta_2, ..., \delta_b]$

$$\Delta = mask(X^TX)(W_0 X-X)$$
$$Z = W_0 -2\eta \Delta$$


## 2.6 Theoretical equivalences


1. TTT-Linear:
   1. $f_{lin}(x)=Wx; W_0=0$ 
   2. $W_t=W_{t-1}-\eta \nabla l(W_0;x)=W_0-\eta\sum_{t=1}^b G_t=2\eta \theta_V X(\theta_K X)^T$ 
      1. 在2.3中公式(4)，f的入参是target_view: $\theta_K X$, $label=\theta_V X$
      2. 所以$\frac{\partial{loss(W_0;x_t)}}{\partial W_0}=2(W_0\theta_K X -\theta_V X)(\theta_K X)^T=-2\theta_V X(\theta_K X)^T$
   3. $z_t=f(\theta_Q x; W_t)=\sum_{s=1}^t\theta_V x(\theta_K x)^T\theta_Q x_t$
2. TTT-MLP
   1. $f_{MLP}$有两层，类似于Tansformer的MLP
  
具体来说，隐藏维度是4×输入维度，然后是GELU激活。为了在TTT期间获得更好的稳定性，f始终包含层归一化 (LN) 和残差连接。

即，$f(x)=x+LN(f_{res}(x))$，其中，$f_{res}$可以是$f_{lin}$或$f_{MLP}$。

# 3- Experiments 
通过与两个基线Transformer和Mamba（现代RNN）比较，研究人员评估了TTT-Linear和TTT-MLP。

- 数据集
  - 继续Mamba论文之后，研究人员在Pile上执行了2k和8k上下文长度的标准实验，Pile是一个用于训练开源LLM的流行文档数据集。

- 主架构
  - Transformer和Mamba使用不同的，除非另有说明，TTT-Linear和TTT-MLP始终使用Mamba架构。

## 比较

### 短上下文：the Pile

在2k上下文中，TTT-Linear（M）、Mamba和Transformer具有相当的性能，线条大部分重叠。

TTT-MLP（M）在较大的FLOP预算下表现稍差。尽管TTT-MLP在每个模型大小上，都比TTT-Linear具有更好的复杂度，但FLOP的额外成本抵消了这种优势。

在8k上下文中，TTT-Linear（M）和TTT-MLP（M）的表现均明显优于Mamba。即使是具有Transformer架构的TTT-MLP（T），性能也比Mamba略好。

另外，研究人员还观察到了一个非常明显的现象：随着上下文长度变长，TTT层相对于Mamba的优势就更大了。


### 长上下文：Books

为了评估长上下文中的功能，研究人员使用了Pile的一个流行子集——Books，对从1k到32k以2个增量的上下文长度进行了实验。

根据上图，可以观察到——
在Books的2k上下文中，Pile 2k的所有观察结果仍然成立，唯一的例外是Mamba的表现略好于TTT-Linear


在32k上下文中，TTT-Linear（M）和TTT-MLP（M）的性能均优于Mamba，与Pile 8k的观察结果类似。即使具有Transformer架构的TTT-MLP（T），在32k上下文中的表现也比Mamba稍好。

在1.3B尺度上，TTT-MLP（T）仅比TTT-MLP（M）稍差。由于缺之清晰的线性拟合，很难推导出经验缩放定律。然而，TTT-MLP（T）的强劲趋势表明，Transformer架构可能更适合超出评估的更大模型和更长上下文。


### 上下文长度作为超参数

虽然输入序列的长度由用户确定，但语言模型处理输入的上下文长度可以由工程师确定。因此，上下文长度也是一个可以选择的超参数。
对于具有线性复杂度的LLM，研究人员选择了困惑度中的argmin，因为每个上下文长度都有相同的FLOP。

从图13中，可以观察到以下结果——

- 性能最好的方法TTT-Linear和TTT-MLP的线几乎完全重叠。Mamba和TF Finetune的线在$10^{20}$ FLOP后也大部分重叠。

- TF Finetune的性能明显优于TF Pretrain，因为它受益于长上下文，而不会在训练FLOP中产生极大的成本。

- 对于所有从头开始训练的方法（包括TF预训练），一旦上下文长度变得太大，困惑度就会变得更糟。

从上图可见，与TTT-Linear相比，TTT-MLP在短上下文中表现稍差，但在长上下文中表现更好。

这一观察结果正符合研究人员的预期，即作为隐藏状态的MLP比线性模型更具表现力。同样，所有方法都具有与Mamba 1.4B相同的训练FLOP。


### 实际运行时间

LLM训练和推理可以分解为前向、后向和生成。

由于前向（在训练和推理期间）和后向都可以并行化，因此研究人员使用对偶形式。生成新token（也称为解码）本质上是顺序的，因此研究人员使用原始形式。

由于资源限制，这项实验是用JAX编写并在TPU上运行的。

然而，由于Mamba（在PyTorch、Triton和CUDA中实现）只能在GPU上运行，因此为了公平比较，研究人员还重写了方法，以在GPU上运行。

具体来说，研究人员在ThunderKittens中编写了一个用于前向的GPU内核。从历史上看，由于并行性和矩阵相乘的使用不当，RNN在前向和后向过程中效率低下。

这个前向内核的目标，是证明mini-batch TTT和这些问题对偶形式的有效性。

图15的左图显示了前向内核批大小为16的延迟。所有模型参数均为1.3B（Mamba为 1.4B）。

对于Transformer，每个token的时间随着上下文长度的增加而线性增长，但对于其他方法则大致保持不变。

此外，研究人员在Triton中编写了另一个用于生成的GPU内核，并在图15的右图中对批大小为512的速度进行了基准测试。

可以看出，TTT-Linear和Mamba的延迟几乎相同，明显小于Transformer和TTT-MLP。

Mamba之后，又看到TTT这么能打的新架构诞生，少不了AI社区的热议。



# discussion

有网友称，这会不会是最接近实时上下文的方法？很想听听大家的想法。这意味着TTT甚至在使用过程中，也能够学习和适应，为长上下文提供更好的性能，而不会产生通常与Transformer相关的高昂计算成本。

如果scaling law依然存在，TTT将带来难以置信的影响。对于长序列，Transformer的计算成本往往很高，当长序列变得更长时，RNN会遗忘。TTT训练巧妙地利用神经网络解决RNN的不足。


# 作者介绍

其中的核心作者是，Yu Sun、Xinhao Li和Karan Dalal。

## Yu Sun
Yu Sun是斯坦福大学计算机专业的博士后，导师是Carlos Guestrin、Tatsu Hashimoto和Sanmi Koyejo。

此前，他曾在加州大学伯克利分校完成了电子工程科学博士学位，导师是Alyosha Efros和Moritz Hardt。他还在康奈尔大学拿到了学士学位。

个人主页中，他介绍自己的研究重点是一种名为测试时间训练（test-time training）的算法框架。其核心思想是，每个测试实例都定义了自己的学习问题，都有自己的泛化目标。这通常使用自监督学习，为每个实例即时训练一个不同的模型来实现的。

在最新研究中，Yu Sun与Xinhao Li在2022年11月共同启动了这一项目。自2023年6月起，Yu Sun专职负责该项目。

他提出了项目的概念框架，设计了mini-batch TTT和对偶形式（dual form）。

## Xinhao Li

Xinhao Li是UC San Diego研二的学生，导师是Xiaolong Wang教授。他本人的研究兴趣主要是深度学习和计算机视觉。

他在斯坦福大学Tatsunori Hashimoto教授的团队中作为访问学生，与Yu Sun博士和其他导师朋友一起工作。在此之前，他曾在电子科技大学获得了学士学位。

在2024年3月之前，Xinhao Li是TTT早期代码库的主要贡献者，这些代码库塑造了最新项目。


## Karan Dalal

Karan Dalal是UC Berkeley电子工程科学系的本科生。他于2023年6月全职加入该项目，与Xinhao Li合作共同领导了当前代码库的开发工作。





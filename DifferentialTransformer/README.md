# Differential Transformer

- Paper Link: [arxiv 2410](https://arxiv.org/pdf/2410.05258)
- GithubCode: [https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py](https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py)

> 注意力机制的改进，不仅显著提升了检索精度，还能缓解LLM的幻觉
## 1- introduce

Transformer Problem 
- attention noise: Transformer tends to allocate only a small proportion of attention scores to the correct answer, while disproportionately focusing on irrelevant context 
  - 我们对于LLM检索、利用长上下文的过程，知之甚少，其注意力过程也需要更多的改进。

Diff Transformer
- cancel attention noise with differential denoising.
- Q, K split to 2 groups and compute two seperare softmax attention maps
  - $attn = softmax(QK) - \lambda softmax(QK)$
- ideal come from
  - The approach is analogous to noise-canceling headphones and differential amplifiers in electrical engineering, where the difference between two signals cancels out common-mode noise

![figure-1]()


## 2- Model Arch: Differential Transformer

### Diff Atten

$[Q_1, Q_2] = XW^Q, [K_1, K_2] = XW^K, V = XW^V$
$$\text{DiffAttn}(X) = [softmax(\frac{Q_1K_1^T}{\sqrt{d}}) - \lambda softmax(\frac{Q_2K_2^T}{\sqrt{d}})]V$$
- $W^Q, W^K, W^V \in \mathbb{R}^{d_\text{model}\times 2d}$
- $\lambda = e^{\lambda_{q_1}\lambda_{k_1}} - e^{\lambda_{q_2}\lambda_{k_2}} + \lambda_{init}$
    - WHY $\lambda_{q_1}, \lambda_{k_1}, \lambda_{q_2}, \lambda_{k_2} \in \mathbb{R}^{d}$
    - Bias: $\lambda_{init} \in (0, 1)$ magic init: $\lambda_{init} = 0.8 - 0.6 \times e^{-0.3(l - 1)}; l \in [1, L]$



### Multi-Head Diff Atten

$\text{head}_i = \text{DiffAttn}(X; W_i^Q, W_i^K, W_i^V, \lambda) $
$\hat{\text{head}_i} = (1 - \lambda_{init})RMSNorm(\text{head}_i) $
$\text{MultiHead(X)}=Concat(\hat{\text{head}_1},...,\hat{\text{head}_h})W^O$
- $W^O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$
- $h = d_{model}/2d$

### Headwise Normalization
> paper44: Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pp. 3–19, 2018.

As differential attention tends to have a sparser pattern, statistical information is more diverse between heads. The LN() operator normalizes each head before concatenation to improve gradient statistics.

### Overall Architecture

![Figure2]()

$$Y^l=\text{MultiHead}(LN(X^l)) + X^l$$
$$X^{l+1}=SwiGLU(LN(Y^l)) + Y^l$$

- $LN() -> RMSNorm$

fake code
```python
def DiffAttn(X, W_q, W_k, W_v, lmd):
    Q1, Q2 = split(X @ W_q)
    K1, K2 = split(X @ W_k)
    V = X @ W_v
    # Qi, Ki: [b,n,d]； V：[b, n, 2d]
    s = 1 / sqrt(d)
    A1 = Q1 @ K1.transpose(−1, −2) ∗ s
    A2 = Q2 @ K2.transpose(−1, −2) ∗ s
    return (softmax(A1) - lmd * softmax(A2)) @ V


def MultiHead(X, W_q, W_k, W_v, W_o, lmd, h):
    O = GroupNorm([
        DiffAttn(X, W_qi, W_ki, W_vi, lmd) for i in range(h)
    ])
    O = O * (1 - lmd_init)
    return Concat(o) @ W_o

```


## 3- Experiments

1. 下游任务
   1. 与之前经过精心调优的Transformer语言模型相比，DIFF Transformer取得了良好的性能。
   2. 尤其是对于长上下文任务，如图4所示，随着上下文长度不断增加，累计平均的负对数似然值（NLL）持续降低，说明Diff Transformer可以更有效地利用不断增加的上下文。
2. 关键信息检索: 「大海捞针」（Needle-In-A-Haystack）测试被广泛用于评估LLM提取长上下文中的关键信息的能力。
   1. DIFF Transformer在提升检索精度的同时也缓解了幻觉现象
   2. 可以发现，与Transformer相比，DIFF Transformer的上下文幻觉明显减轻。这种性能的提高可能源于，改进后的注意力模块能更好第关注任务所需信息，而非不相关的上下文。
3. 缩放特性
   1. 扩展模型规模: 
      1. 68亿参数规模的DIFF Transformer达到了与110亿参数规模Transformer相当的验证损失，但仅需62.2%的参数。
      2. 78亿参数的DIFF Transformer匹配了131亿参数的Transformer的性能，参数量是后者的59.5%。
   2. 扩展训练Token
      1. 使用160B token训练的DIFF Transformer达到了与使用251B token训练的Transformer相当的性能，但仅消耗了63.7%的训练数据。


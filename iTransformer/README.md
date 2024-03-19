# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

- Paper Link: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/pdf/2310.06625.pdf)
- Author's code: [https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py)


## 1- introduce

- Point Out: Transformer-based forecasters, which typically embed multiple variates of the same timestamp into indistinguishable channels and apply attention on these  `temporal tokens` to capture temporal dependencies. 
  - Transformer-based forecasters may be not suitable for multivariate time series forecasting. the points of the same time step that basically represent completely different physical meanings recorded by inconsistent measurements are embeded into <font color=darkred>one token with wiped-out multivariate correlations.</font>
  -  a single time step can struggle to reveal beneficial information: 1) excessively local receptive field; 2) unaligned timestamps of multivariate time points 
- Inverted view: embed the whole time series of each variate independently

- 主要贡献
  - We reflect on the architecture of Transformer and refine that the competent capability of native Transformer components on time series is underexplored(我们对 Transformer 的架构进行了反思，并细化了原生 Transformer 组件在时间序列上的能力还有待探索。)
  - We propose iTransformer that regards independent time series as tokens to capture multivariate correlations by self-attention and utilize layer normalization and feed-forward network modules to learn better series-global representations for time series forecasting(我们提出 iTransformer，将独立时间序列视为token，通过自注意力捕获多元相关性，并利用层归一化和前馈网络模块来学习更好的序列全局表示以进行时间序列预测)
  - 通过实验，iTransformer 在现实世界的预测基准上实现了一致的最先进水平。 我们广泛分析了倒置模块和架构选择，为基于 Transformer 的预测器的未来改进指明了一个有希望的方向。

## 2- Overview

encoder-only architecture of Transformer
1. Embedding the whole series as the token 
   1. author find the approach on the numerical modality can be loss instructive for learning attention maps, which is supported by increasing applications of Patching
   2. formulate: 
      1. $h^0_{n}=\text{Embedding}(X_{:,n})$ -->  $\mathcal{R}^T -MLP-> \mathcal{R}^D$
      2. $H^{l+1}=\text{TrmBlock}(H^l), \ l=0, ..., L-1$ self-attention for mutual interactions, and individually processed by feed-forward networks for series representations
      3. $\hat{Y}_{:,n}=\text{Projection}(h^{L}_n)$ -->  $\mathcal{R}^D -MLP-> \mathcal{R}^S$
2. iTransformers
   1. the attention mechanism should be applicable for multivariate correlation modeling

## 3- INVERTED TRANSFORMER COMPONENTS

- Layer normalization
  - increase the convergence and tranining stability of deep networks
  - which has been studied and proved effective in tackling non-stationary problems [(Kim et al., 2021; Liu et al., 2022b)](https://openreview.net/pdf?id=cGDAkQo1C0p)
  - $\text{LayerNorm}(\mathbf{H})= \{\frac{h_n - \text{Mean}(h_n)}{\sqrt{\text{Var}(h_n)}} | n = 1, ..., N \}$
- Feed-forward network
  -  temporal features extracted by MLPs are supposed to be shared within distinct time series
  -  MLP taught to portray the intrinsic proerties of any time series: amplitude, periodicity, and even freqency spectrums
- Self-attention
  - the inverted model regards the whole series of on variate as an individual process







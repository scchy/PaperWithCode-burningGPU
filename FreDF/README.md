# FREDF: LEARNING TO FORECAST IN FREQUENCY DOMAIN

- Paper Link: [FREDF: LEARNING TO FORECAST IN FREQUENCY DOMAIN](https://arxiv.org/pdf/2402.02399)
- Author's code: [https://github.com/Master-PLC/FreDF](https://github.com/Master-PLC/FreDF)

## 1- introduce

- We recoginze and formulate the impact of <font color=darkred> label autocorrelation ignored by current DF paradigm in forecasting. </font>
- We propose FreDF for time series forecasting. As an embarrasingly simply update to DF, it handles label autocorrelation by learning to forecast in the frequency domain. To our knowledge, it is the first attempt to employ frequency analysis for enhancing forecast paradigms.
- We verify the efficacy of FreDF through extensive experiments, where it outperforms state-of-the-art methods substantially and supports various forecast models.

## 2- Overview

$$L^{freq} = \sum_{n=1}^N|F(n) - \hat{F}(n)|$$
$$L^{\alpha} = \alpha L^{freq} + (1-\alpha)L^{tmp}$$

simple example
```python
def quick_fft_loss_fn(outputs, batch_y):
    loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
    return (loss_auxi.abs()**2).mean()


loss = criterion(outputs, batch_y)
if fft_loss_weight is not None:
    loss = fft_loss_weight * quick_fft_loss_fn(outputs, batch_y) + (1 - fft_loss_weight) * loss
```

[Detail Loss function: auxi_loss_type_fn](./freDF_loss.py)

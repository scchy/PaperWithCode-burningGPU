# Spatio-Temporal Graph Convolutional Networks

- Paper Link: [arXiv](https://arxiv.org/pdf/1709.04875v4.pdf)
- Author's code: [https://github.com/VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18)
- reference blogs: 
    - [Build your first Graph Neural Network model to predict traffic speed in 20 minutes](https://towardsdatascience.com/build-your-first-graph-neural-network-model-to-predict-traffic-speed-in-20-minutes-b593f8f838e5)
    - [Knowing Your Neighbours: Machine Learning on Graphs](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896)
- reference paper: 
    - [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
    - [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)
- reference code: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)
- data 
    - [metr-la.h5](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
    - [sensor_graph](https://github.com/chnsh/DCRNN_PyTorch/tree/pytorch_scratch/data/sensor_graph)
- [dgl docs](https://docs.dgl.ai/install/index.html)


```text
file tree
.
├── data
│   ├── metr-la.h5
│   └── sensor_graph
│       ├── distances_la_2012.csv
│       └── graph_sensor_ids.txt
├── load_data.py
├── main.py
├── model.py
├── sensors2graph.py
├── stgcnwavemodel.pt
├── train.ipynb
└── utils.py
```

## Summary

use historical speed data to predict the speed at a future time step. 
propose a novel deep learning framework STGCN for traffic prediction.

### **ST-Conv blocks**

|  Symbol   | meaning  |
|  ----  | ----  |
| M | length of history time serires  |
| n | number of nodes  |

#### <font color=darkred>TemporalConv</font>: Gated CNNs for Extracting Temporal Features
get time series features
> Note: Channel dim=1 in pytorch nn.Conv2d
$$[P Q] = Conv(x); \\ out = P \odot \sigma (Q)$$
- $x \in \mathbb{R}^{C_i \times M \times n }$
- $[\text{P  Q}] \in \mathbb{R}^{2C_o * (M - K_t + 1) \times n }$

simple code:
```python
class TCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, dia: int=1):
        """TemporalConvLayer
        input_dim:  (batch_size, 1, his_time_seires_len, node_num)
        sample:     [b, 1, 144, 207]
        Args:
            c_in (int): channel in
            c_out (_type_): channel out
            dia (int, optional): The dilation size. Defaults to 1.
        """
        super(TCN, self).__init__()
        self.c_out = c_out * 2
        self.c_in = c_in
        self.conv = nn.Conv2d(
            c_in, self.c_out, (2, 1), 1, padding=(0, 0), dilation=dia
        )

    def forward(self, x):
        # [batch, channel, his_n, node_num] 
        #   kernel only filter on TimeSeries dim  his_n
        c = self.c_out//2
        out = self.conv(x)
        if len(x.shape) == 3: # channel, his_n, node_num
            P = out[:c, :, :]
            Q = out[c:, :, :]
        else:
            P = out[:, :c, :, :]
            Q = out[:, c:, :, :]
        return P * torch.sigmoid(Q)
```

#### <font color=darkred>SpatialConv</font>: Graph CNNs for Extracting Spatial Features
$$out= \Theta_{* \mathcal{G}} x = \sum_{k=0}^{K-1}\theta_k T_k(\tilde{L})x=\sum_{k=0}^{K-1}W^{K, l}z^{k, l}$$
- $Z^{0, l} = H^{l}$
- $Z^{1, l} = \tilde{L} \cdot H^{l}$
- $Z^{k, l} = 2 \cdot \tilde{L} \cdot Z^{k-1, l} - Z^{k-2, l}$
- $\tilde{L} = 2\left(I - \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\right)/\lambda_{max} - I$

> [Recursive formulation for fast filtering](https://arxiv.org/pdf/1606.09375.pdf)

simple code:
```python
class STCN_Cheb(nn.Module):
    def __init__(self, c, A, K=2):
        """spation cov layer
        Args:
            c (int): hidden dimension
            A (adj matrix): adj matrix
        """
        super(STCN_Cheb, self).__init__()
        self.K = K
        self.lambda_max = 2
        self.tilde_L = self.get_tilde_L(A)
        self.weight = nn.Parameter(torch.empty((K * c, c)))
        self.bias = nn.Parameter(torch.empty(c))
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def get_tilde_L(self, A):
        I = torch.diag(torch.Tensor([1] * A.size(0))).float().to(A.device)
        tilde_A = A + I 
        tilde_D = torch.diag(torch.pow(tilde_A.sum(axis=1), -0.5))
        return 2 / self.lambda_max * (I - tilde_D @ tilde_A @ tilde_D) - I

    def forward(self, x):
        # For a signal with Ci channels M n C
        # [batch, channel, his_n, node_num] -> [batch, node_num, his_n, channel] -> [batch, his_n, node_num, channel] 
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        output = self.m_unnlpp(x)
        output = output @ self.weight + self.bias
        # return dim
        output = output.transpose(1, 2)
        output = output.transpose(1, 3)
        return torch.relu(output) 

    def m_unnlpp(self, feat):
        K = self.K
        X_0 = feat
        Xt = [X_0]
        # X_1(f)
        if K > 1:
            X_1 = self.tilde_L @ X_0
            # Append X_1 to Xt
            Xt.append(X_1)
        # Xi(x), i = 2...k
        for _ in range(2, K):
            X_i =  2 * self.tilde_L @ X_1 - X_0
            # Add X_1 to Xt
            Xt.append(X_i)
            X_1, X_0 = X_i, X_1
        # Create the concatenation
        Xt = torch.cat(Xt, dim=-1)
        return Xt
```

#### <font color=darkred>ST-Block</font>

```python
class STBlock(nn.Module):
    def __init__(
        self,
        A,
        K=2,
        TST_channel: List=[64, 16, 64]
        T_dia: List=[2, 4]
    ):
        # St-Conv Block1[  TCN(64, 16*2)->SCN(16, 16)->TCN(16, 64*2) ] 
        super(STBlock, self).__init__()
        self.T1 = TCN(TST_channel[0], TST_channel[1], dia=T_dia[0])
        # STCN_Cheb out have relu
        self.S = STCN_Cheb(TST_channel[1], Lk=A, K=K)
        self.T2 = TCN(TST_channel[1], TST_channel[2], dia=T_dia[1])

    def forward(self, x):
        return self.T2(self.S(self.T1(x)))
```

#### <font color=darkred>STGCN_WAVE</font>

```python

class STGCN_WAVE(nn.Module):
    def __init__(self, 
        c: List, 
        T: int, 
        pred_n: int,
        n: int,
        Lk: torch.Tensor,
        K: int, 
        device: torch.device, 
        control_str: str="TNTSTNTST"
    ):
        """STGCN_WAVE

        Args:
            c (List): blocks will defined model size , len(c) == len(args.control_str.replace('N', '')) - 1
            T (int):  input time series length; v_{t-M+1},...,v_{t}
            pred_n (int):  output time series length; \hat{v_{t+1}},...,\hat{v_{t+H}}
            n (int): num of nodes
            Lk (dgl.DGLHeteroGraph):  graph matrix
            k (int):  SCN_Cheb k 
            device (torch.device): train model in which device
            control_str (str, optional): model architecture. Defaults to "TNTSTNTST".
        """
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str  # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "T":  # Temporal Layer
                self.layers.append(
                    TCN(c[cnt], c[cnt + 1], dia=2**diapower)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(STCN_Cheb(c[cnt], Lk, K=K))
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, c[cnt]]))

        self.output = OutputLayer(c[cnt], T + 1 - 2 ** (diapower), n, pred_n)
        for layer in self.layers:
            layer = layer.to(device)

    def forward(self, x):
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == "N":
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.layers[i](x)
        return self.output(x)

```

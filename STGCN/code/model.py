import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

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



class SCN_Cheb(nn.Module):
    def __init__(self, c, A, K=2):
        """spation cov layer
        Args:
            c (int): hidden dimension
            A (adj matrix): adj matrix
        """
        super(SCN_Cheb, self).__init__()
        D_ = torch.diag(torch.pow(A.sum(axis=1), -0.5))
        self.K = K
        self.DAD = D_ @ A @ D_
        # based on the conception of spectral graph convolution
        # https://github.com/tkipf/gcn
        # reference paper: 
        #   [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
        #   [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)
        # reference blogs:
        #   [Knowing Your Neighbours: Machine Learning on Graphs](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896)
        self.SCN_conv_0 = nn.Conv2d(
            c, c, (2, 1), 1, padding=(0, 0)
        )
        self.SCN_conv_1 = nn.Conv2d(
            K * c, c, (2, 1), 1, padding=(0, 0)
        )

    def init(self):
        stdv = 1.0 / np.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # For a signal with Ci channels M n C
        # [batch, channel, his_n, node_num] -> [batch, node_num, his_n, channel] -> [batch, his_n, node_num, channel] 
        x = self.SCN_conv_0(x)
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        output = self.m_unnlpp(x)
        # return dim
        output = output.transpose(1, 2)
        output = output.transpose(1, 3)
        output = self.SCN_conv_1(output)
        return output 

    def m_unnlpp(self, feat):
        K = self.K
        X_0 = feat
        Xt = [X_0]
        # X_1(f)
        if K > 1:
            h = self.DAD @ X_0
            X_1 = -h
            # Append X_1 to Xt
            Xt.append(X_1)

        # Xi(x), i = 2...k
        for _ in range(2, K):
            print('a')
            h = self.DAD @ X_1
            X_i = -2 * 1 * h - X_0
            # Add X_1 to Xt
            Xt.append(X_i)
            X_1, X_0 = X_i, X_1

        # Create the concatenation
        Xt = torch.cat(Xt, dim=-1)
        return Xt


class SCN_Cheb_linear(nn.Module):
    def __init__(self, c, A, K=2):
        """spation cov layer
        Args:
            c (int): hidden dimension
            A (adj matrix): adj matrix
        """
        super(SCN_Cheb_linear, self).__init__()
        D_ = torch.diag(torch.pow(A.sum(axis=1), -0.5))
        self.K = K
        self.DAD = D_ @ A @ D_
        # based on the conception of spectral graph convolution
        # https://github.com/tkipf/gcn
        # reference paper: 
        #   [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)
        #   [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)
        # reference blogs:
        #   [Knowing Your Neighbours: Machine Learning on Graphs](https://medium.com/stellargraph/knowing-your-neighbours-machine-learning-on-graphs-9b7c3d0d5896)
        self.weight = nn.Parameter(torch.empty((K * c, c)))
        self.bias = nn.Parameter(torch.empty(c))
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

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
        return output 

    def m_unnlpp(self, feat):
        K = self.K
        X_0 = feat
        Xt = [X_0]
        # X_1(f)
        if K > 1:
            h = self.DAD @ X_0
            X_1 = -h
            # Append X_1 to Xt
            Xt.append(X_1)

        # Xi(x), i = 2...k
        for _ in range(2, K):
            h = self.DAD @ X_1
            X_i = -2 * 1 * h - X_0
            # Add X_1 to Xt
            Xt.append(X_i)
            X_1, X_0 = X_i, X_1

        # Create the concatenation
        Xt = torch.cat(Xt, dim=-1)
        return Xt


class FullyConvLayer(nn.Module):
    def __init__(self, c, n):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, n, 1)

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n, pred_n):
        super(OutputLayer, self).__init__()
        # [batch, channel, his_n, node_num] conv his_n -> 1
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.fc = FullyConvLayer(c, pred_n)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2).squeeze(2)


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
        s_cnt = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "T":  # Temporal Layer
                self.layers.append(
                    TCN(c[cnt], c[cnt + 1], dia=2**diapower)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SCN_Cheb(c[cnt], Lk, K=K))
                s_cnt += 2
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, c[cnt]]))
        print(T + 1 - 2 ** (diapower) - s_cnt)
        self.output = OutputLayer(c[cnt], T + 1 - 2 ** (diapower) - s_cnt, n, pred_n)
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

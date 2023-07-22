import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# conda create -n stgcn_wave python=3.8
# conda activate stgcn_wave
# pip install dgl-cu117 -f https://data.dgl.ai/wheels/repo.html
# pip install torch
# pip install pandas
# pip install tables
# pip install scikit-learn
# pip install seaborn
# pip install table
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv
from typing import List
import dgl


class TemporalConvLayer(nn.Module):
    """Temporal convolution layer.

    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    """

    def __init__(self, c_in, c_out, dia=1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(
            c_in, c_out, (2, 1), 1, dilation=dia, padding=(0, 0)
        )

    def forward(self, x):
        return torch.relu(self.conv(x))


class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk):  # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = GraphConv(c, c, activation=F.relu)
        # self.gc = ChebConv(c, c, 3)

    def init(self):
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return torch.relu(output)


class FullyConvLayer(nn.Module):
    def __init__(self, c, n):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, n, 1)

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n, pred_n):
        super(OutputLayer, self).__init__()
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
        Lk: dgl.DGLHeteroGraph,
        p: float, 
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
            p (float): drop out rate
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
                    TemporalConvLayer(c[cnt], c[cnt + 1], dia=2**diapower)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk))
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
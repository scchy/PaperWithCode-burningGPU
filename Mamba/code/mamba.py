
import torch 
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader, Dataset 
from torch.nn import functional as F  
from einops import rearrange
import math 
import os 
from transformers import AutoTokenizer
from argparse import Namespace


USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0

config = Namespace(
    d_model = 8
    ,state_size = 128 # Example state size
    ,seq_len = 100 # Example sequence length
    ,batch_size = 256 # Example batch size
    ,last_batch_size = 81 # only for the very last batch of the dataset
    ,current_batch_size = 256
    ,different_batch_size = False
    ,h_new = None
    ,temp_buffer = None
    ,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)



class S6(nn.Module):
    def __init__(self, cfg):
        super(S6, self).__init__()
        self.seq_len = cfg.seq_len
        self.d_model = cfg.d_model 
        self.state_size = cfg.state_size
        self.fc1 = nn.Linear(self.d_model, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.state_size)
        self.fc3 = nn.Linear(self.d_model, self.state_size)

        self.A = nn.Parameter(
            F.normalize(torch.ones(self.d_model, self.state_size), p=2, dim=-1)
        )
        nn.init.xavier_uniform_(self.A)
        
        self.B = torch.zeros(cfg.batch_size , self.seq_len, self.state_size)
        self.C = torch.zeros(cfg.batch_size , self.seq_len, self.state_size)
        self.delta = torch.zeros(cfg.batch_size , self.seq_len, self.state_size)
        self.dA = torch.zeros(self.state_size, self.seq_len, self.d_model, self.state_size)
        self.dB = torch.zeros(self.state_size, self.seq_len, self.d_model, self.state_size)
        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(self.state_size, self.seq_len, self.d_model, self.state_size)
        self.y = torch.zeros(self.state_size, self.seq_len, self.d_model)

    def discretization(self):
        self.dB = torch.einsum('bld,bln->bldn', self.delta, self.B)
        self.dA = torch.exp(torch.einsum('bld,dn->bldn', self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        self.discretization()
        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            global current_batch_size
            current_batch_size = x.shape[0]
            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB
            else:
                different_batch_size = False
                h_new =  torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
            return self.y

        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        y = torch.zeros_like(x)
        h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB
        # y [batch_size, seq_len, d_model]
        y = torch.einsum('bln,bldn->bld', self.C, h)
        return y


class MambaBlock(nn.Module):
    def __init__(self, cfg):
        super(MambaBlock, self).__init__()
        d_model = cfg.d_model
        self.inp_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.out_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.D = nn.Linear(d_model, 2* d_model)
        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True
        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)
        self.S6 = S6(cfg)
        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(cfg.seq_len, cfg.seq_len, kernel_size=3, padding=1)
        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2*d_model, 2*d_model)
        # rmsnorm
        self.norm = RMSNorm(d_model)
    
    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)
        
        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)
        return x_out


class Mamba(nn.Module):
    def __init__(self, cfg):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(cfg)
        self.mamba_block2 = MambaBlock(cfg)
        self.mamba_block3 = MambaBlock(cfg)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        ) * self.weight
        return output



x = torch.rand(config.batch_size, config.seq_len, config.d_model)
# Create the Mamba model
mamba = Mamba(config)
# rmsnorm
norm = RMSNorm(config.d_model)
x = norm(x)
# Forward pass
test_output = mamba(x)

mamba.mamba_block1(x)
print(f"test_output.shape = {test_output.shape}") 


 
x = mamba.mamba_block1.norm(x)
x_proj = mamba.mamba_block1.inp_proj(x)
# Add 1D convolution with kernel size 3
x_conv = mamba.mamba_block1.conv(x_proj)
x_conv_act = F.silu(x_conv)
# Add linear layer for conv output
x_conv_out = mamba.mamba_block1.conv_linear(x_conv_act)
x_ssm = mamba.mamba_block1.S6(x_conv_out)
x_conv_out.shape 
B = mamba.mamba_block1.S6.fc2(x_conv_out)
C = mamba.mamba_block1.S6.fc3(x)
delta = F.softplus(mamba.mamba_block1.S6.fc1(x))
mamba.mamba_block1.S6.discretization()





x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

# residual skip connection with nonlinearity introduced by multiplication
x_residual = F.silu(mamba.mamba_block1.D(x))
x_combined = x_act * x_residual
x_out = mamba.mamba_block1.out_proj(x_combined)
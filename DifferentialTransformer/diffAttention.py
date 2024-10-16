# python3
# Create Date: 2024-10-16
# Author: Scc_hy
# reference: https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
# ======================================================================================================

import math 
import torch 
from torch import nn
from torch.nn import functional as F 
from argparse import Namespace


config = Namespace(
    n_head=8, 
    n_embd=(32*2)*8,
    n_layers=12,
    max_position_embeddings=1024,
    attn_pdrop=0.1,
    resid_pdrop=0.1
)


class RMSNorm(nn.Module):
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6, 
        elementwise_affine=True, 
        memory_efficient=False
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        # x: [batch, n_head, seq_length, head_dim]
        # rsqrt = 1 / sqrt(x)
        # [batch, n_head, seq_length, head_dim] mean in head_dim
        x_opt = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.rsqrt(x_opt)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'



def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def lambda_init_fn(dp):
    return 0.8 - 0.6 * math.exp(-0.3 * dp)


class CasualMultiheadDiffAttn(nn.Module):
    def __init__(self, cfg, layer_idx=0):
        super(CasualMultiheadDiffAttn, self).__init__()
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_dim = self.n_embd // self.n_head
        self.layer_idx = layer_idx
        self.scaling = torch.full([], math.sqrt(self.head_dim // 2))
        self.max_positions = cfg.max_position_embeddings

        self.q_proj = nn.Linear(self.n_embd, self.n_embd , bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd , bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd , bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd , bias=False)
        
        self.lambda_init = lambda_init_fn(self.layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.head_ln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        
    def _split_heads(self, tensor, n_head, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and n_head
        """
        new_shape = tensor.size()[:-1] + (n_head, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, n_head, seq_length, head_features)

    def _merge_heads(self, tensor, n_head, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (n_head * attn_head_size,)
        return tensor.view(new_shape)
    
    def _casual_mask(self, attn_weights, max_positions=None):
        if max_positions is None:
            max_positions = self.max_positions
        causal_mask = torch.tril(
            torch.ones((max_positions, max_positions), dtype=torch.bool)
        ).view(1, 1, max_positions, max_positions)
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to( attn_weights.device)
        # where mask
        return torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    
    def _diff_attn(self, attn_weights):
        r""" 
            DiffAttn(X) = softmax(\frac{Q_1K_1^T}{\sqrt{d}}) - \lambda softmax(\frac{Q_2K_2^T}{\sqrt{d}})
            \lambda = e^{\lambda_{q_1}\lambda_{k_1}} - e^{\lambda_{q_2}\lambda_{k_2}} + \lambda_{init}
        """
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        b, _, _, l = attn_weights.size()
        attn_weights = attn_weights.view(b, self.n_head, 2, l, l)
        # -> [batch, n_head, seq_length, seq_length]
        attn_weights = attn_weights[:, :, 0, ...] - lambda_full * attn_weights[:, :, 1, ...]
        return self.attn_dropout(attn_weights)

    def forward(self, x):
        """
        Q, K, V = XW^q, XW^k, XW^v
        A_doubleHead = casual_mask( _split_heads(Q) _split_heads(K)^T )
        A_doubleHead = softmax(A_doubleHead)
        ★ A = A_doubleHead[:, :, 0, ...] - \lambda A_doubleHead[:, :, 1, ...]
        O = AV
        O = (1 - lambda_init)O 
        O = _merge_heads(O)W^o
        """
        b, l, e = x.size()
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)  # [batch, seq_length, n_embd]
        
        # 1- split_head -> [batch, n_head * 2, seq_length, head_dim // 2]
        Q = self._split_heads(Q, self.n_head * 2, self.head_dim // 2)
        K = self._split_heads(K, self.n_head * 2, self.head_dim // 2)
        V = self._split_heads(V, self.n_head, self.head_dim) # ->  [batch, n_head, seq_length, head_dim]
        
        # 2- casual_attention ->  [batch, n_head * 2, seq_length, seq_length]
        A = torch.matmul(Q, K.transpose(-1, -2)) / self.scaling  
        A = F.softmax(self._casual_mask(A), dim=-1)

        # 3- diff_attn: 
        #   -> [batch, n_head, seq_length, head_dim]
        A = self._diff_attn(A)

        # 4- groupNorm => [A_i for i in range(h)] 
        O = self.head_ln(torch.matmul(A, V))
        O = O * (1.0 - self.lambda_init)
        # 5- merge heads & proj -> [batch, seq_length, n_embd=n_head*head_dim]
        O = self._merge_heads(O, self.n_head, self.head_dim)
        O = self.o_proj(O)
        return self.resid_dropout(O), A



if __name__ == '__main__': 
    casual_diff_attn = CasualMultiheadDiffAttn(config)
    x = torch.randn(10, 1024, 64*8)
    o, a = casual_diff_attn(x)
    print(o.shape, a.shape, casual_diff_attn.lambda_q1.shape)


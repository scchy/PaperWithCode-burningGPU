# python3
# Create Date: 2024-08-01
# Author: Scc_hy
# ========================================================================

import torch 
from torch import nn 
from torch.optim import SGD


class Task(nn.Module):
    def __init__(self):
        self.theta_K = nn.Param((d1, d2))
        self.theta_V = nn.Param((d1, d2))
        self.theta_Q = nn.Param((d1, d2))
    
    def loss(self, f, x):
        tr_view = self.theta_K @ x
        label_view = self.theta_V @ x
        return MSE(f(tr_view), label_view)


class Learner():
    def __init__(self, task):
        self.task = task 
        # Linear here, but can be any model
        self.model = nn.Linear()
        # online GD here for simplicity
        self.opt = SGD()
    
    def train(self, x):
        # grad function wrt first arg
        # of loss, which is self.model
        grad_fn = grad(self.task.loss)
        grand_in = grad_fn(self.model, x)
        # Starting from current params
        # step in direction of grad_in
        self.optim.step(self.model, grad_in)
    
    def predict(self, x):
        te_view = self.task.theta_Q @ x
        return self.model(te_view)


class TTT_layer(nn.Module):
    def __init__(self):
        self.task = Task()
    
    def forward(self, in_seq):
        state = Learner(self.task)
        out_seq = []
        for tok in in_seq:
            state.train(tok)
            out_seq.append(state.predict(tok))
        return out_seq



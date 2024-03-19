import os
import sys
import torch 
from torch import nn
from transformers.models.openai import OpenAIGPTConfig
from transformers.models.openai import OpenAIGPTModel, OpenAIGPTLMHeadModel
sys.path.append('/home/scc/sccWork/myGitHub/PaperWithCode-burningGPU/DecisionTransformer_and_TT/codeTT')
from base_models import GPT

cfg = OpenAIGPTConfig()
gpt_org = OpenAIGPTLMHeadModel(cfg)
cfg.transition_dim = 4
cfg.subsampled_sequence_length = 128
cfg.step = 2
sequence_length  = cfg.subsampled_sequence_length * cfg.step
cfg.block_size = cfg.subsampled_sequence_length * cfg.transition_dim - 1
cfg.observation_dim = 17
cfg.action_dim = 4
cfg.action_weight = 2.0
cfg.reward_weight = 1.0
cfg.value_weight = 1.5
gpt_ = GPT(cfg)

gpt_.pos_emb.shape


gpt_org
# python3
# reference: https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py
# =========================================================================================

import torch 
from torch import nn
from torch.nn import functional as F
from itf_utils import Encoder, EncoderLayer, AttentionLayer, FullAttention, DataEmbedding_inverted


class iTransformer(nn.Module):
    def __init__(self, cfg):
        super(iTransformer, self).__init__()
        self.task_name = cfg.task_name
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.use_norm = cfg.use_norm
        self.output_attention = cfg.output_attention
        # Embedding 
        self.enc_embedding = DataEmbedding_inverted(
            cfg.seq_len, cfg.d_model, 'fixed', 'h', cfg.dropout
        )
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=cfg.dropout, output_attention=cfg.output_attention),
                        cfg.d_model,
                        cfg.n_heads
                    ),
                    cfg.d_model,
                    cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation
                ) for l in range(cfg.e_layers)
            ],
            norm_layer=nn.LayerNorm(cfg.d_model, cfg.n_heads)
        )
        self.x_mark_dec_flag = cfg.x_mark_dec_flag if hasattr(cfg, "x_mark_dec_flag") else  None
        # x_mark_dec
        if self.x_mark_dec_flag is not None:
            self.x_mark_dec_embd = nn.Linear(self.pred_len, cfg.d_model)
        # Decoder:
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(cfg.d_model, cfg.pred_len, bias=True)
        if self.task_name == ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(cfg.d_model, cfg.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(cfg.dropout)
            self.projection = nn.Linear(cfg.d_model * cfg.enc_in, cfg.num_class)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc (_type_): shape=[B: batch_size, L: seq_len, N: number of vars]
            x_mark_enc (_type_, optional): _description_. Defaults to None.
            x_dec (_type_, optional): _description_. Defaults to None.
            x_mark_dec (_type_, optional): _description_. Defaults to None.

        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means 
            std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= std

        _, _, N = x_enc.shape
        
        # Embedding: [B L N] ->( [B N L] ) -> [B N E]
        # if x_mark_enc: [B L Vars] ->( [B L N=Vars + date_emb] -> [B N L] ) -> [B N E] 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encode: [B N E] -> [B N E]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Out [B N E] -> [B N pred_len] -> [B pred_len N]
        if self.x_mark_dec_flag is not None and x_mark_dec is not None:
            # [B pred_len date_var_N] -> ([B date_var_N pred_len]) -> [B date_var_N E] -> [B 1 E]
            enc_extra = self.x_mark_dec_embd(x_mark_dec.permute(0, 2, 1)).mean(dim=1, keepdim=True)
            # Out [B N E] -> [B N pred_len] -> [B pred_len N]
            dec_out = self.projection(enc_out + enc_extra).permute(0, 2, 1)[:, :, :N]
        else:
            # Out [B N E] -> [B N pred_len] -> [B pred_len N]
            dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None



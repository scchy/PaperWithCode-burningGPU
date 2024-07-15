import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math
import numpy as np 
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
import time
from tqdm.auto import tqdm
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, atten_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(atten_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.output_attention = atten_layers[0].output_attention
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B L D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(self.atten_layers, self.conv_layers):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.atten_layers[-1](x, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)

        return x, attns 
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model 
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.output_attention = attention.output_attention
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        # Feed Forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        self.inner_attention = attention
        self.output_attention = attention.output_attention
        self.query_proj = nn.Linear(d_model, d_keys * n_heads)
        self.key_proj = nn.Linear(d_model, d_keys * n_heads)
        self.value_proj = nn.Linear(d_model, d_keys * n_heads)
        self.out_proj = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads 
    
    def forward(self, q, k, v, attn_mask, tau=None, delta=None):
        B, L, _ = q.shape
        H = self.n_heads 
        
        Q = self.query_proj(q).view(B, L, H, -1)
        K = self.key_proj(k).view(B, L, H, -1)
        V = self.value_proj(v).view(B, L, H, -1)
        
        out, attn = self.inner_attention(Q, K, V, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_proj(out), attn 


class FullAttention(nn.Module):
    def __init__(self,  mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale 
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, Q, K, V, attn_mask, tau=None, delta=None):
        B, L, H, E = Q.shape
        _, S, _, D = V.shape 
        scale = self.scale or 1. / math.sqrt(E)
        # [batch-b, vars=l, head-h, embed-e], [batch-b, vars-s, head-h, embed-e] -> [batch-b, head-h, vars-l, vars-s]
        a_scores = torch.einsum('blhe,bshe->bhls', Q, K)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=Q.device)
            
            a_scores.masked_fill_(attn_mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * a_scores, dim=-1)) # [batch-b, head-h, vars-l, vars-s]
        # [batch-b, head-h, vars-l, vars-s], [batch-b, vars-s, head-h, embed-d] -> [batch-b, vars-l, head-h, embed-d]
        O = torch.einsum("bhls,bshd->blhd", A, V)
        if self.output_attention:
            return O.contiguous(), A
        return O.contiguous(), None


class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len: int, d_model: int, embed_type: str='fixed', freq: str='h', dropout: float=0.1):
        """_summary_
        Args:
            seq_len (int): seq len
            d_model (int): embedding dim
            embed_type (str, optional): embed type. Defaults to 'fixed'.
            freq (str, optional): sequence freq. Defaults to 'h'.
            dropout (float, optional): dropout. Defaults to 0.1.
        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embd = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, x_mark):
        # B L 
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embd(x)
        else:
            x = self.value_embd(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # B V d_model
        return self.dropout(x)


class Dataset_hour(Dataset):
    def __init__(
        self, 
        base_df,
        target_cols,
        flag='train', 
        seq_len=4*7*24,
        pred_len=24, 
        scale=True
        ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.target_cols = target_cols 
        self.scale = scale
        self.base_df = base_df
        self.__prepare_data()


    def __prepare_data(self):
        self.scaler = StandardScaler()
        # 0 day_x, 1 weekday_x, 2  hour_x
        date_series = pd.to_datetime(self.base_df['date'])
        date_emb = np.ones((len(date_series), 4), dtype=np.float32)
        date_emb[:, 0] = (date_series.dt.day - 1) / 30.0 - 0.5
        date_emb[:, 1] = date_series.dt.dayofweek / 6.0 - 0.5
        date_emb[:, 2] = date_series.dt.hour / 23.0 - 0.5
        date_emb[:, 3] = (date_series.dt.month  - 1) / 11.0 - 0.5

        df_raw = self.base_df
        base_ = 30 *  24
        border1s = [0, 12 * base_ - self.seq_len, (12 + 4) * base_ - self.seq_len]
        border2s = [12 * base_, (12 + 4) * base_, (12 + 8) * base_]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        print(f"set_type({self.set_type}): {border1} -> {border2} len={border2 - border1 - self.seq_len - self.pred_len + 1}")
        df_data = df_raw[self.target_cols]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2, :]
        self.data_y = data[border1:border2, :]
        self.date_emb = date_emb[border1:border2, :]
        print(f"df_raw.shape={df_raw.shape} data.shape={data.shape} self.data_x.shape={self.data_x.shape}")      

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        pred_end = s_end  + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:pred_end]
        seq_x_mark = self.date_emb[s_begin:s_end]
        seq_y_mark = self.date_emb[s_end:pred_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_sp(Dataset):
    def __init__(
            self,
            base_df,
            target_cols,
            flag='train',
            seq_len=4 * 7 * 24,
            pred_len=24,
            scale=True,
            return_dt=False,
            border1s=None,
            border2s=None
    ):
        self.return_dt = return_dt
        self.seq_len = seq_len
        self.pred_len = pred_len
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.border1s = border1s
        self.border2s = border2s
        self.target_cols = target_cols
        self.scale = scale
        self.base_df = base_df
        self.__prepare_data()

    def __prepare_data(self):
        self.scaler = StandardScaler()
        # 0 day_x, 1 weekday_x, 2  hour_x
        date_series = pd.to_datetime(self.base_df['date'])
        date_emb = np.ones((len(date_series), 4), dtype=np.float32)
        date_emb[:, 0] = (date_series.dt.day - 1) / 30.0 - 0.5
        date_emb[:, 1] = date_series.dt.dayofweek / 6.0 - 0.5
        date_emb[:, 2] = date_series.dt.hour / 23.0 - 0.5
        date_emb[:, 3] = (date_series.dt.month - 1) / 11.0 - 0.5

        df_raw = self.base_df
        base_ = 30 * 24
        border1s = [0, 8 * base_ - self.seq_len, (8 + 3) * base_ - self.seq_len] if self.border1s is None else self.border1s
        border2s = [8 * base_, (8 + 3) * base_, (8 + 6) * base_] if self.border2s is None else self.border2s

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        print(f"set_type({self.set_type}): {border1} -> {border2} len={border2 - border1 - self.seq_len - self.pred_len + 1}")
        df_data = df_raw[self.target_cols]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2, :]
        self.data_y = data[border1:border2, :]
        self.date_emb = date_emb[border1:border2, :]
        self.date_org = date_series.dt.strftime('%Y%m%d%H').map(int).values[border1:border2]
        print(f"df_raw.shape={df_raw.shape} data.shape={data.shape} self.data_x.shape={self.data_x.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        pred_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:pred_end]
        seq_x_mark = self.date_emb[s_begin:s_end]
        seq_y_mark = self.date_emb[s_end:pred_end]

        if self.return_dt:
            seq_x_dt = self.date_org[s_begin:s_end]
            seq_y_dt = self.date_org[s_end:pred_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_dt, seq_y_dt
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def train(model, train_loader, vali_loader, test_loader, cfg, invert_tf=None):
    device = cfg.device
    model = model.to(device)
    path = cfg.save_dir
    if not os.path.exists(path):
        os.makedirs(path)
    
    best_model_path = os.path.join(path, 'checkpoint.pth')
    time_now = time.time()
    train_steps = len(train_loader)
    model_optim = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    test_best_loss = np.inf
    x_mark_dec_flag = cfg.x_mark_dec_flag if hasattr(cfg, "x_mark_dec_flag") else  None
    ep_bar = tqdm(range(cfg.num_epochs))
    for epoch in ep_bar:
        ep_bar.set_description(f'[ {str(epoch+1).zfill(3)} /{str(cfg.num_epochs).zfill(3)} ]')
        iter_count = 0
        train_loss = []
        inv_loss_record = []
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            seq_x_mark = seq_x_mark.float().to(device)
            seq_y_mark = seq_y_mark.float().to(device)

            outputs = model(batch_x, seq_x_mark, x_mark_dec=seq_y_mark if x_mark_dec_flag else None)
            outputs = outputs[:, -cfg.pred_len:, :]
            batch_y = batch_y[:, -cfg.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            if invert_tf is not None:
                np_outputs = np.stack([invert_tf(i) for i in outputs[:, -cfg.pred_len:, :].detach().cpu().numpy()])
                np_batch_y = np.stack([invert_tf(i) for i in batch_y[:, -cfg.pred_len:, :].detach().cpu().numpy()])
                inv_loss = np.mean((np_outputs - np_batch_y) ** 2)
                inv_loss_record.append(inv_loss)

            train_loss.append(loss.item())

            ep_bar.set_postfix(dict(
                loss=f"{loss.item():.5f}",
                inv_loss=f'{inv_loss:.5f}' if invert_tf is not None else '-'
            ))
            if (i + 1) % 100 == 0:
                # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((cfg.num_epochs - epoch) * train_steps - i)
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()

        # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        train_inv_loss = np.mean(inv_loss_record) if len(inv_loss_record) else 0
        vali_loss = test_model(model, vali_loader, cfg, criterion)
        test_loss = test_model(model, test_loader, cfg, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f}(invLoss: {5:.5f}) Vali Loss: {3:.5f} Test Loss: {4:.5f}".format(
            str(epoch + 1).zfill(3), train_steps, train_loss, vali_loss, test_loss, train_inv_loss))
        if test_best_loss > vali_loss:
            test_best_loss = vali_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model: loss={test_best_loss:.5f}')
    
    model.load_state_dict(torch.load(best_model_path))
    return model



@torch.no_grad()
def test_model(model, te_loader, cfg, criterion):
    device = cfg.device
    model = model.to(device)
    model.eval()
    train_loss = []
    for i, (batch_x, batch_y, seq_x_mark, seq_y_mark) in enumerate(te_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        seq_x_mark = seq_x_mark.float().to(device)
        seq_y_mark = seq_y_mark.float().to(device)

        outputs = model(batch_x, seq_x_mark)
        outputs = outputs[:, -cfg.pred_len:, :]
        batch_y = batch_y[:, -cfg.pred_len:, :].to(device)
        loss = criterion(outputs, batch_y)
        # print(f'outputs.shape = {outputs.shape}, batch_y.shape = {batch_y.shape} loss={loss}')
        train_loss.append(loss.item())

    return np.mean(train_loss)
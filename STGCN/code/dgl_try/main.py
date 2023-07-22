# python3
# Create date: 20230703
# train: python main.py --batch_size=32 --epochs=10
# ===================================================================
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.getcwd()))
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from load_data import data_transform
from model_dgl import *
from sensors2graph import get_adjacency_matrix
from sklearn.preprocessing import StandardScaler
from utils import evaluate_model, evaluate_metric, args_func, device
import dgl


code_path = Path('.').absolute().parent
args = args_func()
with open(code_path.joinpath(Path(args.sensorsfilepath))) as f:
    sensor_ids = f.read().strip().split(",")

disf = code_path.joinpath(Path(args.disfilepath))
distance_df = pd.read_csv(disf, dtype={"from": "str", "to": "str"})
adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
sp_mx = sp.coo_matrix(adj_mx)
# use dgl generate graph
G = dgl.from_scipy(sp_mx)

tsf = code_path.joinpath(Path(args.tsfilepath))
df = pd.read_hdf(tsf)
num_samples, num_nodes = df.shape

tsdata = df.to_numpy()

n_his = args.window
save_path = args.savemodelpath


n_pred = args.pred_len
n_route = num_nodes
blocks = args.channels
# blocks = [1, 16, 32, 64, 32, 128]
drop_prob = 0
num_layers = args.num_layers
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr

W = adj_mx
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[:len_train]
val = df[len_train : len_train + len_val]
test = df[len_train + len_val :]

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(
    blocks, n_his, n_pred, n_route, G, drop_prob, device, args.control_str
).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    tq_bar = tqdm(train_iter)
    tq_bar.set_description(f"[ Train | {epoch:03d} / {epochs:03d} ]")
    for x, y in tq_bar:
        x, y = x.to(device), y.to(device)
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.cpu().item() * y.shape[0]
        n += y.shape[0]
        tq_bar.set_postfix({'loss': "{:.5f}".format(l_sum / (n+0.0001))})

    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print(f"epoch={epoch}, train loss:{l_sum / n:.5f}, validation loss:{val_loss:.5f}")


best_model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
best_model.load_state_dict(torch.load(save_path, map_location='cpu'))


l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
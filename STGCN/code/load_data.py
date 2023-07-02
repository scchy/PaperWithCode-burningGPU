# python3
# Create date: 2023-07-02
# Func: load data
# ==================================================

import numpy as np
import pandas as pd
import torch


def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[:len_train]
    val = df[len_train:len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_route = data.shape[1]
    l = len(data)
    num = l - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])

    cnt = 0
    for i in range(1-n_his-n_pred):
        head = i
        tail = i + n_his
        x[cnt, ...] = data[head:tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    return torch.Tensor(x), torch.Tensor(y).to(device)


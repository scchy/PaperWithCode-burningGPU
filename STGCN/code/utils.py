import numpy as np
import torch
import argparse

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


@torch.no_grad()
def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.cpu().item() * y.shape[0]
        n += y.shape[0]
    return l_sum / n


@torch.no_grad()
def evaluate_metric(model, data_iter, scaler):
    model.eval()
    mae, mape, mse = [], [], []
    for x, y in data_iter:
        x, y = x.to(device), y.to(device)
        y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
        y_pred = scaler.inverse_transform(
            model(x).view(len(x), -1).cpu().numpy()
        ).reshape(-1)
        d = np.abs(y - y_pred)
        mae += d.tolist()
        mape += (d / y).tolist()
        mse += (d * d).tolist()
    
    MAE = np.array(mae).mean()
    MAPE = np.array(mape).mean()
    RMSE = np.sqrt(np.array(mse).mean())
    return MAE, MAPE, RMSE



def args_func():
    parser = argparse.ArgumentParser(description="STGCN_WAVE")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="batch size for training and validation (default: 50)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="epochs for training  (default: 50)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=9, help="number of layers"
    )
    parser.add_argument("--window", type=int, default=144, help="window length")
    parser.add_argument(
        "--sensorsfilepath",
        type=str,
        default="./data/sensor_graph/graph_sensor_ids.txt",
        help="sensors file path",
    )
    parser.add_argument(
        "--disfilepath",
        type=str,
        default="./data/sensor_graph/distances_la_2012.csv",
        help="distance file path",
    )
    parser.add_argument(
        "--tsfilepath", type=str, default="./data/metr-la.h5", help="ts file path"
    )
    parser.add_argument(
        "--savemodelpath",
        type=str,
        default="stgcnwavemodel.pt",
        help="save model path",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=5,
        help="how many steps away we want to predict",
    )
    parser.add_argument(
        "--control_str",
        type=str,
        default="TNTSTNTST",
        help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[1, 16, 32, 64, 32, 128],
        help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
    )
    return parser.parse_args(args=[])

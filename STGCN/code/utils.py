import numpy as np
import torch


@torch.no_grad()
def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in data_iter:
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

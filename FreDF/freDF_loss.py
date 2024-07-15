# python3
# reference: https://github.com/Master-PLC/FreDF/blob/main/exp/exp_long_term_forecasting.py
# ==========================================================================================
import torch 
import numpy as np 
import numpy as np
import torch
from numpy.polynomial import Chebyshev as C
from numpy.polynomial import Hermite as H
from numpy.polynomial import Laguerre as La
from numpy.polynomial import Legendre as L



def leg_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)  # The Legendre series are defined in t\in[-1, 1]
    legendre_polys = np.array([L.basis(i)(tvals) for i in range(degree)])  # Generate the basis functions which are sampled at tvals.
    # tvals = torch.from_numpy(tvals).to(device)
    legendre_polys = torch.from_numpy(legendre_polys).float().to(device)  # shape: [degree, T]

    # This is implemented for 1D series. 
    # For N-D series, here, the data matrix should be transformed as B,T,D -> B,D,T -> BD, T. 
    # The legendre polys should be T,degree
    # Then, the dot should be a matrix multiplication: (BD, T) * (T, degree) -> BD, degree, which is the result of legendre transform.
    coeffs_candidate = torch.mm(legendre_polys, data) / T * 2
    coeffs = torch.stack([coeffs_candidate[i] * (2 * i + 1) / 2 for i in range(degree)]).to(device)
    coeffs = coeffs.transpose(0, 1)  # shape: [B * D, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def laguerre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(0, 5, T)
    laguerre_polys = np.array([La.basis(i)(tvals) for i in range(degree)])

    laguerre_polys = torch.from_numpy(
        laguerre_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals))
    coeffs_candidate = torch.mm(laguerre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(laguerre_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, laguerre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def chebyshev_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T)
    chebyshev_polys = np.array([C.basis(i)(tvals) for i in range(degree)])

    chebyshev_polys = torch.from_numpy(chebyshev_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(1 / torch.sqrt(1 - tvals ** 2))
    coeffs_candidate = torch.mm(chebyshev_polys, data) / torch.pi / T * 2
    # coeffs_candidate = torch.mm(torch.mm(chebyshev_polys, scale), data) / torch.pi * 2
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(chebyshev_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, chebyshev_polys)
        reconstructed_data = reconstructed_data.reshape(B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1

    ndim = data.ndim
    shape = data.shape
    if ndim == 2:
        B = 1
        T = shape[0]
    elif ndim == 3:
        B, T = shape[:2]
        data = data.permute(1, 0, 2).reshape(T, -1)
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([H.basis(i)(tvals) for i in range(degree)])

    hermite_polys = torch.from_numpy(
        hermite_polys).float().to(device)  # shape: [degree, T]
    # tvals = torch.from_numpy(tvals).float().to(device)
    # scale = torch.diag(torch.exp(-tvals ** 2))
    coeffs_candidate = torch.mm(hermite_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)  # shape: [B * D, degree]
    # coeffs = torch.linalg.lstsq(hermite_polys.T, data).solution.T

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, hermite_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs


def auxi_loss_type_fn(
    outputs, 
    batch_y, 
    device, 
    auxi_lambda=1,
    auxi_mode='fft', auxi_type='complex', leg_degree=2, module_first=1,
    auxi_loss_type='MSE'
    ):
    if auxi_mode == "fft":
        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)

    elif auxi_mode == "rfft":
        if auxi_type == 'complex':
            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
        elif auxi_type == 'complex-phase':
            loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
        elif auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
            loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif auxi_type == 'phase':
            loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
        elif auxi_type == 'mag':
            loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
        elif auxi_type == 'mag-phase':
            loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
            loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

    elif auxi_mode == "rfft-D":
        loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)

    elif auxi_mode == "rfft-2D":
        loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)

    elif auxi_mode == "legendre":
        loss_auxi = leg_torch(outputs, leg_degree, device=device) - leg_torch(batch_y, leg_degree, device=device)

    elif auxi_mode == "chebyshev":
        loss_auxi = chebyshev_torch(outputs, leg_degree, device=device) - chebyshev_torch(batch_y, leg_degree, device=device)

    elif auxi_mode == "hermite":
        loss_auxi = hermite_torch(outputs, leg_degree, device=device) - hermite_torch(batch_y, leg_degree, device=device)

    elif auxi_mode == "laguerre":
        loss_auxi = laguerre_torch(outputs, leg_degree, device=device) - laguerre_torch(batch_y, leg_degree, device=device)
    else:
        raise NotImplementedError


    if auxi_loss_type == "MAE":
        # MAE, 最小化element-wise error的模长
        loss_auxi = loss_auxi.abs().mean() if module_first else loss_auxi.mean().abs()  # check the dim of fft
    elif auxi_loss_type == "MSE":
        # MSE, 最小化element-wise error的模长
        loss_auxi = (loss_auxi.abs()**2).mean() if module_first else (loss_auxi**2).mean().abs()
    else:
        raise NotImplementedError

    return auxi_lambda * loss_auxi


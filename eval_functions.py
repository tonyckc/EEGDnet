import math
import torch
import numpy as np
from numpy import mean, sqrt, square
from numpy.linalg import det


def rms(x):
    return sqrt(mean(square(x), axis=1))


def PSD(x):
    x = np.fft.fft(x)
    return np.abs(x) ** 2


def ACC(x, f_y):
    x = x.numpy()
    f_y = f_y.numpy()
    out = 0
    i = 0
    for i in range(x.shape[0]):
        cov_f_y_x = np.cov(f_y[i, :], x[i, :])
        var_f_y = np.var(f_y[i, :])
        var_x = np.var(x[i, :])
        out += cov_f_y_x[0, 1] / (sqrt(var_f_y * var_x))
    return out / (i+1)


def RRMSE_spectral(x, f_y):
    x = x.numpy()
    f_y = f_y.numpy()

    return mean(rms(PSD(f_y) - PSD(x)) / rms(PSD(x)))


def RRMSE_temporal(x, f_y):
    x = x.numpy()
    f_y = f_y.numpy()

    return mean(rms(f_y - x) / rms(x))

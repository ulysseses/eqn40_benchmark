# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import scipy.misc as sm
import utils
import time


def padzeros(mat, dim):
    h, w = dim
    oldh, oldw = mat.shape
    if h < oldh or w < oldw:
        raise ValueError("Size mismatch: (h: %d, w: %d) < (oldh: %d, oldw: %d)" % \
            (h, w, oldh, oldw))
    
    newmat = np.zeros(dim, dtype=mat.dtype)
    newmat[:oldh, :oldw] = mat
    return newmat


def psf2otf(psf, dim):
    h, w = psf.shape
    mat = padzeros(psf, dim)
    mat = np.roll(mat, -h//2, axis=0)
    mat = np.roll(mat, -w//2, axis=1)
    otf = np.fft.fft2(mat)
    return otf


def filterFFT(input, kernel, abs=True, correlate=False):
    newshape = (input.shape[0] * 2, input.shape[1] * 2)
    X = np.fft.fft2(input)
    Y = psf2otf(kernel, input.shape)
    if abs:
        Y = np.abs(Y)
    if correlate:
        Z = X * Y.conj()
    else:
        Z = X * Y
    z = np.real(np.fft.ifft2(Z))
    return z


def deriv_x():
    return np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)


def deriv_y():
    return np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)


class FourierSolver(utils.CommonSolver):
    def __init__(self, k, p, J, l, y):
        super(FourierSolver, self).__init__(k, p, J, l, y)

        # Pre-fetch data
        self.k = k
        self.K = np.abs(psf2otf(k, y.shape)) # conj?
        self.Y = np.fft.fft2(self.y)

        # Pre-calculate denominator
        Dx_mag2 = np.abs(psf2otf(deriv_x(), y.shape)) ** 2
        Dy_mag2 = np.abs(psf2otf(deriv_y(), y.shape)) ** 2
        self.D_mag2 = Dx_mag2 + Dy_mag2
        self.K_mag2 = np.abs(self.K) ** 2

    def benchmark(self, x_hat, eta):
        # Boilerplate to create necessary inputs of benchmarked routine
        d_x, d_y = self.boilerplate(x_hat, eta)
        d_diffT = np.empty_like(d_x, dtype=np.float32)

        # Benchmark 
        start_time = time.time()

        d_diffT[:, 0] = d_x[:, -1] - d_x[:, 0]
        d_diffT[:, 1:] = -np.diff(d_x, axis=1)
        d_diffT[0, :] += d_y[-1, :] - d_y[0, :]
        d_diffT[1:, :] += -np.diff(d_y, axis=0)
        DTd = np.fft.fft2(d_diffT)

        numer = (self.l * eta) * DTd
        numer += np.conj(self.K) * self.Y

        X_hat = numer / ((self.l * eta) * self.D_mag2 + self.K_mag2)
        x_hat = np.real(np.fft.ifft2(X_hat))

        duration = time.time() - start_time

        return x_hat, duration

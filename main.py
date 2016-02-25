# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import scipy.misc as sm
import utils
import fourier
import conjgrad
import scipy.ndimage


def main_boilerplate():
    # Load image and kernel
    img = sm.face()
    x = utils.rgb2ycc(img.astype(np.float32) / 255.)[:, :, 0]
    k = utils.gaussian_kernel(2, 3.5)
    noise = np.random.normal(0., 0.01, x.shape).astype(np.float32)
    return img, x, k, noise


def eval_psnr(x_hat, x, border=0):
    x_hat_crop = x_hat[border:-border, border:-border]
    x_crop = x[border:-border, border:-border]
    diff = x_hat_crop - x_crop
    mse = np.sqrt(np.mean(diff**2))
    psnr = 20. * np.log10(1. / mse)
    return psnr


def benchmark_gisa(eta0, etamax, rho, T, p, J, l, times):
    def gisa(solver, y, k, eta0, etamax, rho, T):
        x_hat = y
        eta = eta0
        itr = 0
        while eta < etamax:
            for t in range(T):
                x_hat, _ = solver.benchmark(x_hat, eta)
            eta *= rho
        return x_hat
    
    # Load image and kernel
    img, x, k, _ = main_boilerplate()
    border = k.shape[0] // 2
    
    # Fourier Solution
    y0 = fourier.filterFFT(x, k, abs=True, correlate=False)
    psnr = 0.
    for t in range(times):
        y = y0 + np.random.normal(0., 0.01, x.shape).astype(np.float32)
        fourier_solver = fourier.FourierSolver(k, p, J, l, y)
        x_hat = gisa(fourier_solver, y, k, eta0, etamax, rho, T)
        x_hat = np.clip(x_hat, 0., 1.)
        psnr += eval_psnr(x_hat, x, border)
    psnr /= times
    print("Fourier PSNR: %.2f" % psnr)
    
    # Conjugate Gradient Solution
    y0 = scipy.ndimage.convolve(x, k, mode='constant')
    psnr = 0.
    for t in range(times):
        y = y0 + np.random.normal(0., 0.01, x.shape).astype(np.float32)
        conj_grad_solver = conjgrad.ConjGradSolver(k, p, J, l, y)
        x_hat = gisa(conj_grad_solver, y, k, eta0, etamax, rho, T)
        x_hat = np.clip(x_hat, 0., 1.)
        psnr += eval_psnr(x_hat, x, border)
    psnr /= times
    print("ConjGrad PSNR: %.2f" % psnr)
    
    # Baseline
    baseline_psnr = eval_psnr(y, x, border)
    print("Baseline PSNR: %.2f" % baseline_psnr)


def benchmark_eqn40(eta, p, J, l, times):
    def helper(solver, y, eta, times, border, name):
        avg_duration = 0.
        for t in range(times):
            _, duration = solver.benchmark(y, eta)
            avg_duration += duration
        avg_duration /= times
        print("%s duration: %.3f" % (name, avg_duration))
    
    # Load image and kernel
    img, x, k, noise = main_boilerplate()
    border = k.shape[0] // 2
    
    # Fourier Solution
    y = fourier.filterFFT(x, k, abs=True, correlate=False) + noise
    fourier_solver = fourier.FourierSolver(k, p, J, l, y)
    helper(fourier_solver, y, eta, times, border, "Fourier")

    # Conjugate Gradient Solution
    y = scipy.ndimage.convolve(x, k, mode='constant') + noise
    conj_grad_solver = conjgrad.ConjGradSolver(k, p, J, l, y)
    helper(conj_grad_solver, y, eta, times, border, "ConjGrad")


if __name__ == '__main__':
    # Common parameters
    eta0 = 1.0
    etamax = 2. ** 8
    rho = 2. * np.sqrt(2)
    T = 1
    p = 0.8
    J = 2
    l = 0.0005
    times = 5
    
    #benchmark_eqn40(eta0, p, J, l, times)
    benchmark_gisa(eta0, etamax, rho, T, p, J, l, times)

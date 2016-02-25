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


def benchmark_gisa(eta0, etamax, rho, T, p, J, times):
    def gisa(y, k, ):
        
    # Load image and kernel
    img, x, k, noise = main_boilerplate()
    border = k.shape[0] // 2
    
    # Fourier Solution
    y = fourier.filterFFT(x, k, abs=True, correlate=False) + noise
    fourier_solver = fourier.FourierSolver(k, p, J, y)
    
    x_hat = y
    eta = eta0
    itr = 0
    while eta < etamax:
        for t in range(T):
            x_hat, duration = fourier_solver.benchmark(x_hat, eta)
            eta *= rho
            
    # Baseline
    x_crop = x[border:-border, border:-border]
    y_crop = y[border:-border, border:-border]
    baseline_psnr = 20. * np.log10(1.0 / np.mean((x_crop - y_crop)**2))
    print("Baseline PSNR: %.2f" % baseline_psnr)
    


def benchmark_eqn40(eta, p, J, times):
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
    fourier_solver = fourier.FourierSolver(k, p, J, y)
    helper(fourier_solver, y, eta, times, border, "Fourier")

    # Conjugate Gradient Solution
    y = scipy.ndimage.convolve(x, k, mode='constant') + noise
    conj_grad_solver = conjgrad.ConjGradSolver(k, p, J, y)
    helper(conj_grad_solver, y, eta, times, border, "ConjGrad")


if __name__ == '__main__':
    # Common parameters
    eta0 = 1.0
    etamax = 2. ** 8
    rho = 2. * np.sqrt(2)
    T = 1
    p = 0.8
    J = 2
    times = 5
    
    benchmark_eqn40(eta0, p, J, times)
    #benchmark_gisa(eta0, etamax, rho, T, p, J, times)
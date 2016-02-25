# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import scipy.misc as sm
import utils
import time
import scipy.sparse.linalg


class ConjGradSolver(utils.CommonSolver):
    def __init__(self, k, p, J, l, y):
        super(ConjGradSolver, self).__init__(k, p, J, l, y)

        self.k_conj = np.flipud(np.fliplr(k))
        self.b1 = scipy.ndimage.convolve(self.y, self.k_conj, mode='constant');

    @staticmethod
    def _matvec_functor(k, k_conj, l, eta, m, n):
        Av = np.empty((m, n), dtype=np.float32)
        Av2 = np.empty((m, n), dtype=np.float32)
        Av_x = np.empty((m, n), dtype=np.float32)
        Av_y = np.empty((m, n), dtype=np.float32)
        
        def inner(k, k_conj, l, eta, m, n, Av, Av2, Av_x, Av_y, v):
            v = v.reshape(m, n)

            scipy.ndimage.convolve(v, k, output=Av, mode='constant');
            scipy.ndimage.convolve(Av, k_conj, output=Av2, mode='constant');
            
            utils.circdiff2D(v, Av_x, 1);
            utils.circdiff2D(v, Av_y, 0);
            Av[:, 0] = Av_x[:, -1] - Av_x[:, 0]
            Av[:, 1:] = -np.diff(Av_x, axis=1)
            Av[0, :] += Av_y[-1, :] - Av_y[0, :]
            Av[1:, :] += -np.diff(Av_y, axis=0)
            Av *= (l * eta)
            Av += Av2

            Av = Av.flatten()
            return Av
        
        matvec = lambda v: inner(k, k_conj, l, eta, m, n, Av, Av2, Av_x, Av_y, v)
        return matvec

    def benchmark(self, x_hat, eta):
        # Boilerplate to create necessary inputs of benchmarked routine
        d_x, d_y = self.boilerplate(x_hat, eta)
        b = np.empty_like(x_hat, dtype=np.float32)
        m, n = x_hat.shape
        A = scipy.sparse.linalg.LinearOperator((m*n, m*n),
            self._matvec_functor(self.k, self.k_conj, self.l, eta, m, n))

        start_time = time.time()

        b[:, 0] = d_x[:, -1] - d_x[:, 0]
        b[:, 1:] = -np.diff(d_x, axis=1)
        b[0, :] += d_y[-1, :] - d_y[0, :]
        b[1:, :] += -np.diff(d_y, axis=0)
        b *= self.l * eta
        b += self.b1

        x_hat, info = scipy.sparse.linalg.cg(A, b.flatten(), x0=x_hat.flatten(),
                                             maxiter=2, tol=1e-4)
        x_hat = x_hat.reshape(m, n)
        #print("######DEBUG info:", info)

        duration = time.time() - start_time

        return x_hat, duration

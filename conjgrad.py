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
    def _matvec_functor(k, k_conj, l, eta_inv, m, n):
        Av = np.empty((m, n), dtype=np.float32)
        Av2 = np.empty((m, n), dtype=np.float32)
        Av2_x = np.empty((m, n), dtype=np.float32)
        Av2_y = np.empty((m, n), dtype=np.float32)
        
        def inner(k, k_conj, l, eta_inv, m, n, Av, Av2, Av2_x, Av2_y, v):
            v = v.reshape(m, n)

            scipy.ndimage.convolve(v, k, output=Av, mode='constant');
            scipy.ndimage.convolve(Av, k_conj, output=Av, mode='constant');
            
            utils.circdiff2D(v, Av2_x, 1);
            utils.circdiff2D(v, Av2_y, 0);
            Av2[:, 0] = Av2_x[:, -1] - Av2_x[:, 0]
            Av2[:, 1:] = -np.diff(Av2_x, axis=1)
            Av2[0, :] += Av2_y[-1, :] - Av2_y[0, :]
            Av2[1:, :] += -np.diff(Av2_y, axis=0)

            Av += l * eta_inv * Av2

            Av = Av.flatten()
            return Av
        
        matvec = lambda v: inner(k, k_conj, l, eta_inv, m, n, Av, Av2, Av2_x, Av2_y, v)
        return matvec

    def benchmark(self, x_hat, eta_inv):
        # Boilerplate to create necessary inputs of benchmarked routine
        d_x, d_y = self.boilerplate(x_hat, eta_inv)
        b = np.empty_like(x_hat, dtype=np.float32)
        m, n = x_hat.shape
        A = scipy.sparse.linalg.LinearOperator((m*n, m*n),
            self._matvec_functor(self.k, self.k_conj, self.l, eta_inv, m, n))

        start_time = time.time()

        b[:, 0] = d_x[:, -1] - d_x[:, 0]
        b[:, 1:] = -np.diff(d_x, axis=1)
        b[0, :] += d_y[-1, :] - d_y[0, :]
        b[1:, :] += -np.diff(d_y, axis=0)
        b *= self.l * eta_inv
        b += self.b1

        x_hat, info = scipy.sparse.linalg.cg(A, b.flatten(), x0=x_hat.flatten(),
                                             maxiter=5, tol=4.98e-2)
        #print("######DEBUG info:", info)

        duration = time.time() - start_time

        return x_hat, duration

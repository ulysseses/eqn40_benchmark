# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import scipy.misc as sm
import abc


def gaussian_kernel(r=1, sd=1.0):
    kernel = np.empty((2*r + 1, 2*r + 1), dtype=np.float32)
    s = 0
    for i in range(2*r + 1):
        for j in range(2*r + 1):
            val = np.exp(-0.5 * (float((i - r)**2) + float((j - r)**2)) / (sd*sd))
            kernel[i, j] = val
            s += val
    kernel /= s
    return kernel


def rgb2ycc(img_rgb):
    img_ycc = np.empty_like(img_rgb, dtype=np.float32)
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]

    y[:] = .299*r + .587*g + .114*b
    cb[:] = .5 -.168736*r -.331364*g + .5*b
    cr[:] = .5 +.5*r - .418688*g - .081312*b

    img_ycc = np.clip(np.round(img_ycc), 0., 1.)
    return img_ycc


def ycc2rgb(img_ycc):
    img_rgb = np.empty_like(img_ycc, dtype=np.float32)
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    r[:] = y + 1.402 * (cr-.5)
    g[:] = y - .34414 * (cb-.5) -  .71414 * (cr-.5)
    b[:] = y + 1.772 * (cb-.5)

    img_rgb = np.clip(np.round(img_rgb), 0., 1.)
    return img_rgb


def circdiff2D(src, dst=None, axis=0):
    if axis not in (0, 1): raise ValueError("axis must be 0 or 1")
    dst = dst if dst is not None else np.empty_like(src, dtype=src.dtype)
    if axis == 0:
        dst[:-1, :] = np.diff(src, axis=0)
        dst[-1, :] = src[0, :] - src[-1, :]
    else:
        dst[:, :-1] = np.diff(src, axis=1)
        dst[:, -1] = src[:, 0] - src[:, -1]
    return dst


class CommonSolver(object):
    def __init__(self, k, p, J, l, y):
        self.k = k
        self.y = y
        self.p = p
        self.J = J
        self.l = l

    def GST(self, y, eta_inv):
        p = self.p
        J = self.J

        # Threshold value
        tau_val = (2*eta_inv*(1-p))**(1/(2-p)) + \
            eta_inv*p*((2*eta_inv*(1-p))**((p-1)/(2-p)))
        # Locate thresholded locations
        thresh_locs = np.abs(y) > tau_val
        # Evaluate soft thresholds
        y_vals = y[thresh_locs]
        x = np.abs(y_vals)
        for _ in xrange(J):
            x = np.abs(y_vals) - eta_inv*p*(x**(p - 1))
        out = np.zeros_like(y)
        out[thresh_locs] = np.sign(y_vals) * x
        return out

    def boilerplate(self, x_hat, eta_inv):
        d_ref_x = circdiff2D(x_hat, axis=1)
        d_ref_y = circdiff2D(x_hat, axis=0)
        d_x = self.GST(d_ref_x, 1./eta_inv)
        d_y = self.GST(d_ref_y, 1./eta_inv)
        return d_x, d_y

    @abc.abstractmethod
    def benchmark(self, x, eta_inv):
        raise NotImplementedError

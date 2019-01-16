#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:22:25 2017

@author: mousse
"""
import numpy as np
from scipy import optimize
'''
 interpolation.py contains interpolation and regression for DeltaDOGS and AlphaDOGS
    NPS( Natural Polyharmonic Spline ):
        P(x) = sum(w_i * phi(r) ) + v.T * [1; x]
        phi(r) = r**3, r = norm(x-xi)
        
        Inter_par:                          Contains the parameter w, v and xi(seldom used) for NPS, a for MAPS.
        interpolateparameterization:        Perform NPS interpolation on DeltaDOGS
        regressionparametarization:         Perform NPS regression on AlphaDOGS
        smoothing_polyharmonic
        interpolate_val:                    Calculate interpolation/regression function values in the parameter space.
        interpolate_grad:                   Calculate the gradient of interpolation/regression in the parameter space.
        interpolate_hessian:                Calculate the hessian matrix of interpolation/regression in the parameter space.
        
'''

################################## Interpolation and Regression ########################################################

class Inter_par():
    def __init__(self, method="NPS", w=0, v=0, xi=0, a=0):
        self.method = "NPS"
        self.w = []
        self.v = []
        self.xi = []
        self.a = []


def interpolateparameterization(xi, yi, inter_par):
    n = xi.shape[0]
    m = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros((m, m))
        for ii in range(m):
            for jj in range(m):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)

        V = np.vstack((np.ones((1, m)), xi))
        A1 = np.hstack((A, np.transpose(V)))
        A2 = np.hstack((V, np.zeros((n + 1, n + 1))))
        yi = yi[np.newaxis, :]
        b = np.concatenate([np.transpose(yi), np.zeros((n + 1, 1))])
        A = np.vstack((A1, A2))
        wv = np.linalg.lstsq(A, b, rcond=-1)
        wv = np.copy(wv[0])
        inter_par.w = wv[:m]
        inter_par.v = wv[m:]
        inter_par.xi = xi
        yp = np.zeros(m)
        for ii in range(m):
            yp[ii] = interpolate_val(xi[:, ii], inter_par)
        return inter_par, yp


def regressionparametarization(xi, yi, sigma, inter_par):
    # Notice xi, yi and sigma must be a two dimension matrix, even if you want it to be a vector.
    # or there will be error
    n = xi.shape[0]
    N = xi.shape[1]
    if inter_par.method == 'NPS':
        A = np.zeros((N, N))
        for ii in range(N):  # for ii =0 to m-1 with step 1; range(1,N,1)
            for jj in range(N):
                A[ii, jj] = (np.dot(xi[:, ii] - xi[:, jj], xi[:, ii] - xi[:, jj])) ** (3.0 / 2.0)
        V = np.concatenate((np.ones((1, N)), xi), axis=0)
        w1 = np.linalg.lstsq(np.dot(np.diag(1 / sigma), V.T), (yi / sigma).reshape(-1, 1), rcond=None)
        w1 = np.copy(w1[0])
        b = np.mean(np.divide(np.dot(V.T, w1) - yi.reshape(-1, 1), sigma.reshape(-1, 1)) ** 2)
        wv = np.zeros([N + n + 1])
        if b < 1:
            wv[N:] = np.copy(w1.T)
            rho = 1000
            wv = np.copy(wv.reshape(-1, 1))
        else:
            rho = 1.1
            fun = lambda rho: smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)[0]
            rho = optimize.fsolve(fun, rho)
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
        inter_par.w = wv[:N]
        inter_par.v = wv[N:N + n + 1]
        inter_par.xi = xi
        yp = np.zeros(N)
        while (1):
            for ii in range(N):
                yp[ii] = interpolate_val(xi[:, ii], inter_par)
            residual = np.max(np.divide(np.abs(yp - yi), sigma))
            if residual < 2:
                break
            rho *= 0.9
            b, db, wv = smoothing_polyharmonic(rho, A, V, sigma, yi, n, N)
            inter_par.w = wv[:N]
            inter_par.v = wv[N:N + n + 1]
    return inter_par, yp


def smoothing_polyharmonic(rho, A, V, sigma, yi, n, N):
    A01 = np.concatenate((A + rho * np.diag(sigma ** 2), np.transpose(V)), axis=1)
    A02 = np.concatenate((V, np.zeros(shape=(n + 1, n + 1))), axis=1)
    A1 = np.concatenate((A01, A02), axis=0)
    b1 = np.concatenate([yi.reshape(-1, 1), np.zeros(shape=(n + 1, 1))])
    wv = np.linalg.lstsq(A1, b1, rcond=None)
    wv = np.copy(wv[0])
    b = np.mean(np.multiply(wv[:N], sigma.reshape(-1, 1)) ** 2 * rho ** 2) - 1
    bdwv = np.concatenate([np.multiply(wv[:N], sigma.reshape(-1, 1) ** 2), np.zeros((n + 1, 1))])
    Dwv = np.linalg.lstsq(-A1, bdwv, rcond=None)
    Dwv = np.copy(Dwv[0])
    db = 2 * np.mean(np.multiply(wv[:N] ** 2 * rho + rho ** 2 * np.multiply(wv[:N], Dwv[:N]), sigma ** 2))
    return b, db, wv


def interpolate_val(x, inter_par):
    # Each time after optimization, the result value x that optimization returns is one dimension vector,
    # but in our interpolate_val function, we need it to be a two dimension matrix.
    '''
    :param x:           The intended position to calculate the gradient of interpolation/regression function
    :param inter_par:   The parameter set for interpolation/regression
    return:             The interpolation/regression function values at x.
    '''
    x = x.reshape(-1, 1)
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        S = xi - x
        return np.dot(v.T, np.concatenate([np.ones((1, 1)), x], axis=0)) + np.dot(w.T, (
            np.sqrt(np.diag(np.dot(S.T, S))) ** 3))


def interpolate_grad(x, inter_par):
    '''
    :param x:           The intended position to calculate the gradient of interpolation/regression function
    :param inter_par:   The parameter set for interpolation/regression
    return:             The column vector of the gradient information at point x.
    '''
    
    x = x.reshape(-1, 1)
    if inter_par.method == "NPS":
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros([n, 1])
        for ii in range(N):
            X = x - xi[:, ii].reshape(-1, 1)
            g = g + 3 * w[ii] * X * np.linalg.norm(X)
        g = g + v[1:]

    return g


def interpolate_hessian(x, inter_par):
    '''
    :param x:           The intended position to calculate the gradient of interpolation/regression function
    :param inter_par:   The parameter set for interpolation/regression
    return:             The hessian matrix of at x.
    '''
    if inter_par.method == "NPS" or inter_par.method == 1:
        w = inter_par.w
        v = inter_par.v
        xi = inter_par.xi
        n = x.shape[0]
        N = xi.shape[1]
        g = np.zeros((n))
        n = x.shape[0]

        H = np.zeros((n, n))
        for ii in range(N):
            X = x[:, 0] - xi[:, ii]
            if np.linalg.norm(X) > 1e-5:
                H = H + 3 * w[ii] * ((X * X.T) / np.linalg.norm(X) + np.linalg.norm(X) * np.identity(n))
        return H


def inter_cost(x,inter_par):
    x = x.reshape(-1, 1)
    M = interpolate_val(x, inter_par)
    DM = interpolate_grad(x, inter_par)
    return M, DM.T


def inter_min(x, inter_par, lb=[], ub=[]):
    # Find the minimizer of the interpolating function starting with x
    rho = 0.9  # backtracking parameter
    n = x.shape[0]
    x0 = np.zeros((n, 1))
    x = x.reshape(-1, 1)
    objfun = lambda x: interpolate_val(x, inter_par)
    grad_objfun = lambda x: interpolate_grad(x, inter_par)
    opt = {'disp': False}
    bnds = tuple([(0, 1) for i in range(int(n))])
    res = optimize.minimize(objfun, x0, jac=grad_objfun, method='TNC', bounds=bnds, options=opt)
    x = res.x
    y = res.fun[0, 0]
    return x, y

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:19:28 2017

@author: mousse
"""
import numpy as np

'''
Utils_SL.py is implemented to generate results for AlphaDOGS and DeltaDOGS applied on safe learning problems.
    - The objective function is denoted as F(x) and safe constraints are denoted as G(x), both are unknown to the user.
    - The initial safe experiment is given by the user. The performance of the initial experiment might be poor, but the 
        safe constraints will be guaranteed. 
    - Our goal is to find the parameters that satisfied the safe constraints and obtain the best performance.
     
    - The safe constraints will always be guaranteed during the optimization process.
     
This script contains following functions:
    
    bounds          :       Generate vertices under lower bound 'bnd1' and upper bound 'bnd2' for 'n' dimensions;
    mindis          :       Generate the minimum distances and corresponding point from point x to set xi;
    circhyp         :       Generate hypercircumcircle for Delaunay simplex, return the circumradius and center;
    normalize_bounds:       Normalize bounds for Delta DOGS optm solver;
    physical_bounds :       Retransform back to the physical bounds for function evaluation;
    
    search_bounds   :       Generate bounds for the Delaunay simplex during the continuous search minimization process;
    hyperplane_equations:   Determine the math expression for the hyperplane;
     
    test_fun        :       Return the test function info;
    fun_eval        :       Transform point from normalized bounds to the physical boudns;
    random_initial  :       Randomly generate the initial points.
    
'''
################################# Utils ####################################################


def bounds(bnd1, bnd2, n):
    #   find vertex of domain for a box domain.
    #   INPUT: n: dimension, bnd1: lower bound, bnd2: upper bound.
    #   OUTPUT: vertex of domain. 2^n number vector of n-D.
    #   Example:
    #           n = 3
    #           bnd1 = np.zeros((n, 1))
    #           bnd2 = np.ones((n, 1))
    #           bnds = bounds(bnd1,bnd2,n)
    #   Author: Shahoruz Alimohammadi
    #   Modified: Dec., 2016
    #   DELTADOGS package
    assert bnd1.shape == (n, 1) and bnd2.shape == (n, 1), 'lb(bnd1) and ub(bnd2) should be 2 dimensional vector.'
    bnds = np.kron(np.ones((1, 2 ** n)), bnd2)
    for ii in range(n):
        tt = np.mod(np.arange(2 ** n) + 1, 2 ** (n - ii)) <= 2 ** (n - ii - 1) - 1
        bnds[ii, tt] = bnd1[ii]
    return bnds


def mindis(x, xi):
    '''
    calculates the minimum distance from all the existing points
    :param x: x the new point
    :param xi: xi all the previous points
    :return: [ymin ,xmin ,index]
    '''
    x = x.reshape(-1, 1)
    assert xi.shape[1] != 0, 'The set of Evaluated points has size 0!'
    dis = np.linalg.norm(xi-x, axis=0)
    val = np.min(dis)
    idx = np.argmin(dis)
    xmin = xi[:, idx].reshape(-1, 1)
    return val, idx, xmin


def modichol(A, alpha, beta):
    #   Modified Cholesky decomposition code for making the Hessian matrix PSD.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    n = A.shape[1]  # size of A
    L = np.identity(n)
    ####################
    D = np.zeros((n, 1))
    c = np.zeros((n, n))
    ######################
    D[0] = np.max(np.abs(A[0, 0]), alpha)
    c[:, 0] = A[:, 0]
    L[1:n, 0] = c[1:n, 0] / D[0]

    for j in range(1, n - 1):
        c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
        for i in range(j + 1, n):
            c[i, j] = A[i, j] - (np.dot((L[i, 0:j] * L[j, 0:j]).reshape(1, j), D[0:j]))[0, 0]
        theta = np.max(c[j + 1:n, j])
        D[j] = np.array([(theta / beta) ** 2, np.abs(c[j, j]), alpha]).max()
        L[j + 1:n, j] = c[j + 1:n, j] / D[j]
    j = n - 1
    c[j, j] = A[j, j] - (np.dot((L[j, 0:j] ** 2).reshape(1, j), D[0:j]))[0, 0]
    D[j] = np.max(np.abs(c[j, j]), alpha)
    return np.dot(np.dot(L, (np.diag(np.transpose(D)[0]))), L.T)


def circhyp(x, N):
    # circhyp     Circumhypersphere of simplex
    #   [xC, R2] = circhyp(x, N) calculates the coordinates of the circumcenter
    #   and the square of the radius of the N-dimensional hypersphere
    #   encircling the simplex defined by its N+1 vertices.
    #   Author: Shahoruz Alimohammadi
    #   Modified: Jan., 2017
    #   DOGS package

    test = np.sum(np.transpose(x) ** 2, axis=1)
    test = test[:, np.newaxis]
    m1 = np.concatenate((np.matrix((x.T ** 2).sum(axis=1)), x))
    M = np.concatenate((np.transpose(m1), np.matrix(np.ones((N + 1, 1)))), axis=1)
    a = np.linalg.det(M[:, 1:N + 2])
    c = (-1.0) ** (N + 1) * np.linalg.det(M[:, 0:N + 1])
    D = np.zeros((N, 1))
    for ii in range(N):
        M_tmp = np.copy(M)
        M_tmp = np.delete(M_tmp, ii + 1, 1)
        D[ii] = ((-1.0) ** (ii + 1)) * np.linalg.det(M_tmp)
        # print(np.linalg.det(M_tmp))
    # print(D)
    xC = -D / (2.0 * a)
    #	print(xC)
    R2 = (np.sum(D ** 2, axis=0) - 4 * a * c) / (4.0 * a ** 2)
    #	print(R2)
    return R2, xC


def normalize_bounds(x0, lb, ub):
    n = len(lb)  # n represents dimensions
    m = x0.shape[1]  # m represents the number of sample data
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j] - lb[i]) / (ub[i] - lb[i])
    return x


def physical_bounds(x0, lb, ub):
    '''
    :param x0: normalized point
    :param lb: real lower bound
    :param ub: real upper bound
    :return: physical scale of the point
    '''
    n = len(lb)  # n represents dimensions
    try:
        m = x0.shape[1]  # m represents the number of sample data
    except:
        m = x0.shape[0]
    x = np.copy(x0)
    for i in range(n):
        for j in range(m):
            x[i][j] = (x[i][j])*(ub[i] - lb[i]) + lb[i]

    return x

# this function can be deleted.
def search_bounds(xi):
    n = xi.shape[0]
    srch_bnd = []
    for i in range(n):
        rimin = np.min(xi[i, :])
        rimax = np.max(xi[i, :])
        temp = (rimin, rimax)
        srch_bnd.append(temp)
    simplex_bnds = tuple(srch_bnd)
    return simplex_bnds


def search_simplex_bounds(xi):
    '''
    Return the n+1 constraints defined by n by n+1 Delaunay simplex xi.
    The optimization for finding minimizer of Sc should be within the Delaunay simplex.
    :param xi: xi should be (n) by (n+1). Each column denotes a data point.
    :return: Ax >= b constraints.
    A: n+1 by n
    b: n+1 by 1
    '''
    # TODO fix when dim = 1.
    n = xi.shape[0]  # dimension of input
    m = xi.shape[1]  # number of input, should be exactly the same as n+1.
    # The linear constraint, which is the boundary of the Delaunay triangulation simplex.
    A = np.zeros((m, n))
    b = np.zeros((m, 1))
    for i in range(m):
        direction_point = xi[:, i].reshape(-1, 1)  # used to determine the type of inequality, <= or >=
        plane_points = np.delete(xi, i, 1)  # should be an n by n square matrix.
        A[i, :], b[i, 0] = hyperplane_equations(plane_points)
        if np.dot(A[i, :].reshape(-1, 1).T, direction_point) < b[i, :]:
            # At this point, the simplex stays at the negative side of the equation, assign minus sign to A and b.
            A[i, :] = np.copy(-A[i, :].reshape(-1, 1).T)
            b[i, 0] = np.copy(-b[i, :])
    return A, b


def hyperplane_equations(points):
    '''
    Return the equation of n points hyperplane in n dimensional space.

    Reference website:
    https://math.stackexchange.com/questions/2723294/how-to-determine-the-equation-of-the-hyperplane-that-contains-several-points

    :param points: Points is an n by n square matrix. Each column represents a data point.
    :return: A and b (both 2 dimensional array) that satisfy Ax = b.
    '''
    n, m = points.shape  # n dimension of points. m should be the same as n
    base_point = points[:, -1].reshape(-1, 1)
    matrix = (points - base_point)[:, :-1].T  # matrix should be n-1 by n, each row represents points - base_point.
    A = np.zeros((1, n))
    b = np.zeros((1, 1))
    for j in range(n):
        block = np.delete(matrix, j, 1)  # The last number 1, denotes the axis. 1 is columnwise while 0 is rowwise.
        A[0, j] = (-1) ** (j+1) * np.linalg.det(block)
    b[0, 0] = np.dot(A, base_point)
    return A, b


def test_fun(fun_arg, n):
    fun = lb = ub = xmin = y0 = fname = []
    if fun_arg == 1:  # 2D test function: Goldstein-price
        # Notice, here I take ln(y)
        lb = -2 * np.ones((2, 1))
        ub = 2 * np.ones((2, 1))
        fun = lambda x: np.log((1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*\
                        (30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)))
        y0 = np.log(3)
        xmin = np.array([0.5, 0.25])
        fname = 'Goldstein-price'

    elif fun_arg == 2:  # schewfel
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        fun = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250
        y0 = -1.6759316 * n  # targert value for objective function
        xmin = 0.8419 * np.ones((n, 1))
        fname = 'Schewfel'

    elif fun_arg == 3:  # rastinginn
        A = 3
        lb = -2 * np.ones((n, 1))
        ub = 2 * np.ones((n, 1))
        fun = lambda x: (sum((x - 0.7) ** 2 - A * np.cos(2 * np.pi * (x - 0.7)))) / 1.5
        y0 = 0.0
        xmin = 0.7 * np.ones((n, 1))
        fname = 'Rastinginn'
    # fun_arg == 4: Lorenz Chaotic system.

    elif fun_arg == 5:  # schwefel + quadratic
        fun = lambda x: - x[0][0] / 2 * np.sin(np.abs(500 * x[0][0])) + 10 * (x[1][0] - 0.92) ** 3
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -0.44528425
        xmin = np.array([0.89536, 0.94188])
        fname = 'Schwefel + Quadratic'

    elif fun_arg == 6:  # Griewank function
        fun = lambda x: 1 + 1 / 4 * ((x[0][0] - 0.67) ** 2 + (x[1][0] - 0.21) ** 2) - np.cos(x[0][0]) * np.cos(
            x[1][0] / np.sqrt(2))
        lb = np.zeros((n, 1))
        ub = np.ones((n,1 ))
        y0 = 0.08026
        xmin = np.array([0.21875, 0.09375])
        fname = 'Griewank'

    elif fun_arg == 7:  # Shubert function
        tt = np.arange(1, 6)
        fun = lambda x: np.dot(tt, np.cos((tt + 1) * (x[0][0] - 0.45) + tt)) * np.dot(tt, np.cos(
            (tt + 1) * (x[1][0] - 0.45) + tt))
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -32.7533
        xmin = np.array([0.78125, 0.25])
        fname = 'Shubert'

    elif fun_arg == 8:  # 2D DimRed test function:  the Branin function
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        fun = lambda x: a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s
        lb = np.array([-5, 0]).reshape(-1, 1)
        ub = np.array([10, 15]).reshape(-1, 1)
        y0 = 0.397887
        # 3 minimizer
        xmin = np.array([0.5427728, 0.1516667])  # True minimizer np.array([np.pi, 2.275])
        # xmin2 = np.array([-np.pi, 12.275])
        # xmin3 = np.array([9.42478, 2.475])
        fname = 'Branin'

    elif fun_arg == 9:  # Six hump camel back function
        fun = lambda x: (4 - 2.1 * (x[0][0] * 3) ** 2 + (x[0][0] * 3) ** 4 / 3) * (x[0][0] * 3) ** 2 - (
                x[0][0] * x[1][0] * 6) + (-4 + 16 * x[1][0] ** 2) * x[1][0] ** 2 * 4
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        y0 = -1.0316
        xmin = np.array([0.029933, 0.3563])
        fname = 'Six hump camel'

    elif fun_arg == 10:  # Rosenbrock function
        fun = lambda x: np.sum(100*(x[:-1] - x[1:]**2)**2) + np.sum((x-1)**2)
        lb = np.zeros((n, 1))
        ub = 2 * np.ones((n, 1))
        y0 = 0
        xmin = normalize_bounds(np.ones((n, 1)), lb, ub).T[0]
        fname = 'Rosenbrock'

    elif fun_arg == 11:    # 3D Hartman 3
        lb = np.zeros((3, 1))
        ub = np.ones((3, 1))
        alpha = np.array([1,1.2,3.0,3.2])
        A = np.array([[3, 10, 30],
                      [0.1, 10, 35],
                      [3, 10, 30],
                      [0.1, 10, 35]])
        P = 1e-4*np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1091, 8732, 5547],
                           [381 , 5743, 8828]])
        fun = lambda x: -np.dot(alpha, np.exp(-np.diag(np.dot(A, (np.tile(x, 4) - P.T)**2))))
        y0 = -3.86278
        xmin = np.array([0.114614, 0.555649, 0.852547])
        fname = 'Hartman 3'

    elif fun_arg == 12:  # 6D Hartman 6
        lb = np.zeros((6, 1))
        ub = np.ones((6, 1))
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])
        # Notice, take -ln(-y)
        fun = lambda x: -np.log(-(-np.dot(alpha, np.exp(-np.diag(np.dot(A, (np.tile(x, 4) - P.T)**2))))))
        y0 = -np.log(-(-3.32237))
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        fname = 'Hartman 6'

    elif fun_arg == 13:  # 12D Toy test: Quadratic, 1 main direction
        a = np.arange(1, 13) / 12
        c = 0.01
        b = np.hstack((50 * np.ones(2), np.tile(c, 10)))
        lb = np.ones((n, 1))
        ub = np.zeros((n, 1))
        fun = lambda x: np.sum(b*(x.T[0] - a) ** 2)

    elif fun_arg == 14:  # 12D Steiner 2 test function
        fun = lambda x: steiner2(x)[0]
        f, xmin = steiner2(np.ones(12))
        y0 = 16.703838
        fname = 'Steiner2'
        lb = -1 * np.ones((12, 1))
        ub = 6 * np.ones((12, 1))
        xmin = normalize_bounds(xmin.reshape(-1, 1), lb, ub).T[0]

    elif fun_arg == 15:  # 10D, schwefel 1D + flat quadratic 9D:
        fun = lambda x: - sum(np.multiply(500 * x[0], np.sin(np.sqrt(abs(500 * x[0]))))) / 250 + np.dot(0.001 * np.arange(2, 11).reshape(-1, 1).T, x[1:]**2)[0, 0]
        y0 = -1.675936
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        xmin = np.hstack((np.array([0.8419]), np.zeros(9)))
        fname = 'DR First Test'

    elif fun_arg == 16:
        # 12D: VERY BAD TEST PROBLEM. because alg starts with the relative error very small, and failed to improve it.
        fun = lambda x: np.exp(0.2 * x[0])[0] + np.exp(0.2 * x[1])[0] + (10*(x[1] - x[0] ** 2)**2 + (x[0] - 1)**2)[0] + np.sum(0.001 * (x[2:] - 0.1 * np.arange(3, 11).reshape(-1, 1)) ** 2 )
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = 2.341281845987039
        # y0 = -3withhold
        xmin = np.hstack((np.array([0.512, 0.723]), .1 * np.arange(3, 11)))
        fname = 'DR Second Test'

    elif fun_arg == 17:
        # DeltaDOGS + ASM high dimension of active subspace test.
        # 10D test problem, first 2D - Schwefel, the rest 8D are quadratic model.
        fun = lambda x: - sum(np.multiply(500 * x[:2], np.sin(np.sqrt(abs(500 * x[:2]))))) / 250 + .01 * np.dot( np.ones((1, 8)), (x[2:] - 0.1 * np.arange(3, 11).reshape(-1, 1))**2 )[0]
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = -3.3518
        xmin = np.hstack(( .8419 * np.ones(2), np.arange(3, 11) ))
        fname = 'DR Third test'

    elif fun_arg == 18:
        fun = lambda x: np.exp(0.7 * x[0] + 0.3 * x[1])
        lb = np.zeros((10, 1))
        ub = np.ones((10, 1))
        y0 = -3.3518
        xmin = np.hstack(( .8419 * np.ones(2), np.arange(3, 11) ))
        fname = 'DR exp test'

    return fun, lb, ub, y0, xmin, fname


def test_safe_fun(safe_fun_arg, n):
    safe_fun = lb = ub = safe_fun_name = L_safe = []
    if safe_fun_arg == 1:
        # The global minimum is included in the safe region.
        safe_fun = lambda x: np.sin( (x - 0.1) * 5 / 4 * np.pi)[0]
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        safe_fun_name = 'sin wave1'
        # suppose the Lipschitz constant of safe constraint is given as follow
        L_safe = 4
    elif safe_fun_arg == 2:
        # The global minimum is included in the safe region.
        safe_fun = lambda x: np.sin( x * 5 / 4 * np.pi)[0]
        lb = np.zeros((n, 1))
        ub = np.ones((n, 1))
        safe_fun_name = 'sin wave2'
        # suppose the Lipschitz constant of safe constraint is given as follow
        L_safe = 4
    # if safe_fun_arg == 2:

    return safe_fun, lb, ub, safe_fun_name, L_safe


def safe_eval_estimate(xE, y_safe, L_safe, x):
    x = x.reshape(-1, 1)
    val, idx, x_nn = mindis(x, xE)
    safe_estimate = y_safe[idx] - L_safe * val
    # TODO multi safe cons.
    if safe_estimate > 0:
        return True
    else:
        return False
    # ---
    # len(np.where(safe_estimate<0))
    # ---
    # if sum(1 for number in safe_estimate if number < 0) > 0:  # negative safe cons exist, fail
    #     return False
    # else:
    #     return True


def steiner2(x):
    x = x.reshape(-1, 1)
    abar = np.array([5.5, -1.0])
    a = np.array([[0, 2],
                  [2, 3],
                  [3, -1],
                  [4, -0.5],
                  [5, 2],
                  [6, 2]])
    p = np.array([2, 1, 1, 5, 1, 1])
    ptilde = np.array([1, 1, 2, 3, 2])
    m = 6
    f = np.sqrt(x[0]**2 + x[6]**2) + np.sqrt((abar[0] - x[5])**2 + (abar[1] - x[11])**2)
    xx = x.reshape(2, 6).T
    f += np.dot(p, np.sqrt(np.sum((a - xx)**2, axis=1)).reshape(-1, 1))
    xxu = np.copy(xx[:-1, :])
    xxl = np.copy(xx[1:, :])
    f += np.dot(ptilde, np.sqrt(np.sum((xxu-xxl)**2, axis=1)).reshape(-1, 1))
    #### target value and xmin
    xmin = np.zeros(12)
    xmin[0] = (a[0, 0] + a[1, 0]) / 3
    xmin[6] = (a[0, 1] + a[1, 1]) / 3
    for i in range(1, 5):
        xmin[i] = (xmin[i-1] + a[i, 0] + a[i+1, 0]) / 3
        xmin[i+m] = (xmin[i-1+m] + a[i, 1] + a[i+1, 1]) / 3
    xmin[m-1] = (xmin[m-2] + a[m-1, 0] + abar[0]) / 3
    xmin[-1] = (xmin[-2] + a[m-1, 1] + abar[1]) / 3
    return f, xmin


def fun_eval(fun, lb, ub, x):
    x = x.reshape(-1, 1)
    x_phy = physical_bounds(x, lb, ub)
    y = fun(x_phy)
    return y


def random_initial(n, m, Nm, Acons, bcons, xU):
    xE = np.empty(shape=[n, 0])
    while xE.shape[1] < m:
        temp = np.random.rand(n, 1)
        temp = np.round(temp * Nm) / Nm
        dis1, _, _ = mindis(temp, xU)
        if dis1 > 1e-6 and np.dot(Acons, temp) - bcons >= 0:
            if xE.shape[1] == 0:
                xE = np.hstack(( xE, temp ))
            else:
                dis2, _, _ = mindis(temp, xE)
                if dis2 > 1e-6:
                    xE = np.hstack(( xE, temp ))
        else:
            continue
    return xE

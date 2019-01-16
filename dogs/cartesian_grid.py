#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:43:14 2017

@author: mousse
"""
import  numpy   as np
from    dogs    import Utils

'''
 cartesian_grid.py file contains the active constraints and inactive constraints idea from DeltaDOGS(Omega_Z)
 
'''
############################### Cartesian Lattice functions ######################


def add_sup(xE, xU, ind_min):
    '''
    To avoid duplicate values in support points for Delaunay Triangulation.
    :param xE: Evaluated points.
    :param xU: Support points.
    :param ind_min: The minimum point's index in xE.
    return: Combination of unique elements of xE and xU and the index of the minimum yp.
    '''
    xmin = xE[:, ind_min]
    xs = np.hstack((xE, xU))
    # Construct the concatenate of xE and xU and return the array that every column is unique
    x_unique = xs[:, 0].reshape(-1, 1)
    for x in xs.T:
        dis, _, _ = Utils.mindis(x.reshape(-1, 1), x_unique)
        if dis > 1e-5:
            x_unique = np.hstack(( x_unique, x.reshape(-1, 1) ))
    # Find the minimum point's index: ind_min
    _, ind_min_new, _ = Utils.mindis(xmin.reshape(-1, 1), x_unique)
    return x_unique, ind_min_new


def ismember(A, B):
    # Assume that A and B are both vectors
    n = A.shape[0]
    I = np.array([])
    if n == 0:
        I = np.hstack((I, 0))
        return I
    else:
        for i in A:
            if i in B:
                I = np.hstack((I, 1))
            else:
                I = np.hstack((I, 0))
        return I


def points_neighbers_find(x, xE, xU, Bin, Ain):
    '''
    This function aims for checking whether it's activated iteration or inactivated.
    If activated: perform function evaluation.
    Else: add the point to support points.
    :param x: Minimizer of continuous search function.
    :param xE: Evaluated points.
    :param xU: Support points.
    :return: x, xE is unchanged. 
                If success == 1: active constraint, evaluate x.
                Else: Add x to xU.
    '''
    x = x.reshape(-1, 1)
    x1 = Utils.mindis(x, np.concatenate((xE, xU), axis=1))[2].reshape(-1, 1)
    active_cons = []
    b = Bin - np.dot(Ain, x)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons.append(i + 1)
    active_cons = np.array(active_cons)

    active_cons1 = []
    b = Bin - np.dot(Ain, x1)
    for i in range(len(b)):
        if b[i][0] < 1e-3:
            active_cons1.append(i + 1)
    active_cons1 = np.array(active_cons1)
    # Explain the following two criterias，both are actived constraints:
    # The first means that x is an interior point.
    # The second means that x and x1 have exactly the same constraints.
    if len(active_cons) == 0 or abs(min(ismember(active_cons, active_cons1)) - 1.0) < 1e-6:
        newadd = 1
        success = 1
        if xU.shape[1] != 0 and Utils.mindis(x, xU)[0] == 0:
            newadd = 0  # Point x Already exists in support points xU, x should be evaluated.
    else:
        success = 0
        newadd = 0
        xU = np.hstack((xU, x))
    return x, xE, xU, success, newadd


def check_activated(x, xE, xU, Bin, Ain):
    # modified for Lambda Delta DOGS
    # inactive step completed
    # Add the new point to the set
    # xE: evaluation points.
    # xU: unevaluated points.
    xs, _ = add_sup(xE, xU, 1)
    # Find closest point to x
    del_general, index, x1 = Utils.mindis(x, xs)
    # Calculate the active constraints at x and x1
    ind = np.where(np.dot(Ain, x) - Bin > -1e-4 )[0]
    ind1 = np.where(np.dot(Ain, x1) - Bin > -1e-4)[0]
    if len(ind) == 0 or min(ismember(ind, ind1)) == 1:
        label = 1
    else:
        label = 0
    return label

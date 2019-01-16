from optimize import snopta, SNOPT_options
import numpy           as np
from scipy.spatial import Delaunay
from dogs import Utils
from dogs import interpolation
import scipy.io        as io
import os
import inspect

'''
 constantK_snopt.py file contains functions used for DeltaDOGS(Lambda) algorithm.
 Using the package optimize (SNOPT) provided by Prof. Philip Gill and Dr. Elizabeth Wong, UCSD.

 This is a script of DeltaDOGS(Lambda) dealing with linear constraints problem which is solved using SNOPT. 
 Notice that this scripy inplements the snopta function. (Beginner friendly)

 The adaptive-K continuous search function has the form:
 Sc(x) = P(x) - K * e(x):

     Sc(x):     constant-K continuous search function;
     P(x):      Interpolation function: 
                    For AlphaDOGS: regressionparameterization because the function evaluation contains noise;
                    For DeltaDOGS: interpolationparameterization;
     e(x):      The uncertainty function constructed based on Delaunay triangulation.

 Function contained:
     tringulation_search_bound:   Search for the minimizer of continuous search function over all the Delaunay simplices 
                                  over the entire domain.
     Adoptive_K_search:           Search over a specific simplex.
     AdaptiveK_search_cost:       Calculate the value of continuous search function.


 LOG Dec. 4, 2018:   Function snopta still violates the constraints!
 
 LOG Dec. 4, 2018:   Put the actual constant into function bounds Flow and Fupp, do not include constant inside 
                        function evaluation F(x).

 LOG Dec. 15, 2018:  The linear derivative A can not be all zero elements. Will cause error.
                     Fixed by adding a redundant constraint: sum x_i, i=1...n. 

 LOG Dec. 16, 2018:  The 1D bounds of x should be defined by xlow and xupp, 
                        do not include them in F and linear derivative A.

 LOG Dec. 18, 2018:  Happened in DeltaDOGS with asm, retransformation, 1D case.
                     The 2D active subspace - DeltaDOGS with SNOPT shows error message:
                        SNOPTA EXIT  10 -- the problem appears to be infeasible
                        SNOPTA INFO  14 -- linear infeasibilities minimized
                     Fixed by introducing new bounds on x variable based on Delaunay simplex.

 LOG Jan. 10, 2019:  Happened in DeltaDOGS safe learning constantk
                        SNOPTA EXIT -- 40  Terminated after numerical difficulties
                        SNOPTA INFO -- 43  cannot satisfy the general constraints
                     In safe learning case, this error is arised by the safe lipschitz estimate g(x') - L * ||x - x'||.
                     Since we are using 2 norm here, the gradient of the safe lipschitz estimate will be 
                                        - L * ( x - x' ) / || x - x' ||
                     The denominator is zero, which caused the objective function exceed the range inf = 1e+20.
                     
                     
 LOG Jan. 11, 2019: Happened in DeltaDOGS safe learning adaptivek                    
                        SNOPTA EXIT  50 -- error in the user-supplied functions
                        SNOPTA INFO  52 -- incorrect constraint derivatives
                    
'''
##################################  Constant K search SNOPT ###################################


def triangulation_search_bound_snopt(inter_par, xi, K, ind_min, y_safe, L_safe):
    # reddir is a vector
    inf = 1e+20
    n = xi.shape[0]  # The dimension of the reduced model.
    xE = inter_par.xi
    # 0: Build up the Delaunay triangulation based on reduced subspace.
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        tri = Delaunay(xi.T, qhull_options=options).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]
    # Sc contains the continuous search function value of the center of each Delaunay simplex

    # 1: Identify the minimizer of adaptive K continuous search function
    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        if R2 < inf:
            # initialize with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            exist = unevaluated_vertices_identification(xE, xi[:, tri[ii, :]])
            if exist == 0:
                Sc[ii] = interpolation.interpolate_val(x, inter_par) - K * (R2 - np.linalg.norm(x - xc) ** 2)
            else:
                val, idx, x_nn = Utils.mindis(x, xE)
                Sc[ii] = interpolation.interpolate_val(x, inter_par) - K * val ** 2

            # discrete min
            # val, idx, vertex = Utils.mindis(x, xE)
            # c = 0.1
            # e = (val + c) ** (1/2) - c ** (1/2)
            # Sc[ii] = interpolation.interpolate_val(x, inter_par) - K * e

            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf


    # Global one, the minimum of Sc has the minimum value of all circumcenters.
    ind = np.argmin(Sc)
    xm, ym = constantk_search_snopt_min(xi[:, tri[ind, :]], inter_par, K, y_safe, L_safe)

    # Local one
    ind = np.argmin(Scl)
    xml, yml = constantk_search_snopt_min(xi[:, tri[ind, :]], inter_par, K, y_safe, L_safe)
    if yml < ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
        result = 'local'
    else:
        result = 'glob'
    return xm, ym, result


def unevaluated_vertices_identification(xE, simplex):
    exist = 0
    N = simplex.shape[1]
    for i in range(N):
        vertice = simplex[:, i].reshape(-1, 1)
        val, idx, x_nn = Utils.mindis(vertice, xE)
        if val == 0:  # vertice exists in evaluated point set
            pass
        else:
            exist = 1
            break
    return exist
#############   Continuous search function Minimization   ############


def constantk_search_snopt_min(simplex, inter_par, K, y_safe, L_safe):
    '''
    The function F is composed as:  1st        - objective
                                    2nd to nth - simplex bounds
                                    n+1 th ..  - safe constraints
    :param x0        :     The mass-center of delaunay simplex
    :param inter_par :
    :param xc        :
    :param R2:
    :param y0:
    :param K0:
    :param A_simplex:
    :param b_simplex:
    :param lb_simplex:
    :param ub_simplex:
    :param y_safe:
    :param L_safe:
    :return:
    '''
    xE = inter_par.xi
    n = xE.shape[0]

    # Determine if the boundary corner exists in simplex, if boundary corner detected:
    # e(x) = || x - x' ||^2_2
    # else, e(x) is the regular uncertainty function.
    exist = unevaluated_vertices_identification(xE, simplex)

    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------
    # simplex = xi[:, tri[ind, :]]
    # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------

    # Find the minimizer of the search fucntion in a simplex using SNOPT package.
    R2, xc = Utils.circhyp(simplex, n)
    # x is the center of this simplex
    x = np.dot(simplex, np.ones([n + 1, 1]) / (n + 1))

    # First find minimizer xr on reduced model, then find the 2D point corresponding to xr. Constrained optm.
    A_simplex, b_simplex = Utils.search_simplex_bounds(simplex)
    lb_simplex = np.min(simplex, axis=1)
    ub_simplex = np.max(simplex, axis=1)

    inf = 1.0e+20

    m = n + 1  # The number of constraints which is determined by the number of simplex boundaries.
    assert m == A_simplex.shape[0], 'The No. of simplex constraints is wrong'

    # TODO: multiple safe constraints in future.
    # nF: The number of problem functions in F(x), including the objective function, linear and nonlinear constraints.
    # ObjRow indicates the number of objective row in F(x).
    ObjRow = 1

    # solve for constrained minimization of safe learning within each open ball of the vertices of simplex.
    # Then choose the one with the minimum continuous function value.
    x_solver = np.empty(shape=[n, 0])
    y_solver = []

    for i in range(n + 1):

        vertex = simplex[:, i].reshape(-1, 1)
        # First find the y_safe[vertex]:
        val, idx, x_nn = Utils.mindis(vertex, xE)
        if val > 1e-10:
            # This vertex is a boundary corner point. No safe-guarantee, we do not optimize around support points.
            continue
        else:
            # TODO: multiple safe constraints in future.
            safe_bounds = y_safe[idx]

            if n > 1:
                # The first function in F(x) is the objective function, the rest are m simplex constraints.
                # The last part of functions in F(x) is the safe constraints.
                # In high dimension, A_simplex make sure that linear_derivative_A won't be all zero.
                nF   = 1 + m + 1  # the last 1 is the safe constraint.
                Flow = np.hstack((-inf, b_simplex.T[0], -safe_bounds))
                Fupp = inf * np.ones(nF)

                # The lower and upper bounds of variables x.
                xlow = np.copy(lb_simplex)
                xupp = np.copy(ub_simplex)

                # For the nonlinear components, enter any nonzero value in G to indicate the location
                # of the nonlinear derivatives (in this case, 2).

                # A must be properly defined with the correct derivative values.
                linear_derivative_A    = np.vstack((np.zeros((1, n)), A_simplex, np.zeros((1, n))))
                nonlinear_derivative_G = np.vstack((2 * np.ones((1, n)), np.zeros((m, n)), 2 * np.ones((1, n))))

            else:

                # For 1D problem, the simplex constraint is defined in x bounds.
                # TODO multiple safe cons.
                # 2 = 1 obj + 1 safe con. Plus one redundant constraint to make matrix A suitable.
                nF = 2 + 1
                Flow = np.array([-inf, -safe_bounds, -inf])
                Fupp = np.array([ inf, inf        ,  inf])
                xlow = np.min(simplex) * np.ones(n)
                xupp = np.max(simplex) * np.ones(n)

                linear_derivative_A    = np.vstack((np.zeros((1, n)), np.zeros((1, n)), np.ones((1, n)) ))
                nonlinear_derivative_G = np.vstack(( 2 * np.ones((2, n)), np.zeros((1, n)) ))

            x0 = x.T[0]

            # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------
            # cd dogs
            # -------  ADD THE FOLLOWING LINE WHEN DEBUGGING --------

            save_opt_for_snopt_ck(n, nF, inter_par, xc, R2, K, A_simplex, L_safe, vertex, exist)

            # Since adaptiveK using ( p(x) - f0 ) / e(x), the objective function is nonlinear.
            # The constraints are generated by simplex bounds, all linear.

            options = SNOPT_options()
            options.setOption('Infinite bound', inf)
            options.setOption('Verify level', 3)
            options.setOption('Verbose', False)
            options.setOption('Print level', -1)
            options.setOption('Scale option', 2)
            options.setOption('Print frequency', -1)
            options.setOption('Summary', 'No')

            result = snopta(dogsobj, n, nF, x0=x0, name='DeltaDOGS_snopt', xlow=xlow, xupp=xupp, Flow=Flow, Fupp=Fupp,
                            ObjRow=ObjRow, A=linear_derivative_A, G=nonlinear_derivative_G, options=options)

            x_solver = np.hstack((x_solver, result.x.reshape(-1, 1)))
            y_solver.append(result.objective)

    y_solver = np.array(y_solver)
    y = np.min(y_solver)
    x = x_solver[:, np.argmin(y_solver)].reshape(-1, 1)

    return x, y


def folder_path():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    folder = current_path[:-5]  # -5 comes from the length of '/dogs'
    return folder


def save_opt_for_snopt_ck(n, nF, inter_par, xc, R2, K, A_simplex, L_safe, vertex, exist):
    var_opt = {}
    folder  = folder_path()
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w']      = inter_par.w
        var_opt['inter_par_v']      = inter_par.v
        var_opt['inter_par_xi']     = inter_par.xi
    var_opt['n']  = n
    var_opt['nF'] = nF
    var_opt['xc'] = xc
    var_opt['R2'] = R2
    var_opt['K']  = K
    var_opt['A']  = A_simplex
    var_opt['L_safe'] = L_safe
    var_opt['vertex'] = vertex
    var_opt['exist']  = exist
    io.savemat(folder + "/opt_info_ck.mat", var_opt)
    return


def constantk_search_cost_snopt(x):
    x = x.reshape(-1, 1)
    folder = folder_path()
    var_opt = io.loadmat(folder + "/opt_info_ck.mat")

    n  = var_opt['n'][0, 0]
    xc = var_opt['xc']
    R2 = var_opt['R2'][0, 0]
    K  = var_opt['K'][0, 0]
    nF = var_opt['nF'][0, 0]
    A  = var_opt['A']
    L_safe = var_opt['L_safe'][0, 0]
    vertex = var_opt['vertex']
    exist  = var_opt['exist'][0, 0]

    # Initialize the output F and G.
    F = np.zeros(nF)

    method = var_opt['inter_par_method'][0]
    inter_par = interpolation.Inter_par(method=method)
    inter_par.w = var_opt['inter_par_w']
    inter_par.v = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']

    p = interpolation.interpolate_val(x, inter_par)
    gp = interpolation.interpolate_grad(x, inter_par)

    if exist == 0:
        e  = R2 - np.linalg.norm(x - xc) ** 2
        ge = - 2 * (x - xc)
    else:  # unevaluated boundary corner detected.
        e  = (np.dot((x - vertex).T, x - vertex))
        ge = 2 * (x - vertex)

    # discrete min
    # c = 0.1
    # norm_ = np.linalg.norm(x - vertex)
    # norm_ = ( 1e-10 if norm_ < 1e-10 else norm_ )
    # e = (norm_ + c) ** (1/2) - c ** (1/2)
    # ge = (1/2) * (norm_ + c) ** (-1/2) * (x - vertex) / norm_

    F[0] = p - K * e  # K0 = 0 because only at 1st iteration we only have 1 function evaluation.
    # F[1] = - L_safe * np.linalg.norm(x - vertex)
    norm2_difference = np.sqrt(np.dot((x - vertex).T, x - vertex))
    norm2_difference = ( 1e-10 if norm2_difference < 1e-10 else norm2_difference )
    DM = gp - K * ge

    if n == 1:
        # TODO multiple safe con.
        # next line is the safe con
        F[1] = - L_safe * norm2_difference
        # next line is the redundant row.
        F[2] = np.sum(x)

    else:
        # nD data has n+1 simplex bounds.
        F[1:-1] = (np.dot(A, x)).T[0]  # broadcast input array from (3,1) into shape (3).
        F[-1]   = - L_safe * norm2_difference

    G = np.hstack((DM.flatten(), (- L_safe * (x - vertex) / norm2_difference).flatten() ))

    return F, G


def dogsobj(status, x, needF, F, needG, G):
    # G is the nonlinear part of the Jacobian
    F, G = constantk_search_cost_snopt(x)
    return status, F, G


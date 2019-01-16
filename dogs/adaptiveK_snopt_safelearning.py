from        optimize        import snopta, SNOPT_options
import      numpy           as np
from        scipy.spatial   import Delaunay
from        dogs            import Utils
from        dogs            import interpolation
import      scipy.io        as io
import      os
import      inspect
'''
 adaptiveK_snopt.py file contains functions used for DeltaDOGS(Lambda) algorithm.
 Using the package optimize (SNOPT) provided by Prof. Philip Gill and Dr. Elizabeth Wong, UCSD.

 This is a script of DeltaDOGS(Lambda) dealing with linear constraints problem which is solved using SNOPT. 
 Notice that this scripy inplements the snopta function. (Beginner friendly)
 
 The adaptive-K continuous search function has the form:
 Sc(x) = (P(x) - y0) / K*e(x):

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
 
 LOG Dec. 16, 2018:  The 1D bounds of x should be defined by xlow and xupp, 
                        do not include them in F and linear derivative A.
                                               
 LOG Dec. 18, 2018:  The 2D active subspace - DeltaDOGS with SNOPT shows error message:
                        SNOPTA EXIT  10 -- the problem appears to be infeasible
                        SNOPTA INFO  14 -- linear infeasibilities minimized
                     Fixed by introducing new bounds on x variable based on Delaunay simplex.
'''
##################################  adaptive K search SNOPT ###################################


def triangulation_search_bound_snopt(inter_par, xi, y0, ind_min, y_safe, L_safe):
    # reddir is a vector
    inf = 1e+20
    n = xi.shape[0]  # The dimension of the reduced model.

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
            Sc[ii] = (interpolation.interpolate_val(x, inter_par) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    if np.min(Sc) < 0:
        func = 'p'
        # The minimum of Sc is negative, minimize p(x) instead.
        Scp = np.zeros(tri.shape[0])
        for ii in range(tri.shape[0]):
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Scp[ii] = interpolation.interpolate_val(x, inter_par)
        # Globally minimize p(x)
        ind = np.argmin(Scp)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
        xm, ym = adaptiveK_p_snopt_min(x, inter_par, y_safe, L_safe)
        result = 'glob'
    else:
        func = 'sc'
        # Global one, the minimum of Sc has the minimum value of all circumcenters.

        ind = np.argmin(Sc)
        R2, xc = Utils.circhyp(xi[:, tri[ind, :]], n)
        # x is the center of this simplex
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))

        # First find minimizer xr on reduced model, then find the 2D point corresponding to xr. Constrained optm.
        A_simplex, b_simplex = Utils.search_simplex_bounds(xi[:, tri[ind, :]])
        lb_simplex = np.min(xi[:, tri[ind, :]], axis=1)
        ub_simplex = np.max(xi[:, tri[ind, :]], axis=1)

        xm, ym = adaptiveK_search_snopt_min(x, inter_par, xc, R2, y0, K0, A_simplex, b_simplex, lb_simplex, ub_simplex, y_safe, L_safe)
        # Local one

        ind = np.argmin(Scl)
        R2, xc = Utils.circhyp(xi[:, tri[ind, :]], n)
        x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))

        A_simplex, b_simplex = Utils.search_simplex_bounds(xi[:, tri[ind, :]])
        lb_simplex = np.min(xi[:, tri[ind, :]], axis=1)
        ub_simplex = np.max(xi[:, tri[ind, :]], axis=1)
        xml, yml = adaptiveK_search_snopt_min(x, inter_par, xc, R2, y0, K0, A_simplex, b_simplex, lb_simplex, ub_simplex, y_safe, L_safe)
        if yml < ym:
            xm = np.copy(xml)
            ym = np.copy(yml)
            result = 'local'
        else:
            result = 'glob'
    return xm, ym, result, func

#############   Continuous search function Minimization   ############


def adaptiveK_search_snopt_min(x0, inter_par, xc, R2, y0, A_simplex, b_simplex, lb_simplex, ub_simplex, y_safe, L_safe):
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
    # Find the minimizer of the search fucntion in a simplex using SNOPT package.

    inf     = 1.0e+20

    n       = x0.shape[0]  # the number of dimension of the problem.
    m       = n + 1  # The number of constraints which is determined by the number of simplex boundaries.

    # TODO: multiple safe constraints in future.

    # nF: The number of problem functions in F(x),
    # including the objective function, linear and nonlinear constraints.

    # ObjRow indicates the numer of objective row in F(x).
    ObjRow  = 1

    # Upper and lower bounds of functions F(x).
    if n > 1:
        # The first function in F(x) is the objective function, the rest are m constraints.
        nF = m + 1 + 1  # the last 1 is the safe constraint.
        Flow    = np.hstack(( -inf * np.ones(1), b_simplex.T[0], np.zeros(1) ))
        Fupp    = inf * np.ones(nF)

        # The lower and upper bounds of variables x.
        xlow    = np.copy(lb_simplex)
        xupp    = np.copy(ub_simplex)

        # For the nonlinear components, enter any nonzero value in G to indicate the location
        # of the nonlinear derivatives (in this case, 2).

        # A must be properly defined with the correct derivative values.
        linear_derivative_A    = np.vstack((np.zeros((1, n)), A_simplex, np.zeros((1, n)) ))
        nonlinear_derivative_G = np.vstack((2 * np.ones((1, n)), np.zeros((m, n)), 2 * np.ones((1, n)) ))

    else:
        # For 1D problem, only have 1 objective function, the only one constraint is defined in x bounds.
        # nF = 2, cuz 1 obj + safe cons.
        #TODO multiple safe cons.
        # plus one redundant constraint:
        nF = 2

        Flow    = np.hstack((-inf * np.ones(1), np.zeros(1) ))
        Fupp    = inf * np.ones(nF)
        xlow    = np.min(b_simplex) * np.ones(n)
        xupp    = np.max(b_simplex) * np.ones(n)

        linear_derivative_A    = np.vstack(( np.zeros((1, n)), np.zeros((1, n)) ))
        linear_derivative_A[0, 0] = 1e-10
        nonlinear_derivative_G = 2 * np.ones((2, n))

    x0      = x0.T[0]
    save_opt_for_snopt(n, nF, inter_par, xc, R2, y0, A_simplex, y_safe, L_safe)

    # Since adaptiveK using ( p(x) - f0 ) / e(x), the objective function is nonlinear.
    # The constraints are generated by simplex bounds, all linear.



    options = SNOPT_options()
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 3)
    options.setOption('Verbose', False)
    options.setOption('Print level', -1)
    options.setOption('Scale option', 2)
    # options.setOption('Print frequency', -1)
    # options.setOption('Summary', 'Yes')

    result = snopta(dogsobj, n, nF, x0=x0, name='DeltaDOGS_snopt', xlow=xlow, xupp=xupp, Flow=Flow, Fupp=Fupp,
                    ObjRow=ObjRow, A=linear_derivative_A, G=nonlinear_derivative_G, options=options)

    x = result.x
    y = result.objective
    y = (- 1 / y if abs(y) > 1e-10 else 1e+10)

    return x.reshape(-1, 1), y


def folder_path():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    folder = current_path[:-5]      # -5 comes from the length of '/dogs'
    return folder


def save_opt_for_snopt(n, nF, inter_par, xc, R2, y0, A_simplex, y_safe, L_safe):
    var_opt = {}
    folder = folder_path()
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w']      = inter_par.w
        var_opt['inter_par_v']      = inter_par.v
        var_opt['inter_par_xi']     = inter_par.xi
    var_opt['n']  = n
    var_opt['nF'] = nF
    var_opt['xc'] = xc
    var_opt['R2'] = R2
    var_opt['y0'] = y0
    var_opt['A']  = A_simplex
    var_opt['y_safe'] = y_safe
    var_opt['L_safe'] = L_safe
    io.savemat(folder + "/opt_info.mat", var_opt)
    return


def adaptivek_search_cost_snopt(x):
    x       = x.reshape(-1, 1)
    folder  = folder_path()
    var_opt = io.loadmat(folder + "/opt_info.mat")

    n  = var_opt['n'][0, 0]
    xc = var_opt['xc']
    R2 = var_opt['R2'][0, 0]
    y0 = var_opt['y0'][0, 0]
    nF = var_opt['nF'][0, 0]
    A  = var_opt['A']
    y_safe = var_opt['y_safe']
    L_safe = var_opt['L_safe'][0, 0]

    # Initialize the output F and G.
    F  = np.zeros(nF)

    method       = var_opt['inter_par_method'][0]
    inter_par    = interpolation.Inter_par(method=method)
    inter_par.w  = var_opt['inter_par_w']
    inter_par.v  = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']

    p  = interpolation.interpolate_val(x, inter_par)
    e  = R2 - np.linalg.norm(x - xc) ** 2
    gp = interpolation.interpolate_grad(x, inter_par)
    ge = - 2 * (x - xc)

    de = (1e-10 if abs(p-y0) < 1e-10 else p - y0)

    F[0] = - e / de  # K0 = 0 because only at 1st iteration we only have 1 function evaluation.
    val, idx, x_nn = Utils.mindis(x, inter_par.xi)
    F[-1] = y_safe[idx] - L_safe * np.linalg.norm(x - x_nn)

    if n > 1:  # nD data has n+1 simplex bounds.
        F[1:-1] = (np.dot(A, x)).T[0]  # broadcast input array from (3,1) into shape (3).

    DM = - ge / de + e * gp / de ** 2
    G  = np.hstack((DM.flatten(), (- L_safe * (x - x_nn) / np.linalg.norm(x - x_nn)).flatten() ))
    
    return F, G


def dogsobj(status, x, needF, F, needG, G):
    # G is the nonlinear part of the Jacobian
    F, G = adaptivek_search_cost_snopt(x)
    return status, F, G


#############   Interpolant Minimization   ############


def adaptiveK_p_snopt_min(x0, inter_par, Acons, bcons):
    inf = 1.0e+20
    n = x0.shape[0]  # the number of dimension of the problem.
    m = Acons.shape[0]  # the number of linear constraints.
    nF = m + 1
    ObjRow = 1
    Flow = np.hstack(( -inf * np.ones(1), bcons[0] ))
    Fupp = inf * np.ones(nF)

    xlow = np.zeros(n)
    xupp = np.ones(n)

    x0 = x0.T[0]
    save_opt_for_snopt_p(n, nF, inter_par, Acons)

    linear_derivative_A = np.vstack(( np.zeros((1, n)), Acons ))
    nonlinear_derivative_G = np.vstack(( 2 * np.ones((1, n)), np.zeros((m, n)) ))

    options = SNOPT_options()
    options.setOption('Infinite bound', inf)
    options.setOption('Verify level', 3)
    options.setOption('Verbose', False)
    options.setOption('Print level', -1)
    options.setOption('Print frequency', -1)
    options.setOption('Summary', 'No')
    # options.setOption('Print filename', 'DDOGS.out')

    result = snopta(pobj, n, nF, x0=x0, name='DeltaDOGS_snopt', xlow=xlow, xupp=xupp, Flow=Flow, Fupp=Fupp,
                    ObjRow=ObjRow, A=linear_derivative_A, G=nonlinear_derivative_G, options=options)

    x = result.x
    y = result.objective
    if abs(y) > 1e-6:
        y = - 1 / y
    else:
        y = 1e+10
    return x.reshape(-1, 1), y


def save_opt_for_snopt_p(n, nF, inter_par, Acons):
    var_opt = {}
    folder = folder_path()
    if inter_par.method == "NPS":
        var_opt['inter_par_method'] = inter_par.method
        var_opt['inter_par_w']      = inter_par.w
        var_opt['inter_par_v']      = inter_par.v
        var_opt['inter_par_xi']     = inter_par.xi
    var_opt['n']  = n
    var_opt['nF'] = nF
    var_opt['A']  = Acons
    io.savemat(folder + "/opt_info_p.mat", var_opt)
    return


def adaptivek_p_cost_snopt(x):
    x = x.reshape(-1, 1)
    folder = folder_path()
    var_opt = io.loadmat(folder + "/opt_info_p.mat")

    n = var_opt['n'][0, 0]
    nF = var_opt['nF'][0, 0]
    A = var_opt['A']

    # Initialize the output F and G.
    F = np.zeros(nF)

    method = var_opt['inter_par_method'][0]
    inter_par = interpolation.Inter_par(method=method)
    inter_par.w = var_opt['inter_par_w']
    inter_par.v = var_opt['inter_par_v']
    inter_par.xi = var_opt['inter_par_xi']
    gp = interpolation.interpolate_grad(x, inter_par)
    F[0] = interpolation.interpolate_val(x, inter_par)
    F[1:] = (np.dot(A, x)).T[0]  # broadcast input array from (3,1) into shape (3).
    G = gp.flatten()
    return F, G


def pobj(status, x, needF, F, needG, G):
    F = adaptivek_p_cost_snopt(x)[0]
    # G is the nonlinear part of the Jacobian
    G = adaptivek_p_cost_snopt(x)[1]
    return status, F, G

import numpy as np
import scipy.io
from functools import partial
from dogs import interpolation
from dogs import Utils
from dogs import adaptiveK_snopt_safelearning
from dogs import cartesian_grid
from dogs import Dogsplot_sl

float_formatter = lambda x: "%.4f" % x
# only print 4 digits
np.set_printoptions(formatter={'float_kind': float_formatter})

# This script shows the main code of Delta DOGS(Lambda) - using Adaptive K continuous search function

n = 1              # Dimension of input data
fun_arg = 2        # Type of objective function
safe_fun_arg = 1   # Type of safe constraint
iter_max = 100     # Maximum number of iterations based on each mesh
MeshSize = 8       # Represents the number of mesh refinement that algorithm will perform
M = 1              # Number of safe constraints

# plot class
plot_index = 1
original_plot = 0
illustration_plot = 0
interpolation_plot = 0
subplot_plot = 0
store_plot = 1     # The indicator to store ploting results as png.
nff = 1  # Number of experiments

# Algorithm choice:
sc = "AdaptiveK"   # The type of continuous search function
alg_name = 'DDOGS/'

# Calculate the Initial trinagulation points
num_iter = 0       # Represents how many iteration the algorithm goes
Nm = 8             # Initial mesh grid size
L_refine = 0       # Initial refinement sign

# Truth function
fun, lb, ub, y0, xmin, fname = Utils.test_fun(fun_arg, n)
func_eval = partial(Utils.fun_eval, fun, lb, ub)
# safe constraints
safe_fun, lb, ub, safe_name, L_safe = Utils.test_safe_fun(safe_fun_arg, n)
safe_eval                           = partial(Utils.fun_eval, safe_fun, lb, ub)

xU = Utils.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)

regret = np.zeros((nff, iter_max))
estimate = np.zeros((nff, iter_max))
datalength = np.zeros((nff, iter_max))
mesh = np.zeros((nff, iter_max))

for ff in range(nff):
    xE      = np.array([[0.2]])
    num_ini = xE.shape[1]
    yE      = np.zeros(xE.shape[1])  # yE stores the objective function value
    y_safe  = np.zeros(xE.shape[1])  # y_safe stores the value of safe constraints.

    # Calculate the function at initial points
    for ii in range(xE.shape[1]):
        yE[ii]     = func_eval(xE[:, ii])
        y_safe[ii] = safe_eval(xE[:, ii])

    if plot_index:
        plot_parameters = {'store': store_plot, 'safe_cons_type': safe_fun_arg, 'safe_cons': 1}
        plot_class = Dogsplot_sl.plot(plot_parameters, sc, num_ini, ff, fun_arg, alg_name)

    inter_par = interpolation.Inter_par(method="NPS")

    for kk in range(MeshSize):
        for k in range(iter_max):
            num_iter += 1

            K0 = np.ptp(yE, axis=0)  # scale the domain

            [inter_par, yp] = interpolation.interpolateparameterization(xE, yE, inter_par)

            ypmin = np.amin(yp)
            ind_min = np.argmin(yp)

            # Calcuate the unevaluated function:
            yu = np.zeros([1, xU.shape[1]])
            if xU.shape[1] != 0:
                for ii in range(xU.shape[1]):
                    yu[0, ii] = (interpolation.interpolate_val(xU[:, ii], inter_par) - y0) / Utils.mindis(xU[:, ii], xE)[0]
            else:
                yu = np.array([[]])

            xi, ind_min = cartesian_grid.add_sup(xE, xU, ind_min)
            xc, yc, result, func = adaptiveK_snopt_safelearning.triangulation_search_bound_snopt(inter_par, xi, y0, ind_min, y_safe, L_safe)

            xc_grid = np.round(xc * Nm) / Nm
            success = Utils.safe_eval_estimate(xE, y_safe, L_safe, xc_grid)
            xc_eval = ( np.copy(xc_grid) if success == 1 else np.copy(xc) )
            # Dogsplot_sl.safe_continuous_constantK_search_1d_plot(xE, xU, func_eval, safe_eval, L_safe, K, xc_eval, Nm)

            # The following block represents inactivate step: Shahrouz phd thesis P148.
            if Utils.mindis(xc_eval, xE)[0] < 1e-6:  # xc_grid already exists, mesh refine.
                Nm *= 2
                print('===============  MESH Refinement  ===================')
                break
            else:
                xE = np.hstack((xE, xc_eval))
                yE = np.hstack((yE, func_eval(xc_eval)))
                y_safe = np.hstack((y_safe, safe_eval(xc_eval)))
            summary = {'alg': 'ori', 'xc_grid': xc_grid, 'xmin': xmin, 'Nm': Nm, 'y0': y0}
            Dogsplot_sl.print_summary(num_iter, k, kk, xE, yE, summary)

    Alg = {'name': alg_name}
    Dogsplot_sl.save_data(xE, yE, inter_par, ff, Alg)
    Dogsplot_sl.save_results_txt(xE, yE, fname, y0, xmin, Nm, Alg, ff, num_ini)
    Dogsplot_sl.dogs_summary_plot(xE, yE, y0, ff, xmin, fname, alg_name)

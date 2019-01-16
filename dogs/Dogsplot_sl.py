"""
Created on Tue Oct 31 15:45:35 2017

@author: mousse
"""
import      os
import      inspect
import      shutil
import      scipy
import      numpy               as np
import      matplotlib.pyplot   as plt
from        matplotlib.ticker   import PercentFormatter
from        scipy               import io
from        functools           import partial
from        dogs                import Utils
from        dogs                import interpolation
import math
# from        dogs                import DR_adaptiveK
# from        dogs                import discrete_min
'''
Dogsplot.py is implemented to generate results for AlphaDOGS and DeltaDOGS containing the following functions:
    dogs_summary                 :   generate summary plots, 
                                           candidate points 
                                           for each iteration
    plot_alpha_dogs              :   generate plots for AlphaDOGS
    plot_delta_dogs              :   generate plots for DeltaDOGS
    plot_detla_dogs_reduced_dim  :   generate plots for Dimension reduction DeltaDOGS
'''
####################################  Plot Initialization  ####################################


class plot:
    def __init__(self, plot_parameters, sc, num_ini, ff, fun_arg, alg_name):
        '''
        :param plot_index:  Generate plot or not
        :param store_plot:  Store plot or not
        :param sc:          Type of continuous search function, can be 'ConstantK' or 'AdaptiveK'.
        :param num_ini:     Number of initial points
        :param ff:          The times of trials.
        :param fun_arg:     Type of truth function.
        :return:    init_comp : The initialization of contour plots is complete
                    itrp_init_comp : The initialization of interpolation is complete
                    type:   The type of continuous search function.

        '''

        self.store_plot = plot_parameters['store']
        self.type = sc
        self.num_ini = num_ini
        self.ff = ff
        self.fun_arg = fun_arg
        self.num_safe_cons = plot_parameters['safe_cons']
        self.init_comp = 0
        # Make experiments plot folder:
        plot_folder = folder_path(alg_name, ff)
        if os.path.exists(plot_folder):
            # if the folder already exists, delete that folder to restart the experiments.
            shutil.rmtree(plot_folder)
        os.makedirs(plot_folder)


############################################ Plot ############################################
def print_summary(num_iter, k, kk, xE, yE, summary, mesh_refine=0, nm=0):
    n = xE.shape[0]
    Nm = summary['Nm']
    xmin = summary['xmin']
    y0 = summary['y0']
    mindis = format(np.linalg.norm(xmin - xE[:, np.argmin(yE)]), '.4f')
    curdis = format(np.linalg.norm(xmin - xE[:, -1]), '.4f')
    Pos_Reltv_error = str(np.round(np.linalg.norm(xmin - xE[:, np.argmin(yE)])/np.linalg.norm(xmin)*100, decimals=4))+'%'
    Val_Reltv_error = str(np.round((np.min(yE)-y0)/np.abs(y0)*100, decimals=4)) + '%'
    iteration_name = ('Mesh refine' if mesh_refine == 1 else 'Exploration')
    print('============================   ', iteration_name, '   ============================')
    if summary['alg'] == "DR":
        xr = summary['xr']
        w = summary['w'] 
        for i in range(w.shape[0]):
            w[i, 0] = format(w[i, 0], '.4f')
        
        print('%5s' % 'Iter', '%4s' % 'k', '%4s' % 'kk', '%4s' % 'Nm', '%10s' % 'feval_min', '%10s' % 'fevalmin_dis',
              '%10s' % 'cur_f', '%10s' % 'curdis', '%15s' % 'Pos_Reltv_Error', '%15s' % 'Val_Reltv_Error')
        
        print('%5s' % num_iter, '%4s' % k, '%4s' % kk, '%4s' % Nm, '%10s' % format(min(yE), '.4f'),
              '%10s' % mindis, '%10s' % format(yE[-1], '.4f'), '%10s' % curdis, '%15s' % Pos_Reltv_error,
              '%15s' % Val_Reltv_error)
        print('%10s' % '1DSearch', '%10s' % 'RD_Mesh')  # , '%20s' % 'ASM')
        print('%10s' % xr.T[0], '%10s' % nm)  # , '%20s' % w.T[0])
    else:
        print('%5s' % 'Iter', '%4s' % 'k', '%4s'%'kk', '%4s' % 'Nm', '%10s' % 'feval_min', '%10s' % 'fevalmin_dis',
              '%10s' % 'curdis', '%10s' % 'cur_f', '%15s' % 'Pos_Reltv_Error', '%15s' % 'Val_Reltv_Error')
        print('%5s' % num_iter, '%4s' % k, '%4s' % kk, '%4s' % Nm, '%10s' % format(min(yE), '.4f'),
              '%10s' % mindis, '%10s'%curdis, '%10s' % format(yE[-1], '.4f'), '%15s' % Pos_Reltv_error,
              '%15s' % Val_Reltv_error)
    if mesh_refine == 0:
        print('%10s' % 'Best x = ', '%10s' % np.round(xE[:, np.argmin(yE)], decimals=5))
        print('%10s' % 'Current x = ', '%10s' % np.round(xE[:, -1], decimals=5))
    print('====================================================================')
    return 


def dogs_summary_plot(xE, yE, y0, ff, xmin, fun_name, alg_name):
    '''
    This function generates the summary information of DeltaDOGS optimization
    :param yE:  The function values evaluated at each iteration
    :param y0:  The target minimum of objective function.
    :param folder: Identify the folder we want to save plots. "DDOGS" or "DimRed".
    :param xmin: The global minimizer of test function, usually presented in row vector form.
    :param ff:  The number of trial.
    '''
    n = xE.shape[0]
    plot_folder = folder_path(alg_name, ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    xmin = np.copy(xmin.reshape(-1, 1))
    N = yE.shape[0]  # number of iteration
    yE_best = np.zeros(N)
    yE_reltv_error = np.zeros(N)
    for i in range(N):
        yE_best[i] = min(yE[:i+1])
        yE_reltv_error[i] = (np.min(yE[:i+1]) - y0) / np.abs(y0) * 100
    # Plot the function value of candidate point for each iteration
    fig, ax1 = plt.subplots()
    plt.grid()
    # The x-axis is the function count, and the y-axis is the smallest value DELTA-DOGS had found.
    # plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + ': The function value of candidate point', y=1.05)
    ax1.plot(np.arange(N) + 1, yE_best, label='Function value of Candidate point', c='b')
    ax1.plot(np.arange(N) + 1, y0*np.ones(N), label='Global Minimum', c='r')
    ax1.set_ylabel('Function value', color='b')
    ax1.tick_params('y', colors='b')
    plt.xlabel('Iteration number')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Plot the relative error on the right twin-axis.
    ax2 = ax1.twinx()
    ax2.plot(np.arange(N) + 1, yE_reltv_error, 'g--', label=r'Relative Error=$\frac{f_{min}-f_{0}}{|f_{0}|}$')
    ax2.set_ylabel('Relative Error', color='g')

    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params('y', colors='g')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    # Save the plot
    plt.savefig(plot_folder + "/Candidate_point.png", format='png', dpi=1000)
    plt.close(fig)
    ####################   Plot the distance of candidate x to xmin of each iteration  ##################
    fig2 = plt.figure()
    plt.grid()
    # plt.title(alg_name[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + r': The distance of candidate point to $x_{min}$', y=1.05)
    xE_dis = np.zeros(N)
    for i in range(N):
        index = np.argmin(yE[:i+1])
        xE_dis[i] = np.linalg.norm(xE[:, index].reshape(-1,1) - xmin)
    plt.plot(np.arange(N) + 1, xE_dis, label="Distance with global minimizer")
    plt.ylabel('Distance value')
    plt.xlabel('Iteration number')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.savefig(plot_folder + "/Distance.png", format='png', dpi=1000)
    plt.close(fig2)
    # fig2, ax3 = plt.subplots()
    # plt.grid()
    # plt.title(Alg[:-1] + r'$\Delta$-DOGS' + ' ' + str(n) + 'D ' + fun_name + r': The distance of candidate point to $x_{min}$', y=1.05)
    # xE_dis = np.zeros(N)
    # for i in range(N):
    #     index = np.argmin(yE[:i+1])
    #     xE_dis[i] = np.linalg.norm(xE[:, index].reshape(-1,1) - xmin)
    # ax3.plot(np.arange(N) + 1, xE_dis, c='b', label="Distance with global minimizer")
    # ax3.set_ylabel('Function value', color='b')
    # ax3.tick_params('y', colors='b')
    #
    # plt.xlabel('Iteration number')
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    #
    # ax4 = ax3.twinx()
    # ax4.plot(np.arange(1, yE.shape[0]+1), np.ones(yE.shape[0]), 'g', label='Dimension of Active Subspace')
    # ax4.set_ylabel('Dimension', color='g')
    # ax4.tick_params('y', colors='g')
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    #
    # plt.savefig(plot_folder + "/Distance.png", format='png', dpi=1000)
    # plt.close(fig2)
    return 


def save_data(xE, yE, inter_par, ff, Alg):
    name = Alg['name']
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + name + str(ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    dr_data = {}
    dr_data['xE'] = xE
    dr_data['yE'] = yE
    dr_data['inter_par_method'] = inter_par.method
    dr_data['inter_par_w'] = inter_par.w
    dr_data['inter_par_v'] = inter_par.v
    dr_data['inter_par_xi'] = inter_par.xi
    io.savemat(plot_folder + '/data.mat', dr_data)
    return


def save_results_txt(xE, yE, fun_name, y0, xmin, Nm, Alg, ff, num_ini):
    BestError = (np.min(yE) - y0) / np.linalg.norm(y0)
    Error = (yE - y0) / np.linalg.norm(y0)
    if np.min(Error) > 0.01:
        # relative error > 0.01, optimization performance not good.
        idx = '1% accuracy failed'
    else:
        idx = np.min(np.where(Error-0.01 < 0)[0]) - num_ini
    name = Alg['name']
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + name + str(ff)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    with open(plot_folder + '/DminOptmResults.txt', 'w') as f:
        f.write('=====  ' + name[:-1] + ': General information report  ' + str(xE.shape[0]) + 'D' + ' ' + fun_name +
                '=====' + '\n')
        f.write('%40s %10s' % ('Number of Dimensions = ', str(xE.shape[0])) + '\n')
        f.write('%40s %10s' % ('Total Number of Function evaluations = ', str(xE.shape[1])) + '\n')
        f.write('%40s %10s' % ('Actual Error when stopped = ', str(np.round(BestError, decimals=6)*100)+'%') + '\n')
        # f.write('%40s %10s' % ('Best minimizer when stopped = ', str))
        f.write('%40s %10s' % ('Mesh size when stopped = ', str(Nm)) + '\n')
        f.write('%40s %10s' % ('Evaluations Required for 1% accuracy = ', str(idx)) + '\n')
    return
############################    Test function calculator    ################################


def function_eval(X, Y, fun_arg):
    n = 2
    if fun_arg == 1:  # GOLDSTEIN-PRICE FUNCTION
        Z = (1+(X+Y+1)**2*(19-4*X+3*X**2-14*Y+6*X*Y+3*Y**2))*\
            (30+(2*X-3*Y)**2*(18-32*X+12*X**2+48*Y-36*X*Y+27*Y**2))
        y0 = 3
        xmin = np.array([0.5, 0.25])

    elif fun_arg == 2:  # Schwefel
        Z = (-np.multiply(X, np.sin(np.sqrt(abs(500 * X)))) - np.multiply(Y, np.sin(np.sqrt(abs(500 * Y))))) / 2
        y0 = - 1.6759 * n
        xmin = 0.8419 * np.ones((n))

    elif fun_arg == 5:  # Schwefel + Quadratic
        Z = - np.multiply(X / 2, np.sin(np.sqrt(abs(500 * X)))) + 10 * (Y - 0.92) ** 2
        y0 = -10
        xmin = np.array([0.89536, 0.94188])

    elif fun_arg == 6:  # Griewank
        Z = 1 + 1 / 4 * ((X - 0.67) ** 2 + (Y - 0.21) ** 2) - np.cos(X) * np.cos(Y / np.sqrt(2))
        y0 = 0.08026
        xmin = np.array([0.21875, 0.09375])

    elif fun_arg == 7:  # Shubert
        s1 = 0
        s2 = 0
        for i in range(5):
            s1 += (i + 1) * np.cos((i + 2) * (X - 0.45) + (i + 1))
            s2 += (i + 1) * np.cos((i + 2) * (Y - 0.45) + (i + 1))
        Z = s1 * s2
        y0 = -32.7533
        xmin = np.array([0.78125, 0.25])

    elif fun_arg == 8:
        a = 1
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        Z = a*(Y - b*X**2 + c*X - r)**2 + s*(1-t)*np.cos(X) + s
        lb = np.array([-5, 0])
        ub = np.array([10, 15])
        y0 = 0.397887
        xmin = np.array([0.5427728, 0.1516667])  # True minimizer np.array([np.pi, 2.275])
        # 3 minimizer
        xmin2 = np.array([0.123893, 0.8183333])  # xmin2 = np.array([-np.pi, 12.275])
        xmin3 = np.array([0.9616520, 0.165])  # xmin3 = np.array([9.42478, 2.475])

    elif fun_arg == 10:
        x = y = np.linspace(-0.05, 4.05, 1000)
        X, Y = np.meshgrid(x, y)

        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        y0 = 0
        xmin = np.ones(2)

    elif fun_arg == 11:
        Z = 1
        y0 = -3.86278
        xmin = np.array([0.114614, 0.555649, 0.852547])

    elif fun_arg == 12:
        Z = 1
        y0 = -3.32237
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    elif fun_arg == 18:
        Z = np.exp(0.7 * X + 0.3 * Y)
        y0 = -3.32237
        xmin = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    return Z, y0, xmin


def folder_path(alg_name, ff):
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/" + alg_name + str(ff)      # -5 comes from the length of '/dogs'
    return plot_folder


def continuous_search_1d_plot(xE, fun_eval):
    '''
    Given evaluated points set xE, and the objective function. Plot the
    interpolation, uncertainty function and continuous search function.
    :param xE:          evaluated points set xE
    :param fun_eval:    obj function
    :return:
    '''
    N = xE.shape[1]
    yE = np.zeros(N)
    for i in range(N):
        yE[i] = fun_eval(xE[:, i])
    inter_par = interpolation.Inter_par(method='NPS')
    inter_par, _ = interpolation.interpolateparameterization(xE, yE, inter_par)

    x      = np.linspace(0, 1, 1000)
    y, yp  = np.zeros(x.shape), np.zeros(x.shape)

    for i in range(x.shape[0]):
        y[i]  = fun_eval(x[i])
        yp[i] = interpolation.interpolate_val(x[i], inter_par)

    xi = np.copy(xE)
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)

    xe = np.copy(xi)
    xe_plot = np.zeros((tri.shape[0], 2000))
    e_plot = np.zeros((tri.shape[0], 2000))
    sc_plot = np.zeros((tri.shape[0], 2000))

    n = xE.shape[0]
    K = 3
    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])
        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], 2000)
        temp_Sc = np.zeros(len(x_))
        temp_e = np.zeros(len(x_))
        p = np.zeros(len(x_))
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x_)):
            p[jj] = interpolation.interpolate_val(x_[jj], inter_par)
            temp_e[jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            temp_Sc[jj] = p[jj] - K * temp_e[jj]

        e_plot[ii, :] = temp_e
        xe_plot[ii, :] = x_
        sc_plot[ii, :] = temp_Sc

    sc_min = np.min(sc_plot, axis=1)
    index_r = np.argmin(sc_min)
    index_c = np.argmin(sc_plot[index_r, :])
    sc_min_x = xe_plot[index_r, index_c]
    sc_min = min(np.min(sc_plot, axis=1))

    plt.figure()
    plt.plot(x, y, c='k')
    plt.plot(x, yp, c='b')
    for i in range(len(tri)):
        plt.plot(xe_plot[i, :], sc_plot[i, :], c='r')
        plt.plot(xe_plot[i, :], 3 * e_plot[i, :] - 2.3, c='g')
    plt.scatter(xE, yE, c='b', marker='s')
    plt.scatter(sc_min_x, sc_min, c='r', marker='s')
    plt.ylim(-2.5, 2)
    # plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()
    return


def safe_continuous_constantK_search_1d_plot(xE, xU, fun_eval, safe_eval, L_safe, K, xc_min, Nm):
    '''
    Given evaluated points set xE, and the objective function. Plot the
    interpolation, uncertainty function and continuous search function.
    :param xE:
    :param xU:
    :param fun_eval:
    :param safe_eval:
    :param L_safe:
    :param K:
    :param xc_min:
    :param Nm:
    :return:
    '''
    N = xE.shape[1]
    yE = np.zeros(N)
    for i in range(N):
        yE[i] = fun_eval(xE[:, i])
    inter_par = interpolation.Inter_par(method='NPS')
    inter_par, _ = interpolation.interpolateparameterization(xE, yE, inter_par)

    x      = np.linspace(0, 1, 1000)
    y, yp  = np.zeros(x.shape), np.zeros(x.shape)

    for i in range(x.shape[0]):
        y[i]  = fun_eval(x[i])
        yp[i] = interpolation.interpolate_val(x[i], inter_par)

    xi = np.hstack(( xE, xU))
    sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
    tri = np.zeros((xi.shape[1] - 1, 2))
    tri[:, 0] = sx[:xi.shape[1] - 1]
    tri[:, 1] = sx[1:]
    tri = tri.astype(np.int32)

    xe_plot = np.zeros((tri.shape[0], 2000))
    e_plot = np.zeros((tri.shape[0], 2000))
    sc_plot = np.zeros((tri.shape[0], 2000))

    n = xE.shape[0]
    for ii in range(len(tri)):
        temp_x = np.copy(xi[:, tri[ii, :]])
        simplex = xi[:, tri[ii, :]]

        # Determine if the boundary corner exists or not in simplex
        exist = 0
        for i in range(simplex.shape[1]):
            vertice = simplex[:, i].reshape(-1, 1)
            val, _, _ = Utils.mindis(vertice, xE)
            if val == 0:
                pass
            else:
                exist = 1
                break

        x_ = np.linspace(temp_x[0, 0], temp_x[0, 1], 2000)
        temp_Sc = np.zeros(len(x_))
        temp_e = np.zeros(len(x_))
        p = np.zeros(len(x_))
        R2, xc = Utils.circhyp(xi[:, tri[ii, :]], n)
        for jj in range(len(x_)):
            p[jj] = interpolation.interpolate_val(x_[jj], inter_par)

            if exist == 0:
                temp_e[jj] = (R2 - np.linalg.norm(x_[jj] - xc) ** 2)
            else:
                val, _, _  = Utils.mindis(x_[jj], xE)
                temp_e[jj] = val ** (2)
            # val, idx, vertex = Utils.mindis(x_[jj], xE)
            # c = 0.1
            # temp_e[jj] = (val + c) ** (1/2) - c ** (1/2)
            temp_Sc[jj] = p[jj] - K * temp_e[jj]

        e_plot[ii, :] = temp_e
        xe_plot[ii, :] = x_
        sc_plot[ii, :] = temp_Sc

    # The minimizer of continuous search must be subject to safe constraints.
    # sc_min = np.min(sc_plot, axis=1)
    # index_r = np.argmin(sc_min)
    # index_c = np.argmin(sc_plot[index_r, :])
    # sc_min_x = xe_plot[index_r, index_c]
    # sc_min = min(np.min(sc_plot, axis=1))

    safe_plot = {}
    ## plot the safe region
    for ii in range(xE.shape[1]):
        y_safe = safe_eval(xE[:, ii])

        safe_index = []
        y_safe_plot = []
        safe_eval_lip = lambda x: y_safe - L_safe * np.sqrt(np.dot((x - xE[:, ii]).T, x - xE[:, ii]))
        for i in range(x.shape[0]):
            safe_val = safe_eval_lip(x[i])
            y_safe_plot.append(safe_val[0])
            if safe_val > 0:
                safe_index.append(i)
        name = str(ii)
        safe_plot[name] = [safe_index, y_safe_plot]

    # ==================  First plot =================
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    # plot the essentials for DeltaDOGS
    plt.plot(x, y, c='k')
    plt.plot(x, yp, c='b')

    for i in range(len(tri)):

        if i == 0 or i == len(tri) - 1:
            amplify_factor = 3
        else:
            amplify_factor = 50

        plt.plot(xe_plot[i, :], sc_plot[i, :], c='r')
        plt.plot(xe_plot[i, :], amplify_factor * e_plot[i, :] - 5.5, c='g')
    plt.scatter(xE, yE, c='b', marker='s')

    yc_min = sc_plot.flat[np.abs(xe_plot - xc_min).argmin()]
    plt.scatter(xc_min, yc_min, c='r', marker='^')

    # plot the safe region in cyan
    xlow, xupp   = safe_region_plot(x, safe_plot)
    y_vertical   = np.linspace(-2, 2, 100)
    xlow_y_vertical = xlow * np.ones(100)
    xupp_y_vertical = xupp * np.ones(100)
    plt.plot(xlow_y_vertical, y_vertical, color='cyan', linestyle='--')
    plt.plot(xupp_y_vertical, y_vertical, color='cyan', linestyle='--')

    plt.ylim(-6.5, 3.5)
    plt.gca().axes.get_yaxis().set_visible(False)

    # ==================  Second plot =================
    plt.subplot(2, 1, 2)

    y_safe_all = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y_safe_all[i] = safe_eval(x[i])
    plt.plot(x, y_safe_all)
    zero_indicator = np.zeros(x.shape)
    plt.plot(x, zero_indicator, c='k')

    x_scatter = np.hstack((xE, xc_min))
    y_scatter = np.zeros(x_scatter.shape[1])
    for i in range(x_scatter.shape[1]):
        ind = np.argmin(np.abs(x_scatter[:, i] - x))
        y_scatter[i] = y_safe_all[ind]

    plt.scatter(x_scatter[:, :-1][0], y_scatter[:-1], c='b', marker='s')
    plt.scatter(x_scatter[:, -1], y_scatter[-1], c='r', marker='^')

    # plot the safe region
    xlow, xupp = safe_region_plot(x, safe_plot)
    low_idx = np.argmin(np.abs(xlow - x))
    upp_idx = np.argmin(np.abs(xupp - x))
    ylow_vertical = np.linspace(y_safe_all[low_idx], 2, 100)
    yupp_vertical = np.linspace(y_safe_all[upp_idx], 2, 100)
    xlow_y_vertical = xlow * np.ones(100)
    xupp_y_vertical = xupp * np.ones(100)
    plt.plot(xlow_y_vertical, ylow_vertical, color='cyan', linestyle='--')
    plt.plot(xupp_y_vertical, yupp_vertical, color='cyan', linestyle='--')
    plt.ylim(-1, 2.2)
    plt.gca().axes.get_yaxis().set_visible(False)

    # plt.show()
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    plot_folder = current_path[:-5] + "/plot/DDOGS/0"

    num_iter = xE.shape[1] - 1 + math.log(Nm/8, 2)
    plt.savefig(plot_folder + '/pic' + str(int(num_iter)) +'.png', format='png', dpi=250)
    plt.close(fig)
    return


def safe_region_plot(x, safe_plot):
    '''
    plot the safe region with lipschitz constant of each
    :param x:
    :param safe_plot:
    :return:
    '''
    inf = 1e+20
    N = len(safe_plot.keys())
    xlow = inf
    xupp = -inf
    for i in range(N):
        name  = str(i)
        index  = safe_plot[name][0]
        x_safe = x[index]
        y_indicator = 2 * np.ones(x_safe.shape[0])
        line = plt.plot(x_safe, y_indicator)
        plt.setp(line, color='cyan', linewidth=3)
        # It is possible that x is close to the boundary of safe region, no more points is detected as safe near this x.
        if len(x_safe) > 0:
            xlow = min(xlow, np.min(x_safe))
            xupp = max(xupp, np.max(x_safe))
        else:
            pass
    return xlow, xupp

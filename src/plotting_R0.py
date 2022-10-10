""" Plotting related routines """

from matplotlib.pyplot import title, tight_layout, savefig, close, show
from matplotlib.pyplot import figure, xlabel, xlim, ylim
from matplotlib.pyplot import ylabel, plot, scatter, hlines
from matplotlib.pyplot import rcParams
from numpy import diff, argsort
from numpy import min, max
from settings import true_pars


def plot_ML_est(like_str, save_mode,
                fig_path, plot_data, true_params=False):
    '''Plot results from ML estimation'''

    Sf, _, data_m, j, like, min_pars, _, Sftrue, optim = plot_data

    figure(figsize=(9, 6))
    rcParams.update({'font.size': 16})
    like_lat = '$\ell_{%s}(\\theta)$' % (like_str)

    if true_params is True:
        title('From true params \n %s objective' % like_lat)
    else:
        title('Sim %s \n %s objective' % (j, like_lat))

    data_p = -diff(Sf)
    data_p_true = -diff(Sftrue)
    ylabel('Daily New Cases')
    if true_pars['gamma']>1/10:  # we used 1/14 for ODE and 1/7 for Gillespie
        ylim(0, 700)
    else:
        ylim(0, 400)  

    xlim(0, len(data_p))
    xlabel('Time [arb. units] \n $\ell_{%s}(\\theta^{*})=$ %6.2e' %
           (like_str, like) +
           '   $R0^{*}=$ %4.2f   $k^{*}=$%6.4f' %
           (min_pars['R0'], min_pars['k']) +
           '   $n^{*}=$ %3.1f   $\\gamma^{*}=$%6.4f' %
           (min_pars['n'], min_pars['gamma']))
    plot(data_m)
    plot(data_p)
    plot(data_p_true)
    tight_layout()
    if save_mode is True:
        savefig(fig_path + '%03d_fit%s.png' % (j, optim), dpi=150)
        close()
    else:
        show()


def plot_CI(param_name, save_mode, fig_path, data, CIlist, crit, min_ll, k, seedval):
    '''Plot likelihood profile with confidence interval'''
    
    if param_name == 'R0':
        res_idx = 0
        x_label = '$R_0$'
        y_label = '$\ell(R_0)$'
    elif param_name == 'k':
        res_idx = 1
        x_label = '$k$'
        y_label = '$\ell(k)$'
    elif param_name == 'n':
        res_idx = 2
        x_label = '$n$'
        y_label = '$\ell(n)$'
    elif param_name == 'gamma':
        res_idx = 3
        x_label = '$\gamma$'
        y_label = '$\ell(\gamma)$'

    else:
        print('Unknown parameter', flush=True)
        return -1

    # Sort the data by increasing value of the parameter
    sorted_idx = argsort(data[:, res_idx])

    figure(figsize=(9, 6))
    rcParams.update({'font.size': 16})
    plot(data[sorted_idx, res_idx], - (data[sorted_idx, 4] - min_ll), color='blue')
    scatter(data[sorted_idx, res_idx], - (data[sorted_idx, 4] - min_ll),
            color='blue', s=10)
    hlines(-crit/2, min(data[:, res_idx]), max(data[:, res_idx]), linestyle='--')
    hlines(-crit/2, min(CIlist), max(CIlist), color='orange', alpha=0.9)
    xlabel(x_label)
    ylabel(y_label)
    tight_layout()
    if save_mode is True:
        savefig(fig_path + '%s_CI_%g_%d.png' % (param_name, k, seedval), dpi=150)
        close()
    else:
        show()

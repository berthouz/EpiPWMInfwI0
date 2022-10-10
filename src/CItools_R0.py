""" Confidence interval related routines """

from scipy.stats import chi2
from numpy import zeros, delete, array
from scipy.optimize import fmin
from funcs_R0 import cost_func_given_param


def xopt_2_dict_given_param(xopt, param_name, param_val):
    '''Specify dictionary of parameters given that
       parameter param_name wasn't involved in inference'''

    if param_name == 'I0':
        return {'I0': param_val,
                'R0': xopt[0],
                'k': xopt[1],
                'n': xopt[2],
                'gamma': xopt[3]}
    elif param_name == 'R0':
        return {'I0': xopt[0],
                'R0': param_val,
                'k': xopt[1],
                'n': xopt[2],
                'gamma': xopt[3]}
    elif param_name == 'k':
        return {'I0': xopt[0],
                'R0': xopt[1],
                'k': param_val,
                'n': xopt[2],
                'gamma': xopt[3]}
    elif param_name == 'n':
        return {'I0': xopt[0],
                'R0': xopt[1],
                'k': xopt[2],
                'n': param_val,
                'gamma': xopt[3]}
    elif param_name == 'gamma':
        return {'I0': xopt[0],
                'R0': xopt[1],
                'k': xopt[2],
                'n': xopt[3],
                'gamma': param_val}
    else:
        print('Specified parameter does not exist', flush=True)
        return -1


def find_MLE_given_param(init, fixed_pars, param_name, param_val):
    '''Returns a dictionary containing MLE
       This routines differs from find_MLE in that inference
       is done given param_val of parameter named param_name
       init: we use inferred parameters as single initial cond
       fixed_pars: usual set but to be extended with known par'''

    if param_name == 'I0':
        init = delete(init, 0)
    if param_name == 'R0':
        init = delete(init, 1)
    elif param_name == 'k':
        init = delete(init, 2)
    elif param_name == 'n':
        init = delete(init, 3)
    elif param_name == 'gamma':
        init = delete(init, 4)
    else:
        print('Specified parameter does not exist', flush=True)
        return -1

    fixed_pars += (param_name, param_val)

    xopt = fmin(func=cost_func_given_param, args=fixed_pars,
                maxiter=1e3,
                x0=init,
                xtol=1e-7, ftol=1e-8)
    final_nll = cost_func_given_param(xopt, *fixed_pars)
    min_pars = xopt_2_dict_given_param(xopt, param_name, param_val)

    return min_pars, final_nll


def find_CI(param_name, mle_pars, mle_ll_val, param_tick, nb_ticks,
            data_m, like_str, tmax, tcount, N):
    '''Procedure to determine confidence interval for param param_name

    Likelihood profile method. We calculate the 99% profile-likelihood 
    based confidence interval given by the two values of the parameter
    at which the curve intersects the horizontal line drawn at -$\chi^2/2$

    Inputs:
        param_name: 'R0' or 'k' or 'gamma' or 'n'
        mle_pars: MLE parameters
        mle_ll_val: likelihood value from MLE
        param_tick: the search step
        nb_ticks: the number of ticks
        data_m: the data
        like_str: which likelihood
        tmax: for simulations
        tcount: for simulations
        N: size of system

    Ouputs:
        mydata: array of ML estimates + likelihood
    '''

    crit = chi2.isf(0.01, 1)  # 99% intervals (1-0.99=0.01)
    CIres = zeros([nb_ticks, 6])  # 5 MLE pars + 1 ll value
    CIlist = []

    like_ci = mle_ll_val  # based on min likelihood
    fixed_pars = (like_str, data_m, N, tmax, tcount)
    param_val = mle_pars[param_name]

    init = array([mle_pars['I0'], mle_pars['R0'], mle_pars['k'],
                  mle_pars['n'], mle_pars['gamma']])

    CIres[0, :] = array([mle_pars['I0'], mle_pars['R0'], mle_pars['k'],
                         mle_pars['n'], mle_pars['gamma'], mle_ll_val])

    idx = 1
    while - (like_ci - mle_ll_val) >= -crit * 1.5 and idx < (nb_ticks / 2):

        param_val -= param_tick
        xopts, like_ci = find_MLE_given_param(init, fixed_pars,
                                              param_name, param_val)
        CIres[idx, :] = array([xopts['I0'], xopts['R0'], xopts['k'],
                               xopts['n'], xopts['gamma'], like_ci])

        if -(like_ci - mle_ll_val) >= -crit / 2:  # within CI
            CIlist.append(param_val)

        idx += 1

    if idx == (nb_ticks / 2):  # storage space is too small
        print('Likely to have a very flat likelihood profile', flush=True)

    param_val = mle_pars[param_name]  # return to MLE value

    like_ci = mle_ll_val
    while - (like_ci - mle_ll_val) >= -crit * 1.5 and idx < nb_ticks:

        param_val += param_tick
        xopts, like_ci = find_MLE_given_param(init, fixed_pars,
                                              param_name, param_val)
        CIres[idx, :] = array([xopts['I0'], xopts['R0'], xopts['k'],
                               xopts['n'], xopts['gamma'], like_ci])

        if -(like_ci - mle_ll_val) >= -crit / 2:
            CIlist.append(param_val)

        idx += 1

    return CIres[:idx, :], CIlist, crit


""" Infer parameters using ML given noisified ODE realisations

    Input: filename (one realisation per row)
    Output: 
        - Figures (in ../outputs/dir1/dir2/dir3/fits/')
        - Inferred parameters (in ../outputs/dir1/dir2/dir3/arrays/')
        - Initial conditions (in ../outputs/dir1/dir2/dir3/arrays/')
        where dir1 is horizon_%d with %d = tmax
              dir2 is outputs_%s with %s = filename (without extension)
              dir3 is %s with %s = likelihood string (N/P/G)

    Use NIPE_0005 or mNIPE_0005 to launch analysis on cluster: 
        qsub NIPE_0005 (job array)
        qsub -t 350 mNIPE_0005
"""

from scipy.optimize import fmin, minimize
from pathlib import Path
from numpy import zeros, random, save, inf, delete
from numpy import genfromtxt, diff
import sys

from skopt.space import Space, Real
from skopt.sampler import Lhs

from SIR_R0_model import SIR_R0_model
from funcs_R0 import cost_func, likelihood
from plotting_R0 import plot_ML_est

from os.path import splitext

from settings import N, I0, true_pars


def pretty_print_params(pars, preamble=''):
    '''Formatted printing of inferred parameters for easy reading'''
    if isinstance(pars, dict):
        print(preamble+': %2.1f, %f, %3.1f, %4.3f' % (pars['R0'],
                                                      pars['k'],
                                                      pars['n'],
                                                      pars['gamma']))
    else:
        print(preamble+': %2.1f, %f, %3.1f, %4.3f' % (pars[0],
                                                      pars[1],
                                                      pars[2],
                                                      pars[3]))


def get_data(filename, tmax, tcount):
    '''Load data

    Inputs:
        filename
        tmax: max time to be used for inference (should be less than length provided)
        tcount: what was the number of reports used when generating the data
    Outputs:
        data (cropped to tmax)
        tmax
        tcount'''

    data = genfromtxt('../data/' + filename, delimiter=',')
    foldername = splitext(filename)[0]
    return data[:, :tmax], tcount, foldername


def generate_random_initial_conditions(nb_initial_confs):
    space = Space(  # (R0, k, n and gamma in that order)
        [Real(low=0.2, high=10, prior='uniform', transform='identity'),
         Real(low=0.00001, high=0.05, prior='uniform', transform='identity'),
         Real(low=3, high=20, prior='uniform', transform='identity'),
         Real(low=0.001, high=0.1, prior='uniform', transform='identity')])

    # maximin optimized hypercube sampling
    lhs = Lhs(criterion="maximin", iterations=10000)
    inits = lhs.generate(space.dimensions, nb_initial_confs)
    # inits = space.rvs(nb_initial_confs)  # A simpler alternative

    # Impose constraints on the solutions... namely:
    # (a) \tau must be positive (this requires denominator to be positive)
    # (b) ratio \tau / \gamma should not be too large -> denominator >= 1.5
    # Choice of 1.5 is quite arbitrary... Encountered issue with 1.25
    # Alternative choice: tau<0.1  
    to_delete = []  # list of indices to delete
    for i, init in enumerate(inits):
        if ((init[2] - 1) - init[0]) < 1.5:
        # tau = init[2] * init[3] / ((init[2] - 1) - init[0])
        # if (tau < 0) or (tau > 0.1):
            # print('Deleting initial condition no %d' % i)
            to_delete.append(i)
    inits = delete(inits, to_delete, axis=0)

    print('Returning %d valid initial conditions' % len(inits))
    return inits


def set_initial_conditions(random=True, nb_initial_confs=5):
    '''Returns either random initial conditions or true parameters'''
    if random:
        return generate_random_initial_conditions(nb_initial_confs)
    else:
        return [[true_pars['R0'], true_pars['k'],
                 true_pars['n'], true_pars['gamma']]]


def find_MLE(inits, fixed_pars, optim='_NM_'):
    '''Returns a dictionary containing MLE'''

    min_ll = inf
    for i, init in enumerate(inits):
        print('find_MLE: Initial condition', init)
        if optim == '_CG_':
            xopt = \
                minimize(fun=cost_func, x0=init, args=fixed_pars,
                         method='CG',
                         options={'gtol': 1e-06, 'norm': inf, 'eps': 1.5e-08,
                                  'maxiter': None, 'disp': False,
                                  'return_all': False})

            final_nll = cost_func(xopt['x'], *fixed_pars)
        else:  # default
            xopt, fopt, iter, funcalls, warnflag = \
                fmin(func=cost_func, args=fixed_pars,
                     maxiter=5e3,
                     x0=init,
                     xtol=1e-7, ftol=1e-8,
                     full_output=True)
            # print('fopt = ', fopt)
            # print('iter = ', iter)
            # print('funcalls = ', funcalls)
            # print('warnflag', warnflag)
            # print('xopt = ')
            # print(xopt)
            final_nll = cost_func(xopt, *fixed_pars)

        pretty_print_params(init, 'Initial conditions')
        print('likelihood is: %6.2f' % final_nll)
        pretty_print_params(xopt, 'Inferred parameters')

        if final_nll < min_ll:
            min_ll = final_nll
            min_pars = {'R0': xopt[0],
                        'k': xopt[1],
                        'n': xopt[2],
                        'gamma': xopt[3]}

    return min_pars, min_ll


def runall(fname, tmax, tcount, like_str='N', nb_ICs=10,
           save_mode=True, seedval=42, optim='_NM_'):

    # Initialise seed
    random.seed(int(seedval))

    # Get data
    data, tcount, foldername = get_data(fname, int(tmax), int(tcount))

    # Set initial conditions (common to all -- to reduce time)
    inits = set_initial_conditions(True, int(nb_ICs))

    # Solve ODE for true parameters for reference
    SIRepidemic = SIR_R0_model(N, I0, true_pars)
    _, St, _, _ = SIRepidemic.run(tmin=0, tmax=int(tmax), tcount=int(tcount))

    # Decrement seedval by 1 because job arrays only start from 1
    seedval = int(seedval) - 1  # if passed on the command line, seedval will be a string

    print(seedval, flush=True)
    print('Finding MLE', flush=True)

    data_m = data[seedval]  # Incidence data

    # Purely for information
    true_ll = likelihood(like_str,
                         data[seedval], -diff(St),
                         *(N, true_pars['k']))
    print('true_ll = %6.2f' % true_ll)

    # pre-defined fixed arguments (none of these are subject to inference)
    fixed_pars = (like_str, data_m, N, I0, int(tmax), int(tcount))

    # find MLE
    min_pars, min_ll = find_MLE(inits, fixed_pars, optim)

    # run SIR epidemic with MLE parameters
    SIRepidemic = SIR_R0_model(N, I0, min_pars)
    _, Sf, If, _ = SIRepidemic.run(tmin=0, tmax=int(tmax), tcount=int(tcount))

    # plot MLE results
    plot_data = (Sf, If, data_m, seedval, min_ll, min_pars, min_pars['R0'], St, optim)
    if bool(save_mode) is True:
        fig_path = '../outputs/horizon%d/outputs_%s/%s/fits/' % \
            (tmax, foldername, like_str)
        Path(fig_path).mkdir(parents=True, exist_ok=True)
    else:
        fig_path = ''
    plot_ML_est(like_str, bool(save_mode), fig_path, plot_data)

    # ------- STORE CURRENT RESULTS TO ARRAY -------
    metares = [min_pars['R0'], min_pars['k'],
               min_pars['n'], min_pars['gamma'],
               min_ll]

    # ------- SAVE OVERALL RESULTS TO NPY ------
    if bool(save_mode) is True:
        res_path = '../outputs/horizon%d/outputs_%s/%s/arrays/' % \
            (tmax, foldername, like_str)
        Path(res_path).mkdir(parents=True, exist_ok=True)
        save(res_path+'%s%spars.npy' % (seedval, optim), metares)
        save(res_path+'%s%sinits.npy' % (seedval, optim), inits)


if __name__ == '__main__':

    if len(sys.argv) > 1:  # Parameters were passed from command line
        runall(*sys.argv[1:])
    else:
        # params: fname (in directory ../data)
        #         tmax (max time)
        #         tcount (number of reports when generating datalines)
        #         like_str (N/P/G, def='N'),
        #         number of initial conditions (def=10)
        #         save_mode (bool, def=True),
        #         seedval (def=102)
        #         optimisation method (def='NM')
        runall('ODE_with_noise_negbin_0p0005.csv', 150, 201, 'N', 15, True, 42, '_NM_')

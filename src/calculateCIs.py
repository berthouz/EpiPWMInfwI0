""" Calculate summary statistics on the CI for all realisations 

    To use this code, main_R0 must first be used to produce the output files.
    Then the output files need to be combined into a single file all_xx_pars.npy where xx is the value of optim (L45)
    Value of 'optim' on L45 is default but needs to be edited manually if non-default value was used in main_R0
    L33-36 need to be edited manually to reflect values used in main_R0
    File name L44 needs to be edited manually.
    Figures will be placed in ../figures/horizonxx/CIs/ where xx takes value tmax (L33)

    Use CI_0005 or mCI_0005 to launch analysis on cluster: 
        qsub CI_0005 (job array)
        qsub -t 350 mCI_0005

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from settings import N, I0, true_pars
from numpy import random, where, delete, savetxt, mean, std, empty, save
from plotting_R0 import plot_CI
from CItools_R0 import find_CI
from main_R0 import get_data

from os.path import splitext
from pathlib import Path

def runall(seedval):

    seedval = int(seedval) - 1  # Needed because cluster doesn't accept job indices starting at 0. 

    tmax = 150
    tcount = 201
    like_str = 'N'
    k = 0.0005

    true_gamma = true_pars['gamma']
    true_n = true_pars['n']
    true_R0 = true_pars['R0']

    true_pars.update({'k': k})  # To avoid having to edit settings.py separately

    fname = 'ODE_with_noise_negbin_%s.csv' % str(k).replace('.','p')
    optim = '_NM_'
    foldername = splitext(fname)[0]
    res_path = '../outputs/horizon%s/outputs_%s/%s/arrays/' % \
                (tmax, foldername, like_str)
    fig_path = '../figures/horizon%s/CIs/' % tmax
    Path(fig_path).mkdir(parents=True, exist_ok=True)

    # Load results to analyse
    res = np.load(res_path+'%s%spars.npy' % ('all', optim))  # After combining all files from cluster


    # --------------------------------------------------
    #           Exclude outliers: estimated n > N 
    # --------------------------------------------------

    outlierIdx = where(res[:,2]>N)[0]
    if len(outlierIdx)>0:
        print('Excluding %d estimates' % len(outlierIdx))
        res = delete(res, outlierIdx, axis=0)


    CIres = empty((1,3))

    # --------------------------------------------------
    #        Confidence interval for R0 
    # --------------------------------------------------

    # Get data
    data, tcount, foldername = get_data(fname, int(tmax), int(tcount))
    data_m = data[int(seedval)]  # Incidence data

    # Get ML parameters and min_ll
    min_pars = {'R0': res[int(seedval), 0],
                'k': res[int(seedval), 1],
                'n': res[int(seedval), 2],
                'gamma': res[int(seedval), 3]}
    min_ll = res[int(seedval), 4]

    # Find confidence interval for tau
    print('Calculating confidence interval', flush=True)
    R0_tick = 0.005
    nb_ticks = 150  # This value may be insufficient for high values of the dispersion parameter -- increase to 300 for example
    mydata, CIlist, crit = find_CI('R0', min_pars, min_ll, R0_tick, nb_ticks, 
                                data_m, like_str,
                                int(tmax), int(tcount), N, I0)

    plot_CI('R0', True, fig_path, mydata, CIlist, crit, min_ll, true_pars['k'], int(seedval))
    print('CI = %f with max = %f and min = %f' % ((max(CIlist)-min(CIlist)), min(CIlist), max(CIlist)))
    CIres[0,:] = [max(CIlist)-min(CIlist), min(CIlist), max(CIlist)]

    save(fig_path+'CIarray_%d.npy' % int(seedval), CIres)



if __name__ == '__main__':

    if len(sys.argv) > 1:  # Parameters were passed from command line
        runall(*sys.argv[1:])
    else:
        # params: realisation number
        runall(350)

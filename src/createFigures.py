""" Generate the following figures:
    - Unindentifiability curve
    - Histograms of inferred parameters with true value for reference
    - Histogram of estimated R0
    - Pair plots (need to activate on L176)

    To use this code, main_R0 must first be used to produce the output files.
    Then the output files need to be combined (combineOutputs.py) into a single file all_xx_pars.npy where xx is the value of optim (L43)
    
    Input: file name on L42
    Output: figures (in directory ../figures/horizonxx) where xx is value of tmax (L30) 

    Need to manually edit parameters L30-34 to reflect what was used in main_R0.py
    Need to manually edit variable optim (L43) to reflect optimisation method used in main_R0.py
"""

import numpy as np
import matplotlib.pyplot as plt

from settings import N, true_pars
from numpy import random, where, delete, log, arange, savetxt, mean
from plotting_R0 import plot_CI
from CItools_R0 import find_CI
from SIR_R0_model import SIR_R0_model
from main_R0 import get_data, find_MLE

from os.path import splitext
from pathlib import Path

tmax = 100
tcount = 201
like_str = 'N'
seedval = 4
k = 0.0  # No dispersion parameter in Gillespie

true_gamma = true_pars['gamma']
true_n = true_pars['n']
true_R0 = true_pars['R0']

true_pars.update({'k': k})  # To avoid having to edit settings.py separately

fname = 'Gillespie.csv'
optim = '_NM_'
foldername = splitext(fname)[0]
res_path = '../outputs/horizon%d/outputs_%s/%s/arrays/' % \
            (tmax, foldername, like_str)
fig_path = '../figures/horizon%d/' % tmax
Path(fig_path).mkdir(parents=True, exist_ok=True)

# Load results to analyse
res = np.load(res_path+'%s%spars.npy' % ('all', optim))  # After combining all files from cluster


# --------------------------------------------------
#           Exclude outliers: estimated n > N 
# --------------------------------------------------

outlierIdx = where(res[:,3]>N)[0]
if len(outlierIdx)>0:
    print('Excluding %d estimates for poor degree estimate' % len(outlierIdx))
    res = delete(res, outlierIdx, axis=0)

# --------------------------------------------------
#           Exclude outliers: estimated R0 > 10 
#  NB: No real rationale for this ... just cosmetic
# --------------------------------------------------

outlierIdx = where(res[:,1]>10)[0]
if len(outlierIdx)>0:
    print('Excluding %d estimates for poor R0 estimate' % len(outlierIdx))
    res = delete(res, outlierIdx, axis=0)


# --------------------------------------------------
#           Unindentifiability curve
# --------------------------------------------------

# Solve ODE for true parameters to calculate $S_\infty$
SIRepidemic = SIR_R0_model(N, true_pars)
_, Strue, _, _ = SIRepidemic.run(tmin=0, tmax=int(tcount), tcount=int(tcount))
S_inf = Strue[-1] / N  # fraction of susceptible at the end of the epidemic
print('Fraction susceptible $S_\infty$ = %f' % S_inf)

# Calculate the growth factor
# The growth factor is calculated based on the true parameters
true_tau = true_R0 * true_gamma / (true_n - 1 - true_R0)
growth_factor = true_tau * (true_n - 2) - true_gamma

# Set a range of n values and calculate both forms of \tau (Eqs. 19 and 20)
min_n = min(res[:,2])
max_n = 14
n_vec = arange(min_n, max_n, (max_n - min_n) / 10000)
tau_eq_19 = (growth_factor + true_gamma) / (n_vec - 2)
# tau_eq_20 = true_gamma * log(S_inf) / (n_vec * S_inf - log(S_inf) - n_vec)  # closure=1
tau_eq_20 = true_gamma * (S_inf**(1/n_vec)-S_inf**(2/n_vec)) / (S_inf**(2/n_vec)-S_inf)  # closure=n/(n-1)

# Plot
plt.figure(figsize=(9, 6))
plt.rcParams.update({'font.size': 16})

# Calculating ranges of tau parameters (to define plot boundaries)
tau = res[:, 1] * res[:, 4] / (res[:, 3] - 1 - res[:, 1])
plt.scatter(tau, res[:, 3], color='blue', s=10)
plt.plot(tau_eq_19, n_vec, linestyle=':', color='black')
plt.plot(tau_eq_20, n_vec, linestyle='-.', color='green')
plt.plot(true_tau, true_n, 'r*')
plt.xlabel(r'$\tau$')
plt.ylabel('n')

plt.xlim([min(tau)-0.0025,max(tau)+0.0025])
plt.ylim([1, 15])

plt.tight_layout()
plt.savefig(fig_path + 'n_vs_tau_%g_%d.png' % (k, tmax))


# --------------------------------------------------
#        Histogram of estimated R0 values 
# --------------------------------------------------

plt.figure(figsize=(9, 6))
plt.rcParams.update({'font.size': 12})

min_R0 = np.min(res[:,1])
max_R0 = np.max(res[:,1])
print('Min_R0 = %f, max_R0 = %f' % (min_R0, max_R0))

n = plt.hist(res[:, 1], bins=np.arange(min_R0 - 0.5, max_R0 + 0.5, 0.025), density=True)
max_prob_density = np.max(n[0])
plt.xlabel('$R_0$')
plt.ylabel('Probability density')
plt.xlim((min_R0 - 0.5, max_R0 + 0.5))
plt.ylim((0, max_prob_density + 1))
plt.axvline(true_R0, color='red', linestyle='dashed', linewidth=1)

plt.tight_layout()
# plt.show()
plt.savefig(fig_path + 'R0_histogram_%g_%d.png' % (k, tmax))
savetxt(fig_path + 'R0_histogram_%g_%d.csv' % (k, tmax), res[:,0])

print('Mean R0 = %f' % mean(res[:,1]))


# --------------------------------------------------
#    Plotting histograms of all inferred values
# --------------------------------------------------

fig, axs = plt.subplots(1, 5, figsize=(9,6))
plt.rcParams.update({'font.size': 12})

varnames = ['I0', 'R0', 'n', 'gamma', 'k']
orgidx = [0, 1, 3, 4, 2]

for i, varname in enumerate(varnames):
    axs[i].hist(res[:, orgidx[i]], density=True, bins=25)
    figtitle = varname
    axs[i].set_xlabel(varname)

    if varname == 'I0':
        axs[i].set_xlim((min(0, np.min(res[:, orgidx[i]])), max(10, np.max(res[:, orgidx[i]]))))
        axs[i].set_xlabel(r'$I0$')
        # do not plot true_pars['I0'] since there is no true value for Gillespie simulations
    elif varname == 'R0':
        axs[i].set_xlim((min(1.5, np.min(res[:, orgidx[i]])), max(2.5, np.max(res[:, orgidx[i]]))))
        axs[i].set_xlabel(r'$R_0$')
        axs[i].axvline(true_pars['R0'], color='red', linestyle='dashed', linewidth=1)
    elif varname == 'k':
        axs[i].set_xlim((0,0.01))
        axs[i].set_xlim((min(0.0, np.min(res[:, orgidx[i]])), max(0.01, np.max(res[:, orgidx[i]]))))
        # do not plot true_pars['k'] since there is no true value for Gillespie simulations
    elif varname == 'n':
        axs[i].set_xlim((2,11))
        axs[i].set_xlim((min(2, np.min(res[:, orgidx[i]])), max(11, np.max(res[:, orgidx[i]]))))
        axs[i].axvline(true_pars['n'], color='red', linestyle='dashed', linewidth=1)
    else:
        axs[i].set_xlim((0.03,0.1))
        axs[i].set_xlim((min(0.03, np.min(res[:, orgidx[i]])), max(0.1, np.max(res[:, orgidx[i]]))))
        axs[i].set_xlabel(r'$\gamma$')
        axs[i].axvline(true_pars['gamma'], color='red', linestyle='dashed', linewidth=1)
axs[0].set_ylabel('Probability density')


plt.tight_layout()
#plt.show()
plt.savefig(fig_path + 'all_histograms_%g_%d.png' % (k, tmax))
savetxt(fig_path + 'all_histograms_%g_%d.csv' % (k, tmax), res[:, orgidx])

print('Mean values for all parameters: <I0>=%f <R0>=%f <n>=%f <gamma>=%f k=<%f>' % 
      (mean(res[:,0]), mean(res[:,1]), mean(res[:,3]), mean(res[:,4]), mean(res[:,2])))


if 0:
    # --------------------------------------------------
    #                     Pair plots
    # --------------------------------------------------

    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(data=res[:,:5], columns=[r"$I0$", "$R_0$", "k", "n", "$\gamma$"])
    sns.pairplot(df)
    #plt.show()
    plt.savefig(fig_path + 'pairplots_%g_%d.png' % (k, tmax))


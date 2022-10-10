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

from settings import N, I0, true_pars
from numpy import random, where, delete, log, arange, savetxt, mean
from plotting_R0 import plot_CI
from CItools_R0 import find_CI
from SIR_R0_model import SIR_R0_model
from main_R0 import get_data, find_MLE

from os.path import splitext
from pathlib import Path

tmax = 60
tcount = 201
like_str = 'N'
seedval = 350
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
fig_path = '../figures/horizon%s/' % tmax
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


# --------------------------------------------------
#           Unindentifiability curves
# --------------------------------------------------

# Solve ODE for true parameters to calculate $S_\infty$
SIRepidemic = SIR_R0_model(N, I0, true_pars)
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

tau = res[:, 0] * res[:, 3] / (res[:, 2] - 1 - res[:,0])
# tau = res[:, 0] * res[:, 3] / (res[:, 2] - 2)
plt.scatter(tau, res[:, 2], color='blue', s=10)
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

min_R0 = np.min(res[:,0])
max_R0 = np.max(res[:,0])
print('Min_R0 = %f, max_R0 = %f' % (min_R0, max_R0))

n = plt.hist(res[:, 0], bins=np.arange(min_R0 - 0.5, max_R0 + 0.5, 0.025), density=True)
max_prob_density = np.max(n[0])
plt.xlabel('$R_0$')
if (k==0.0005) and (tmax==150):  # first plot gets the vertical labels, the others do not
    plt.ylabel('Probability density')
plt.xlim((min_R0 - 0.5, max_R0 + 0.5))
plt.ylim((0, max_prob_density + 1))
plt.axvline(true_R0, color='red', linestyle='dashed', linewidth=1)

plt.xlim([1,3])
plt.ylim([0,5])
plt.title('$k=%g$' % k)
plt.tight_layout()
# plt.show()
plt.savefig(fig_path + '/R0_histogram_%g_%d.png' % (k, tmax))
savetxt(fig_path + 'R0_histogram_%g_%d.csv' % (k, tmax), res[:,0])

print('Mean R0 = %f' % mean(res[:,0]))

# --------------------------------------------------
#    Plotting histograms of all inferred values
# --------------------------------------------------

fig, axs = plt.subplots(1, 4, figsize=(9,6))
plt.rcParams.update({'font.size': 12})

varnames = ['R0', 'n', 'gamma', 'k']
orgidx = [0, 2, 3, 1]

for i, varname in enumerate(varnames):
    axs[i].hist(res[:, orgidx[i]], density=True, bins=25)
    figtitle = varname
    axs[i].set_xlabel(varname)
    if varname == 'R0':
        axs[i].set_xlim((min(1.5, np.min(res[:, orgidx[i]])), max(2.5, np.max(res[:, orgidx[i]]))))
        axs[i].set_xlabel(r'$R_0$')
        axs[i].axvline(true_pars['R0'], color='red', linestyle='dashed', linewidth=1)
    elif varname == 'k':
        axs[i].set_xlim((0,0.01))
        axs[i].set_xlim((min(0.0, np.min(res[:, orgidx[i]])), max(0.01, np.max(res[:, orgidx[i]]))))
        if true_pars['k']>0.0:
            axs[i].axvline(true_pars['k'], color='red', linestyle='dashed', linewidth=1)
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

print('Mean values for all parameters: <R0>=%f <n>=%f <gamma>=%f k=<%f>' % 
      (mean(res[:,0]), mean(res[:,2]), mean(res[:,3]), mean(res[:,1])))


if 0:
    # --------------------------------------------------
    #                     Pair plots
    # --------------------------------------------------

    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(data=res[:,:4], columns=[r"$R_0$", "k", "n", "$\gamma$"])
    sns.pairplot(df)
    #plt.show()
    plt.savefig(fig_path + 'pairplots_%g_%d.png' % (k, tmax))


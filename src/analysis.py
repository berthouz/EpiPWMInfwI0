""" Generate the following figures:
    - Unindentifiability curve
    - Histograms of inferred parameters with true value for reference
    - Pair plots

    Input: file name on line 15
    Output: figures (in directory ../figures/) 
"""

import numpy as np
import matplotlib.pyplot as plt

from settings import N, I0, true_pars
from numpy import random
from plotting_R0 import plot_CI
from CItools_R0 import find_CI
from main_R0 import get_data, find_MLE

from os.path import splitext
from pathlib import Path

tmax = 150
tcount = 201
like_str = 'N'
seedval = 102
fname = 'ODE_with_noise_negbin.csv'
optim = '_NM_'
foldername = splitext(fname)[0]
res_path = '../outputs/outputs_%s/%s/arrays/' % \
            (foldername, like_str)
fig_path = '../figures/'
Path(fig_path).mkdir(parents=True, exist_ok=True)

# Load results to analyse
res = np.load(res_path+'%s%spars.npy' % (1000, optim))

# --------------------------------------------------
#           Unindentifiability curve
# --------------------------------------------------

true_gamma = true_pars['gamma']
true_n = true_pars['n']
true_R0 = true_pars['R0']

# Calculate tau and growth factor
true_tau = true_R0 * true_gamma / (true_n - 2)
growth_factor = true_tau * (true_n - 2) - true_gamma

# Plot
plt.figure(figsize=(9, 6))
plt.rcParams.update({'font.size': 16})

# Calculating ranges of tau parameters (to define plot boundaries)
tau = res[:, 0] * res[:, 3] / (res[:, 2] - 2)
plt.scatter(tau, res[:, 2], color='blue', s=10)
min_tau = np.min(tau)
max_tau = np.max(tau)
print('Range is %f %f' % (min_tau, max_tau))

tau_vec = np.arange(np.min(tau), max_tau, (max_tau - np.min(tau)) / 100000)
theoretical_n = 2 + (growth_factor + true_gamma) / tau_vec

plt.plot(tau_vec, theoretical_n, linestyle=':', color='black')
plt.plot(true_tau, true_n, 'r*')
plt.xlabel(r'$\tau$')
plt.ylabel('n')

plt.xlim([min_tau-0.0025,max_tau+0.0025])
plt.ylim([-1, 15])

plt.tight_layout()
# plt.show()
plt.savefig('../figures/n_vs_tau_0.0005.png')


# --------------------------------------------------
#        Histogram of estimated R0 values 
# --------------------------------------------------

plt.figure(figsize=(9, 6))
plt.rcParams.update({'font.size': 12})

min_R0 = np.min(res[:,0])
max_R0 = np.max(res[:,0])
print('Min_R0 = %f, max_R0 = %f' % (min_R0, max_R0))

n = plt.hist(res[:, 0], bins=np.arange(min_R0 - 0.5, max_R0 + 0.5, 0.05), density=True)
max_prob_density = np.max(n[0])
plt.xlabel('$R_0$')
plt.ylabel('Probability density')
plt.xlim((min_R0 - 0.5, max_R0 + 0.5))
plt.ylim((0, max_prob_density + 1))
plt.axvline(true_R0, color='red', linestyle='dashed', linewidth=1)

plt.tight_layout()
# plt.show()
plt.savefig('../figures/R0_histogram_0.0005.png')


# --------------------------------------------------
#                     Pair plots
# --------------------------------------------------

import seaborn as sns
import pandas as pd
df = pd.DataFrame(data=res[:,:4], columns=[r"$R_0$", "k", "n", "$\gamma$"])
sns.pairplot(df)
#plt.show()
plt.savefig('../figures/pairplots_0.0005.png')


# --------------------------------------------------
#    Plotting histograms of all inferred values
# --------------------------------------------------

fig, axs = plt.subplots(1, 4, figsize=(9,6))
plt.rcParams.update({'font.size': 12})

varnames = ['R0', 'k', 'n', 'gamma']

for i, varname in enumerate(varnames):
    axs[i].hist(res[:, i], density=True, bins=25)
    figtitle = varname
    axs[i].set_xlabel(varname)
    if varname == 'R0':
        axs[i].set_xlim((min(1.5, np.min(res[:, i])), max(2.5, np.max(res[:, i]))))
        axs[i].set_xlabel(r'$R_0$')
        axs[i].axvline(true_pars['R0'], color='red', linestyle='dashed', linewidth=1)
    elif varname == 'k':
        axs[i].set_xlim((0,0.01))
        axs[i].set_xlim((min(0.0, np.min(res[:, i])), max(0.01, np.max(res[:, i]))))
        axs[i].axvline(true_pars['k'], color='red', linestyle='dashed', linewidth=1)
    elif varname == 'n':
        axs[i].set_xlim((2,11))
        axs[i].set_xlim((min(2, np.min(res[:, i])), max(11, np.max(res[:, i]))))
        axs[i].axvline(true_pars['n'], color='red', linestyle='dashed', linewidth=1)
    else:
        axs[i].set_xlim((0.03,0.1))
        axs[i].set_xlim((min(0.03, np.min(res[:, i])), max(0.1, np.max(res[:, i]))))
        axs[i].set_xlabel(r'$\gamma$')
        axs[i].axvline(true_pars['gamma'], color='red', linestyle='dashed', linewidth=1)
axs[0].set_ylabel('Probability density')

        
plt.tight_layout()
#plt.show()
plt.savefig('../figures/all_histograms_0.0005.png')


# --------------------------------------------------
#        Confidence interval for R0 
# --------------------------------------------------

# Load ICs
inits = np.load(res_path+'%s%sinits.npy' % (1000, optim))

# Initialise seed
random.seed(int(seedval))

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
mydata, CIlist, crit = find_CI('R0', min_pars, min_ll, R0_tick,
                            data_m, like_str,
                            int(tmax), int(tcount), N, I0)

plot_CI('R0', True, '../figures/', mydata, CIlist, crit, min_ll)

'''Default settings

Defines dictionary of parameters which can either involve $\tau$ or $R_0$'''

__author__ = "Luc Berthouze"


topdir = '/Volumes/LocalDataHD/lb203/OneDrive - University of Sussex' \
         '/MySrc/EpiPWMInfwI0/'

N = 10000  # total number of nodes

true_pars = {
    'I0': 1,
    'n': 6.0,
    'gamma': 1/7,
    'k': 0.0005,
    'R0': 2.0
}

true_pars.update({'tau': true_pars.get('R0') * true_pars.get('gamma')
                 / ((true_pars.get('n') - 1) - true_pars.get('R0'))})

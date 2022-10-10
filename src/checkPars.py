""" Visually compare ODEs when using inferred parameters and true parameters

    Input: 
        - R0, k, n, gamma, tcount
    Output: 
        - 3-panel figure (S/I/R) (not saved)
"""

import sys
from numpy import random, diff, array
from SIR_R0_model import SIR_R0_model
from settings import N, I0, true_pars
from matplotlib.pyplot import plot, show, title, legend, subplot, figure, rcParams


def check_pars(*args):

    if len(args) == 5:
        pars = {'R0': float(args[0]),
                'k': float(args[1]),
                'n': float(args[2]),
                'gamma': float(args[3])}
        tcount = int(args[4])
    elif len(args) == 6:
        pars = {'R0': float(args[1]),
                'k': float(args[2]),
                'n': float(args[3]),
                'gamma': float(args[4])}
        I0 = float(args[0])  # Overwriting import from settings
        tcount = int(args[5])
    else:
        print('Cannot process this number of parameters')
        return
    
    
    # Solve ODE for true parameters for reference
    SIRepidemic = SIR_R0_model(N, I0, pars)
    _, St, It, Rt = SIRepidemic.run(tmin=0, tmax=tcount, tcount=tcount)
    
    # Solve ODE for true parameters for reference
    SIRepidemic = SIR_R0_model(N, I0, true_pars)
    _, Strue, Itrue, Rtrue = SIRepidemic.run(tmin=0, tmax=tcount, tcount=tcount)
    S_inf = Strue[-1]/N  # fraction of susceptible at the end of the epidemic
    
    figure(figsize=(18, 6))
    rcParams.update({'font.size': 12})
    
    subplot(1,3,1)
    plot(-diff(St),color='blue')
    plot(-diff(Strue), color='red')
    title('Susceptible')
    legend(['Estimated', 'True'])

    subplot(1,3,2)
    plot(It, color='blue')
    plot(Itrue, color='red')
    title('Infected')
    legend(['Estimated', 'True'])
    
    subplot(1,3,3)
    plot(Rt, color='blue')
    plot(Rtrue, color='red')
    title('Recovered')
    legend(['Estimated', 'True'])

    show()
            

if __name__ == '__main__':

    if len(sys.argv) > 1:  # Parameters were passed from command line
        check_pars(*sys.argv[1:])
    else:
        # params: 
        #         R0
        #         k
        #         n
        #         gamma
        #         nb of days
        check_pars(1.934, 1.758e-03, 3.000, 4.013e-03, 201)

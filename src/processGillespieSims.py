''' Process the Gillespie realisations to make them inference ready

    Input: Gillespie realisations in ./data/DataGillespieSim in .csv format 
    Output: Single data file (./data/Gillespie.csv') containing all realisations

    The Gillespie realisations have random exponential times. This routine produces realisations with integer times.
    Concretely, by taking the number of infectious at largest time prior to integer times.
    For example, if events at t=0.2, 0.5, 0.99 and 1.03, I(day=1) = I(t=0.99). 
'''

from numpy import genfromtxt, reshape, linspace, diff, where, vstack, ceil, any, empty, savetxt
from glob import glob
import matplotlib.pyplot as plt

Tmax = 100
alldataline = empty((0, Tmax))
for f in glob('../data/DataGillespieSim/*.csv'):
    # print('Processing: %s' % f, flush=True)
    data = genfromtxt(f, delimiter=',')
    if len(data)==0:  # This happens if an epidemic died out in which case IZK produced an empty file! 
        print('Cannot process %s' % f)
    else:
        idx = [where(data[:,0]<i+1)[0][-1] for i in range(Tmax)]  # index of highest time prior to integer times (starting from 1)
        resampleddata = data[0,:-1].reshape((1,2))  # First row is maintained (time 0)
        resampleddata = vstack((resampleddata, data[idx,:-1]))  # Times are still float
        resampleddata[:,0] = ceil(resampleddata[:,0])  # Integer times

        # Check for gaps (for information purposes only as it has no bearing on anything)
        if any(resampleddata[:,0]!=linspace(0,Tmax,Tmax+1)):
            print('There are episodes of multiple days without event')

        dataline = resampleddata[:,1]  # We can ignore the times now

        if 0: # Plot only
            plt.plot(-diff(dataline))
            plt.show()
        else: # Produce incidence data
            alldataline = vstack((alldataline, -diff(dataline)))

savetxt('../data/Gillespie.csv', alldataline, delimiter=',', fmt='%d')
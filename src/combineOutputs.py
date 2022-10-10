""" Create single npy file combining output pars files from cluster runs 
    
    - Assumes (and does not check) that all runs are available (from 0 to n -- input parameter)
    - Reads and stores sequentially (makes it easier to match with the fit figures)
    - Name of optimisation scheme used (optional parameter -- default _NM_)
    - Produces output file named: all%spars.npy where %s is optim
"""

import sys
from numpy import load, empty, save

if __name__ == '__main__':

    nb_inputs = len(sys.argv)
    if nb_inputs > 2:  # Parameters were passed from command line
        if nb_inputs==3 or nb_inputs==4:
            n = int(sys.argv[1])
            nb_pars = int(sys.argv[2])
            if nb_inputs==3:
                optim = '_NM_'
            else:
                optim = str(sys.argv[3])
            
            combinedarray = empty((n, nb_pars + 1))  # nb_pars parameters + ll
            for i in range(n):
                pars = load('%d%spars.npy' % (i+1, optim))
                combinedarray[i,:] = pars
            
            save('all%spars.npy' % optim, combinedarray)
        else:
            print('Usage: combineOutputs.py n nb_pars [optim (default _NM_)]', flush=True)
    else:
        print('Usage: combineOutputs.py n nb_pars [optim (default _NM_)]', flush=True)

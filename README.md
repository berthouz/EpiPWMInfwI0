# EpiPWMInfwI0.git
Pair-wise model parameter inference from Gillespie realisations

Code used to perform the ML inference on Gillespie realisations in paper "Approximate likelihoods and limits of information content in data for inference of epidemics on networks" by Kiss et al. The key difference with code from EpiPWMInf is that here we also infer $I_0$ to model the stochasticity in the onset of the exponential growth. It is possible to do without but fit is much poorer.

Folder data contains:
* Folder DataGillespieSim containing 500 realisations (130 of which died out)
* Gillespie.csv: file produced using src/processGillespieSims and containing incidence data (370 realisations) on each (integer) day


Folder src contains the course code:
* NIPE_0005, mNIPE_0005, CI_0005, mCI_0005: sample scripts to launch code on cluster (see headers of main_R0.py and calculateCIs)
* main_R0.py: main routine. Performs inference for one realisation. This routine relies on the following sub-routines:
- SIR_R0_model.py: SIR model reparametrised in terms of R0
- funcs_R0.py: misc routines
- plotting_R0.py: plotting routines
* calculateCIs: calculates confidence intervals for chosen parameter for one realisation. This routine relies on:
- CItools_R0.py: CI-related routines
* settings.py: General parameters
* combinedOutputs.py: Combine output files from cluster job arrays. Needed before running createFigures.py.
* createFigures.py: Produces figures from paper. 
* processGillespieSims.py: Processes the raw Gillespie simulations to make them inference ready (integer day)

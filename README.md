# EpiPWMInf.git
Pair-wise model parameter inference

Code used to perform the ML inference on data obtained directly from the pair-wise mean-field model with negative binomial noise in paper "Approximate likelihoods and limits of information content in data for inference of epidemics on networks" by Kiss et al. 

Folder data contains example files of data from the pair-wise mean-field model with negative binomial noise (dispersion level 0.05, 0.01, 0.005, 0.001, 0.0005)

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
* checkPars.py: Convenience routine to compare estimated parameters with true parameters

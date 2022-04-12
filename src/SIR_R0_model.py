from scipy.integrate import odeint
from numpy import array, linspace, concatenate

'''SIR homogeneous pairwise model (reparametrised for $R_0$)'''
class SIR_R0_model():
    def __init__(self, N, I0, dict_pars):

        self.N = N
        self.I = I0
        self.n = dict_pars['n']
        self.R0 = dict_pars['R0']
        self.gamma = dict_pars['gamma']

        self.S = self.N - self.I
        self.R = 0
        self.SI = self.n * self.S * self.I / self.N
        self.SS = self.n * self.S * self.S / self.N

    def run(self, tmin=0, tmax=200, tcount=201):
        '''Run homogeneous pairwise simulation for tcount steps
           Inputs: simulation length parameters
           Outputs: time series of S, I and R + times'''

        def _dSIR_homogeneous_pairwise_(X, t):
            '''ODE for SIR homogeneous pairwise (07/11/2021)'''
            S, I, SI, SS = X
            tau = self.R0 * self.gamma / ((self.n - 1) - self.R0)
            dSdt = -tau * SI
            dIdt = tau * SI - self.gamma * I
            dSIdt = -self.gamma * SI + \
                tau * ((self.n - 1) * SI * (SS - SI) / (self.n * S) - SI)
            dSSdt = -2 * tau * (self.n - 1) * SI * SS / (self.n * S)
            dX = array([dSdt, dIdt, dSIdt, dSSdt])
            return dX

        def _dSIR_homogeneous_pairwise_old_(X, t):
            '''ODE for SIR homogeneous pairwise'''
            S, I, SI, SS = X
            nm1_over_n = (self.n - 1) / self.n
            gammaR0_over_n = self.gamma * self.R0 / (self.n - 2)
            dSdt = -gammaR0_over_n * SI
            dIdt = gammaR0_over_n * SI - self.gamma * I
            dSIdt = -(gammaR0_over_n + self.gamma) * SI + \
                nm1_over_n * gammaR0_over_n * SI * (SS - SI) / S
            dSSdt = -2 * nm1_over_n * gammaR0_over_n * SI * SS / S
            dX = array([dSdt, dIdt, dSIdt, dSSdt])
            return dX

        X0 = array([self.S, self.I, self.SI, self.SS])  # current values
        steps = linspace(tmin, tcount-1, tcount)
        times = concatenate([[0], steps])
        X = odeint(_dSIR_homogeneous_pairwise_, X0, times)
        S_ts, I_ts, SI_ts, SS_ts = X.T
        R_ts = self.N - S_ts - I_ts
        return times[1:tmax+2], S_ts[1:tmax+2], I_ts[1:tmax+2], R_ts[1:tmax+2]

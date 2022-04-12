from scipy.stats import nbinom, truncnorm
from numpy import sum, log, inf, diff
from SIR_R0_model import SIR_R0_model
from settings import I0


def likelihood(l_str, A, B, *args):
    '''Calculate likelihood for model defined by:
    string l_str,
    A = data,
    B = model'''

    N, k = args

    # ------- GAUSSIAN ---------
    if l_str == 'G':
        # a = (-1 * len(A) / 2) * log(2 * pi) + sum((-1 / 2) * ((A - B) ** 2))
        myclip_a = 0
        myclip_b = inf
        a, b = (myclip_a - B) / k, (myclip_b - B) / k
        # for i in range(len(A)):
        #     print('inputs: ', A[i], a[i], B[i], k, flush=True)
        #     print('truncnorm.logpdf: ', truncnorm.logpdf(A[i], a[i], b[i], B[i], k), flush=True)
        ll = sum(truncnorm.logpdf(A, a, b, B, k))
        return -ll

    # -------- POISSON ---------
    if l_str == 'P':
        a = sum(-B + A * log(B))
        return -1 * a

    # -------- NEGBIN -----------
    if l_str == 'N':
        n = 1 / k
        p = 1 / (1 + k * B)
        return -sum(nbinom.logpmf(A, n, p))



def cost_func(x, *args):
    '''Return loss likelihood given inferred parameters (x)

    Inputs:
        x contains R0, tau, k, n and gamma (estimated parameters)
        args contains:
            likelihood string: 'N'/'P'/'G'
            data
            N
            I0
            tmax
            tcount

    Outputs:
        loss
    '''

    if len(x) != 4:
        print('Unexpected number of parameters in the likelihood. \
               Should be 4 is %d' % len(x), flush=True)
        return inf

    if isinstance(x, dict):
        R0, k, n, gamma = x['R0'], x['k'], x['n'], x['gamma']
    else:
        R0, k, n, gamma = x[0], x[1], x[2], x[3]

    # one or more parameter might go off-bound
    # typically k becoming slightly negative
    if R0 <= 0 or k <= 0 or n <= 2 or gamma < 0:
        return inf

    l_str, data, N, I0, tmax, tcount = args

    cur_pars = {'I0': I0, 'R0': R0, 'k': k, 'n': n, 'gamma': gamma}
    SIRepidemic = SIR_R0_model(N, I0, cur_pars)  # instantiate a model
    t, Sf, If, Rf = SIRepidemic.run(tmin=0, tmax=tmax, tcount=tcount)

    dataf = -diff(Sf)

    # print('Costfunc: with parameters: ', R0, k, n, gamma)
    return likelihood(l_str, data, dataf, *(N, k))


def cost_func_given_param(x, *args):
    '''Return loss likelihood given:
       inferred parameters (x)
       fixed parameters + param_name + param_val (args)
       '''

    if len(x) != 3:
        print('Unexpected number of parameters in the likelihood. \
               Should be 3 is %d' % len(x), flush=True)
        return inf

    l_str, data, N, I0, tmax, tcount, param_name, param_val = args

    if param_name == 'R0':
        R0, k, n, gamma = param_val, x[0], x[1], x[2]
    elif param_name == 'k':
        R0, k, n, gamma = x[0], param_val, x[1], x[2]
    elif param_name == 'n':
        R0, k, n, gamma = x[0], x[1], param_val, x[2]
    elif param_name == 'gamma':
        R0, k, n, gamma = x[0], x[1], x[2], param_val
    else:
        print('Specified parameter does not exist', flush=True)
        return -1

    # one or more parameter might go off-bound
    # typically k becoming slightly negative
    if R0 <= 0 or k <= 0 or n <= 2 or gamma < 0:
        return inf

    cur_pars = {'R0': R0, 'k': k, 'n': n, 'gamma': gamma}
    SIRepidemic = SIR_R0_model(N, I0, cur_pars)  # instantiate a model
    t, Sf, If, Rf = SIRepidemic.run(tmin=0, tmax=tmax, tcount=tcount)

    dataf = -diff(Sf)

    return likelihood(l_str, data, dataf, *(N, k))

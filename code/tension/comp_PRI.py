### Uniaxial tension problem
# Comparison of different methods for managing statistical (sampling)
# uncertainty on a uniaxial tension test.
#
# Compares a number of approaches:
# BV         = basis value
# PI         = plug in
#
# Zachary del Rosario, Mar. 2018

import numpy as np
import matplotlib.pyplot as plt
import time
import pyutil.numeric as ut

from scipy.stats import norm, nct, t, chi2
from scipy.optimize import bisect
from pyutil.plotting import linspecer

plt.style.use('ggplot')
np.random.seed(101)
np.set_printoptions(precision=3)

##################################################
## Script parameters
##################################################
# Full sampling
M     = int(1e3)               # Replications
L_ALL = [10000]                # MC samples
N_ALL = [100, 500, 1000, 5000] # Sample count sweep
# DEBUG COUNTS
# M     = int(1e2)    # Replications
# L_ALL = [100,1000]  # MC samples
# N_ALL = [20,50,100] # Sample count sweep

# Select case
MYCASE = 0

# Design parameters
#                     Rel,  Con,  Pop
PARAM = np.array([[  0.90, 0.95,   0.99],  #  A-basis;  lax reliability
                  [  0.99, 0.95,   0.99],  #  A-basis;  med reliability
                  [1-1e-7, 0.95,   0.99],  #  A-basis;  strict reliability
                  [1-1e-7, 1-1e-7, 0.99]]) #  Z-basis;  strict reliability

# RV parameters
MU_CR  = 600.   * 1e+6 # mean, critical stress     [Pa]
MU_A   = 100.   * 1e+6 # mean, axial force         [N]
TAU_CR = MU_CR * 0.1  # std. dev., critical stress [Pa]
TAU_A  = MU_A  * 0.1  # std. dev., axial force     [N]
# Fixed DV
RADIUS = 1.0           # cylinder radius            [m]

D_LO = 0.02; D_HI = 0.035 # thickness bounds for binary search [m]
DX   = np.sqrt(np.finfo(float).eps)

##################################################
# Helper functions
##################################################
fA_c   = lambda d: np.pi*(2*RADIUS*d + d**2)
fSig_a = lambda d: MU_A  / fA_c(d)
fGam   = lambda d: TAU_A / fA_c(d)

def fK_pc(p,c,n):
    return nct.ppf(c,n-1,-norm.ppf(1-p)*np.sqrt(n)) / np.sqrt(n)

def fZ_score(d,s_c,s):
    return (s_c - fSig_a(d)) / np.sqrt(s**2+fGam(d)**2)

def fRel_fcn(d,s_c,s):
    return norm.cdf(fZ_score(d,s_c,s))

def fRel_eff(d):
    return norm.cdf(fZ_score(d,MU_CR,TAU_CR))

def fD_bv(B,Rel):
    return np.sqrt((TAU_A*norm.ppf(Rel)+MU_A)/np.pi/B+RADIUS) - RADIUS

def fA_star(Rel,mu_cr=MU_CR,tau_cr=TAU_CR,M=0):
    z = norm.ppf(Rel)
    return ((mu_cr-M)*MU_A + np.sqrt(z**2*(mu_cr-M)**2*TAU_A**2 + z**2*MU_A**2*tau_cr**2 \
                        - z**4*tau_cr**2*TAU_A**2)) / ((mu_cr-M)**2 - z**2*tau_cr**2)

def fD_star(Rel,mu_cr=MU_CR,tau_cr=TAU_CR,M=0):
    A_star = fA_star(Rel,mu_cr=mu_cr,tau_cr=tau_cr,M=M)
    return np.sqrt(A_star/np.pi+RADIUS**2) - RADIUS

def g_lim(d,X):
    # Limit state function; g(-) > 0 for success
    # Usage
    #   g = g_lim(d,X)
    # Arguments
    #   d = Design variable
    #   X = Random variables;
    #     = [\sigma_cr, F_a]
    # Returns
    #   g = Limit state value

    return X[0] - X[1]/fA_c(d)

def del_rho(X,T):
    # Gradient factor
    # Usage
    #   D = del_rho(X,T)
    # Arguments
    #   X = Random variables;
    #     = [\sigma_cr, F_a]
    #   T = Parameters;
    #     = [\sigma_cr^0,\sigma^2]
    # Returns
    #   D = Differential;
    #     = [\partial_{\sigma_cr^0}, \sigma^2]

    D = np.zeros((2,X.shape[0]))
    D[0] = (X-T[0])/T[1]
    D[1] = (X-T[0])**2/T[1]**2 - 0.5/T[1]
    return D

def hes_rho(X,T):
    # Hessian factor
    # Usage
    #   H = del_rho(X,T)
    # Arguments
    #   X = Random variables;
    #     = [\sigma_cr, F_a]
    #   T = Parameters;
    #     = [\sigma_cr^0,\sigma^2]
    # Returns
    #   H = Hessian factor

    H = np.zeros((2,2,X.shape[0]))
    H[0,0] = -1/T[1] + (X-T[0])**2/T[1]**2
    H[0,1] = (X-T[0]) + (X-T[0])**3/T[1]**2
    H[1,0] = H[0,1]
    H[1,1] = -2*(X-T[0])**2/T[1]**3 + (X-T[0])**4/T[1]**4
    return H

##################################################
# Design sweep
##################################################
Rel = PARAM[MYCASE][0]
Con = PARAM[MYCASE][1]
Pop = PARAM[MYCASE][2]

beta_cr = norm.ppf( Rel )
zc      = norm.ppf(Con)

d_s = fD_star(Rel)

D_lsm_mc = np.zeros((len(N_ALL),M,len(L_ALL)))
R_lsm_mc = np.zeros((len(N_ALL),M,len(L_ALL)))
M_lsm_mc = np.zeros((len(N_ALL),M,len(L_ALL)))
D_pri_mc = np.zeros((len(N_ALL),M,len(L_ALL)))
R_pri_mc = np.zeros((len(N_ALL),M,len(L_ALL)))
M_pri_mc = np.zeros((len(N_ALL),M,len(L_ALL)))

t0 = time.time()
ut.print_progress(0,len(N_ALL),bar_length=60)
for jnd in range(len(N_ALL)):
    for ind in range(M):
        ## Sampling simulation
        X_dat = np.random.multivariate_normal(mean=[MU_CR,MU_A],
                                              cov=np.diag([TAU_CR**2,TAU_A**2]),
                                              size=N_ALL[jnd]).T
        mu_dat = np.mean(X_dat[0])
        s2_dat = np.var(X_dat[0])
        T_dat  = [mu_dat,s2_dat]

        # Common random numbers
        Z_sim = np.random.multivariate_normal(np.zeros(2),cov=np.eye(2),size=max(L_ALL))
        X_sim = (np.array([mu_dat,MU_A]) \
              + np.dot(Z_sim,np.diag([np.sqrt(s2_dat),TAU_A]))).T

        for lnd in range(len(L_ALL)):
            # Common simulation values
            D  = del_rho(X_sim[0,:L_ALL[lnd]],T_dat)
            H  = hes_rho(X_sim[0,:L_ALL[lnd]],T_dat)
            Sg = np.diag([s2_dat/N_ALL[jnd],2*s2_dat**2/(N_ALL[jnd]-1)])

            ## Monte Carlo PI + estimated MD LSSM
            def obj_lsm(d):
                G   = g_lim(d,X_sim[:,:L_ALL[lnd]])
                h   = np.mean(G)

                # Compute margin
                dD = np.mean(D*(G-h), axis = 1)
                m = zc * np.sqrt(np.dot(dD,np.dot(Sg,dD)))

                Pr_succ_p = np.mean( (G - m) > 0 )

                return Pr_succ_p - Rel

            D_lsm_mc[jnd,ind,lnd] = bisect(obj_lsm, D_LO, D_HI,
                                    xtol = 1e-14, maxiter = int(1e3))
            R_lsm_mc[jnd,ind,lnd] = fRel_eff(D_lsm_mc[jnd,ind,lnd])
            M_lsm_mc[jnd,ind,lnd] = (D_lsm_mc[jnd,ind,lnd]-d_s)/d_s

            ## Monte Carlo PRI approach
            def obj_pri(d):
                # Analysis
                G  = g_lim(d,X_sim[:,:L_ALL[lnd]])
                F  = np.mean(G <= 0)
                dF = np.mean(D*(G <= 0), axis = 1)
                dB = -dF * ut.grad(1-F, norm.ppf)[0]
                # Approximate PRI
                mu_B    = norm.ppf(1-F)
                sig2_B  = np.dot(dB, np.dot(Sg, dB))
                beta_tl = mu_B / np.sqrt(1+sig2_B)

                return beta_tl - beta_cr

            D_pri_mc[jnd,ind,lnd] = bisect(obj_pri, D_LO, D_HI,
                                    xtol = 1e-14, maxiter = int(1e3))
            R_pri_mc[jnd,ind,lnd] = fRel_eff(D_pri_mc[jnd,ind,lnd])
            M_pri_mc[jnd,ind,lnd] = (D_pri_mc[jnd,ind,lnd]-d_s)/d_s

    ut.print_progress(jnd+1,len(N_ALL),bar_length=60)
t1 = time.time()

##################################################
# Post-process
##################################################
D_lsm_mc.sort(axis=1); D_pri_mc.sort(axis=1)
R_lsm_mc.sort(axis=1); R_pri_mc.sort(axis=1)
M_lsm_mc.sort(axis=1); M_pri_mc.sort(axis=1)

D_lsm_mc_mu = np.mean(D_lsm_mc,axis=1); D_pri_mc_mu = np.mean(D_pri_mc,axis=1)
R_lsm_mc_mu = np.mean(R_lsm_mc,axis=1); R_pri_mc_mu = np.mean(R_pri_mc,axis=1)
M_lsm_mc_mu = np.mean(M_lsm_mc,axis=1); M_pri_mc_mu = np.mean(M_pri_mc,axis=1)

D_lsm_mc_lo = D_lsm_mc[:,int(M*((1-Con)))]; D_pri_mc_lo = D_pri_mc[:,int(M*((1-Con)))]
R_lsm_mc_lo = R_lsm_mc[:,int(M*((1-Con)))]; R_pri_mc_lo = R_pri_mc[:,int(M*((1-Con)))]
M_lsm_mc_lo = M_lsm_mc[:,int(M*((1-Con)))]; M_pri_mc_lo = M_pri_mc[:,int(M*((1-Con)))]

D_lsm_mc_hi = D_lsm_mc[:,int(M*(1-(1-Con)))]; D_pri_mc_hi = D_pri_mc[:,int(M*(1-(1-Con)))]
R_lsm_mc_hi = R_lsm_mc[:,int(M*(1-(1-Con)))]; R_pri_mc_hi = R_pri_mc[:,int(M*(1-(1-Con)))]
M_lsm_mc_hi = M_lsm_mc[:,int(M*(1-(1-Con)))]; M_pri_mc_hi = M_pri_mc[:,int(M*(1-(1-Con)))]

##################################################
# Report
##################################################
print("Execution time: {} sec".format(t1-t0))

### Label formatting
Labels = ['L={0:1.0e}'.format(l) for l in L_ALL]
Colors = linspecer(len(L_ALL)*2, colorBlindFlag=True)

### Effective margin --------------------------------------------------
plt.figure()
## Data
for lnd in range(len(L_ALL)):
    # MC PI + MD LSSM
    plt.fill_between(N_ALL, 100*M_lsm_mc_lo[:,lnd], 100*M_lsm_mc_hi[:,lnd],
                     color = Colors[2*lnd,:], alpha = 0.1)
    plt.plot(N_ALL, 100*M_lsm_mc_lo[:,lnd],
             color = Colors[2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, 100*M_lsm_mc_hi[:,lnd],
             color = Colors[2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, 100*M_lsm_mc_mu[:,lnd],
             color = Colors[2*lnd,:], linewidth = 1.5,
             label = 'PI+MD LSSM; '+Labels[lnd])
    # MC PRI
    plt.fill_between(N_ALL, 100*M_pri_mc_lo[:,lnd], 100*M_pri_mc_hi[:,lnd],
                     color = Colors[1+2*lnd,:], alpha = 0.1)
    plt.plot(N_ALL, 100*M_pri_mc_lo[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, 100*M_pri_mc_hi[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, 100*M_pri_mc_mu[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 1.5,
             label = 'PRI; '+Labels[lnd])

## Annotation
plt.plot(N_ALL,[0]*len(N_ALL), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Margin (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
# Export
plt.savefig('../../images/comp_PRI_Meff_c{0:}.png'.format(MYCASE))
plt.close()

### Effective reliability --------------------------------------------------
# Post-process: Failure probabilities
F_lsm_mc_lo = 1 - R_lsm_mc_lo; F_pri_mc_lo = 1 - R_pri_mc_lo
F_lsm_mc_hi = 1 - R_lsm_mc_hi; F_pri_mc_hi = 1 - R_pri_mc_hi
F_lsm_mc_mu = 1 - R_lsm_mc_mu; F_pri_mc_mu = 1 - R_pri_mc_mu

### Effective margin
plt.figure()
## Data

for lnd in range(len(L_ALL)):
    # MC PI + MD LSSM
    plt.fill_between(N_ALL, F_lsm_mc_lo[:,lnd], F_lsm_mc_hi[:,lnd],
                     color = Colors[2*lnd,:], alpha = 0.1)
    plt.plot(N_ALL, F_lsm_mc_lo[:,lnd],
             color = Colors[2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, F_lsm_mc_hi[:,lnd],
             color = Colors[2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, F_lsm_mc_mu[:,lnd],
             color = Colors[2*lnd,:], linewidth = 1.5,
             label = 'PI+MD LSSM; '+Labels[lnd])
    # MC PRI
    plt.fill_between(N_ALL, F_pri_mc_lo[:,lnd], F_pri_mc_hi[:,lnd],
                     color = Colors[1+2*lnd,:], alpha = 0.1)
    plt.plot(N_ALL, F_pri_mc_lo[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, F_pri_mc_hi[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 0.5)
    plt.plot(N_ALL, F_pri_mc_mu[:,lnd],
             color = Colors[1+2*lnd,:], linewidth = 1.5,
             label = 'PRI; '+Labels[lnd])

## Annotation
plt.plot(N_ALL,[1-Rel]*len(N_ALL), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Complementary Reliability (log Pr)')
plt.yscale('log')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
# Export
plt.savefig('../../images/comp_PRI_Feff_c{0:}.png'.format(MYCASE))
plt.close()

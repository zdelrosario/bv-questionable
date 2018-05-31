### Uniaxial tension problem
# Comparison of different methods for managing statistical (sampling)
# uncertainty on a uniaxial tension test.
#
# Compares a number of approaches:
# BV         = basis value
# PI         = plug in
# HD LSSM    = hierarchical difference limit state sample margin
#
# Zachary del Rosario, Mar. 2018

import numpy as np
import matplotlib.pyplot as plt
import time
import pyutil.numeric as ut
import pandas as pd

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
M     = int(3e3); L_MC  = int(3e3); suf = "_refined"
# M     = int(2e2); L_MC  = int(5e2); suf = ""
N_ALL = [20,50,100,500,1000] # Sample count sweep

# Select case
MYCASE = 2

# Design parameters
#                     Rel,  Con,  Pop
PARAM = np.array([[  0.90, 0.95, 0.99],    #  A-basis;  lax reliability
                  [  0.99, 0.95, 0.99],    #  A-basis;  med reliability
                  [1-1e-7, 0.95, 0.99],    #  A-basis;  strict reliability
                  [1-1e-7, 1-1e-7, 0.99]]) #  Z-basis;  strict reliability

# RV parameters
MU_CR  = 600.   * 1e+6 # mean, critical stress     [Pa]
MU_A   = 100.   * 1e+6 # mean, axial force         [N]
TAU_CR = MU_CR * 0.1  # std. dev., critical stress [Pa]
TAU_A  = MU_A  * 0.1  # std. dev., axial force     [N]
# Fixed DV
RADIUS = 1.0           # cylinder radius            [m]

D_LO = 0.01; D_HI = 0.08 # thickness bounds for binary search [m]
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

##################################################
# Design sweep
##################################################
Rel = PARAM[MYCASE][0]
Con = PARAM[MYCASE][1]
Pop = PARAM[MYCASE][2]

beta_cr = norm.ppf( Rel )
zc      = norm.ppf(Con)

d_s = fD_star(Rel)

D_mip = np.zeros((len(N_ALL),M)); D_pi = np.zeros((len(N_ALL),M))
R_mip = np.zeros((len(N_ALL),M)); R_pi = np.zeros((len(N_ALL),M))
M_mip = np.zeros((len(N_ALL),M)); M_pi = np.zeros((len(N_ALL),M))
D_bv  = np.zeros((len(N_ALL),M)); D_psm = np.zeros((len(N_ALL),M))
R_bv  = np.zeros((len(N_ALL),M)); R_psm = np.zeros((len(N_ALL),M))
M_bv  = np.zeros((len(N_ALL),M)); M_psm = np.zeros((len(N_ALL),M))

M_samp = np.zeros((len(N_ALL),M))

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

        ## Analytic BV approach
        B = mu_dat - fK_pc(Pop, Con, N_ALL[jnd]) * np.sqrt(s2_dat)
        D_bv[jnd,ind] = fD_bv(B,Rel)
        R_bv[jnd,ind] = fRel_eff(D_bv[jnd,ind])
        M_bv[jnd,ind] = (D_bv[jnd,ind]-d_s) / d_s

        ## Analytic PI approach
        D_pi[jnd,ind] = fD_star(Rel,mu_cr=mu_dat,tau_cr=np.sqrt(s2_dat))
        R_pi[jnd,ind] = fRel_eff(D_pi[jnd,ind])
        M_pi[jnd,ind] = (D_pi[jnd,ind]-d_s) / d_s

        ## SOMC PI + exact limit state margin
        Mu_ens = t.rvs(N_ALL[jnd]-1,size=L_MC)*np.sqrt(TAU_CR**2/(N_ALL[jnd]-1)) + MU_CR
        S2_ens = chi2.rvs(N_ALL[jnd]-1,size=L_MC)*TAU_CR**2/(N_ALL[jnd]-1)
        Rd_ens  = np.zeros(L_MC)

        # def obj_mip(d):
        #     # Approximate margin
        #     Rd_ens = np.array([fRel_fcn(d,Mu_ens[ind],np.sqrt(S2_ens[ind])) \
        #                        for ind in range(L_MC)]) - Rel
        #     Rd_ens.sort()
        #     p_tmp = Rd_ens[int(Con*L_MC)]
        #     # Solve reliability problem
        #     d_tmp = fD_star(Rel+p_tmp,mu_cr=mu_dat,tau_cr=np.sqrt(s2_dat))
        #     # Check margin consistency
        #     Rd_ens = np.array([fRel_fcn(d_tmp,Mu_ens[ind],np.sqrt(S2_ens[ind])) \
        #                        for ind in range(L_MC)]) - Rel
        #     Rd_ens.sort()
        #     p = Rd_ens[int(Con*L_MC)]

        #     return p - p_tmp

        # D_mip[jnd,ind] = bisect(obj_mip, D_LO, D_HI,
        #                         xtol = 1e-14, maxiter = int(1e3))

        # DEBUG -- try a fixed point iteration
        def iter_mip(d):
            # Approximate margin
            # Rd_ens = np.array([fRel_fcn(d,Mu_ens[ind],np.sqrt(S2_ens[ind])) \
            #                    for ind in range(L_MC)]) - Rel
            Rd_ens = np.array([fRel_fcn(d,Mu_ens[ind],np.sqrt(S2_ens[ind])) \
                               for ind in range(L_MC)]) - fRel_eff(d)
            Rd_ens.sort()
            p_tmp = Rd_ens[int(Con*L_MC)]
            # Solve reliability problem
            return fD_star(Rel+p_tmp,mu_cr=mu_dat,tau_cr=np.sqrt(s2_dat))

        d_tmp = d_s
        for knd in range(20):
            d_tmp = iter_mip(d_s)
        D_mip[jnd,ind] = d_tmp

        R_mip[jnd,ind] = fRel_eff(D_mip[jnd,ind])
        M_mip[jnd,ind] = (D_mip[jnd,ind]-d_s)/d_s

    ut.print_progress(jnd+1,len(N_ALL),bar_length=60)
t1 = time.time()

##################################################
# Post-process
##################################################
D_mip.sort(axis=1); D_pi.sort(axis=1); D_bv.sort(axis=1)
R_mip.sort(axis=1); R_pi.sort(axis=1); R_bv.sort(axis=1)
M_mip.sort(axis=1); M_pi.sort(axis=1); M_bv.sort(axis=1)
M_samp.sort(axis=1)

D_mip_mu = np.mean(D_mip,axis=1); D_pi_mu = np.mean(D_pi,axis=1)
R_mip_mu = np.mean(R_mip,axis=1); R_pi_mu = np.mean(R_pi,axis=1)
M_mip_mu = np.mean(M_mip,axis=1); M_pi_mu = np.mean(M_pi,axis=1)
D_bv_mu  = np.mean(D_bv,axis=1);  D_psm_mu  = np.mean(D_psm,axis=1)
R_bv_mu  = np.mean(R_bv,axis=1);  R_psm_mu  = np.mean(R_psm,axis=1)
M_bv_mu  = np.mean(M_bv,axis=1);  M_psm_mu  = np.mean(M_psm,axis=1)
M_samp_mu = np.mean(M_samp,axis=1)

# D_mip_lo = D_mip[:,int(M*((1-Con)/2))]; D_pi_lo = D_pi[:,int(M*((1-Con)/2))]
# R_mip_lo = R_mip[:,int(M*((1-Con)/2))]; R_pi_lo = R_pi[:,int(M*((1-Con)/2))]
# M_mip_lo = M_mip[:,int(M*((1-Con)/2))]; M_pi_lo = M_pi[:,int(M*((1-Con)/2))]
# D_bv_lo = D_bv[:,int(M*((1-Con)/2))];   D_psm_lo = D_psm[:,int(M*((1-Con)/2))]
# R_bv_lo = R_bv[:,int(M*((1-Con)/2))];   R_psm_lo = R_psm[:,int(M*((1-Con)/2))]
# M_bv_lo = M_bv[:,int(M*((1-Con)/2))];   M_psm_lo = M_psm[:,int(M*((1-Con)/2))]
# M_samp_lo = M_samp[:,int(M*((1-Con)/2))]

# D_mip_hi = D_mip[:,int(M*(1-(1-Con)/2))]; D_pi_hi = D_pi[:,int(M*(1-(1-Con)/2))]
# R_mip_hi = R_mip[:,int(M*(1-(1-Con)/2))]; R_pi_hi = R_pi[:,int(M*(1-(1-Con)/2))]
# M_mip_hi = M_mip[:,int(M*(1-(1-Con)/2))]; M_pi_hi = M_pi[:,int(M*(1-(1-Con)/2))]
# D_bv_hi = D_bv[:,int(M*(1-(1-Con)/2))];   D_psm_hi = D_psm[:,int(M*(1-(1-Con)/2))]
# R_bv_hi = R_bv[:,int(M*(1-(1-Con)/2))];   R_psm_hi = R_psm[:,int(M*(1-(1-Con)/2))]
# M_bv_hi = M_bv[:,int(M*(1-(1-Con)/2))];   M_psm_hi = M_psm[:,int(M*(1-(1-Con)/2))]
# M_samp_hi = M_samp[:,int(M*(1-(1-Con)/2))]

D_mip_lo = D_mip[:,int(M*((1-Con)))]; D_pi_lo = D_pi[:,int(M*((1-Con)))]
R_mip_lo = R_mip[:,int(M*((1-Con)))]; R_pi_lo = R_pi[:,int(M*((1-Con)))]
M_mip_lo = M_mip[:,int(M*((1-Con)))]; M_pi_lo = M_pi[:,int(M*((1-Con)))]
D_bv_lo = D_bv[:,int(M*((1-Con)))];   D_psm_lo = D_psm[:,int(M*((1-Con)))]
R_bv_lo = R_bv[:,int(M*((1-Con)))];   R_psm_lo = R_psm[:,int(M*((1-Con)))]
M_bv_lo = M_bv[:,int(M*((1-Con)))];   M_psm_lo = M_psm[:,int(M*((1-Con)))]
M_samp_lo = M_samp[:,int(M*((1-Con)))]

D_mip_hi = D_mip[:,int(M*(1-(1-Con)))]; D_pi_hi = D_pi[:,int(M*(1-(1-Con)))]
R_mip_hi = R_mip[:,int(M*(1-(1-Con)))]; R_pi_hi = R_pi[:,int(M*(1-(1-Con)))]
M_mip_hi = M_mip[:,int(M*(1-(1-Con)))]; M_pi_hi = M_pi[:,int(M*(1-(1-Con)))]
D_bv_hi = D_bv[:,int(M*(1-(1-Con)))];   D_psm_hi = D_psm[:,int(M*(1-(1-Con)))]
R_bv_hi = R_bv[:,int(M*(1-(1-Con)))];   R_psm_hi = R_psm[:,int(M*(1-(1-Con)))]
M_bv_hi = M_bv[:,int(M*(1-(1-Con)))];   M_psm_hi = M_psm[:,int(M*(1-(1-Con)))]
M_samp_hi = M_samp[:,int(M*(1-(1-Con)))]

##################################################
# Write results
##################################################
df = pd.DataFrame(
    data = {
        "N"       : N_ALL,
        "R"       : [Rel] * len(N_ALL),
        "C"       : [Con] * len(N_ALL),
        "P"       : [Pop] * len(N_ALL),

        "M_bv_mu" : M_bv_mu,
        "M_bv_lo" : M_bv_lo,
        "M_bv_hi" : M_bv_hi,

        "R_bv_mu" : R_bv_mu,
        "R_bv_lo" : R_bv_lo,
        "R_bv_hi" : R_bv_hi,

        "M_pi_mu" : M_pi_mu,
        "M_pi_lo" : M_pi_lo,
        "M_pi_hi" : M_pi_hi,

        "R_pi_mu" : R_pi_mu,
        "R_pi_lo" : R_pi_lo,
        "R_pi_hi" : R_pi_hi,

        "M_mip_mu" : M_mip_mu,
        "M_mip_lo" : M_mip_lo,
        "M_mip_hi" : M_mip_hi,

        "R_mip_mu" : R_mip_mu,
        "R_mip_lo" : R_mip_lo,
        "R_mip_hi" : R_mip_hi
    }
)

df.to_csv("../../data/semi_MIP_c{0:}{1:}.csv".format(MYCASE, suf))

##################################################
# Report
##################################################
print("Execution time: {} sec".format(t1-t0))

Colors = linspecer(5)

### Effective margin
plt.figure()
## Data
# Basis Value
plt.plot(N_ALL,100*M_bv_lo,
         color = Colors[0,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL,100*M_bv_mu,
         color = Colors[0,:], linewidth = 2.0, label='AN+BV')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i] * 0.95] * 2,
             [100 * M_bv_mu[i], 100 * M_bv_lo[i]],
             color = Colors[0, :],
             linewidth = 0.5)

# Plug In
plt.plot(N_ALL,100*M_pi_lo,
         color = Colors[1,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL,100*M_pi_mu,
         color = Colors[1,:], linewidth = 2.0, label = 'AN+PI')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i]] * 2,
             [100 * M_pi_mu[i], 100 * M_pi_lo[i]],
             color = Colors[1, :],
             linewidth = 0.5)

# PI + semi-exact margin in probability
plt.plot(N_ALL, 100*M_mip_lo,
         color = Colors[4,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, 100*M_mip_mu,
         color = Colors[4,:], linewidth = 2.0, label='semi MIP')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i] * 1.05] * 2,
             [100 * M_mip_mu[i], 100 * M_mip_lo[i]],
             color = Colors[4, :],
             linewidth = 0.5)

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
plt.savefig('../../images/semi_MIP_Meff_c{0:}{1:}.png'.format(MYCASE, suf))
plt.close()

### Effective reliability
# Post-process: Failure probabilities
F_bv_lo = 1 - R_bv_lo; F_bv_hi = 1 - R_bv_hi; F_bv_mu = 1 - R_bv_mu
F_pi_lo = 1 - R_pi_lo; F_pi_hi = 1 - R_pi_hi; F_pi_mu = 1 - R_pi_mu
F_mip_lo = 1 - R_mip_lo; F_mip_hi = 1 - R_mip_hi; F_mip_mu = 1 - R_mip_mu

plt.figure()
## Data
# Basis Value
plt.plot(N_ALL, F_bv_lo,
         color = Colors[0,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, F_bv_mu,
         color = Colors[0,:], linewidth = 2.0, label = 'AN+BV')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i] * 0.95] * 2,
             [F_bv_mu[i], F_bv_lo[i]],
             color = Colors[0, :],
             linewidth = 0.5)

# Plug In
# plt.plot(N_ALL, F_pi_lo,
#          color = Colors[1,:], linewidth = 1.0, linestyle = ":")
# plt.plot(N_ALL, F_pi_mu,
#          color = Colors[1,:], linewidth = 2.0, label = 'AN+PI')
# for i in range(len(N_ALL)):
#     plt.plot([N_ALL[i]] * 2,
#              [F_pi_mu[i], F_pi_lo[i]],
#              color = Colors[1, :],
#              linewidth = 0.5)

# PI + semi-exact margin in probability
plt.plot(N_ALL, F_mip_lo,
         color = Colors[4,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, F_mip_mu,
         color = Colors[4,:], linewidth = 2.0, label = 'semi MIP')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i] * 1.05] * 2,
             [F_mip_mu[i], F_mip_lo[i]],
             color = Colors[4, :],
             linewidth = 0.5)

## Annotation
plt.plot(N_ALL,[1-Rel]*len(N_ALL), 'k--',
         label = 'Requested', linewidth = 2.0)
plt.xlabel('Sample Count')
plt.ylabel('Effective Failure Chance (Pr)')
plt.yscale('log')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
# Export
# plt.savefig('../../images/semi_MIP_Feff_c{0:}{1:}.png'.format(MYCASE, suf))
plt.savefig('../../images/semi_MIP_BV_Feff_c{0:}{1:}.png'.format(MYCASE, suf))
plt.close()

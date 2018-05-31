### Uniaxial tension problem
# Comparison of different methods for managing statistical (sampling)
# uncertainty on a uniaxial tension test.
#
# Compares a number of approaches:
# BV         = basis value
# PI         = plug in
# PI + Delta = plug in + sampling margin
# PRI        = predictive reliability index
#
# Zachary del Rosario, Feb. 2018

import numpy as np
import matplotlib.pyplot as plt
import time
import pyutil.numeric as ut
import pandas as pd

from scipy.stats import norm, nct, t, chi2
from scipy.optimize import bisect

plt.style.use('ggplot')
np.random.seed(101)
np.set_printoptions(precision=3)

##################################################
## Script parameters
##################################################
M     = int(1e3)           # Replications
N_ALL = [20,50,100,200,500,1000,2000] # Sample count sweep

# Select case
MYCASE = 1

# Design parameters
#                     Rel,  Con,  Pop
PARAM = np.array([[  0.90, 0.95, 0.99],  #  A-basis;  lax reliability
                  [1-1e-7, 0.95, 0.99]]) #  A-basis;  strict reliability

# RV parameters
MU_CR  = 600.   * 1e+6 # mean, critical stress     [Pa]
MU_A   = 100.   * 1e+6 # mean, axial force         [N]
TAU_CR = MU_CR * 0.1  # std. dev., critical stress [Pa]
TAU_A  = MU_A  * 0.1  # std. dev., axial force     [N]
# Fixed DV
RADIUS = 1.0           # cylinder radius            [m]

D_LO = 1e-10; D_HI = RADIUS * 0.999 # thickness bounds for binary search [m]
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

def fA_star(Rel,M=0):
    z = norm.ppf(Rel)
    return ((MU_CR-M)*MU_A + np.sqrt(z**2*(MU_CR-M)**2*TAU_A**2 + z**2*MU_A**2*TAU_CR**2 \
                        - z**4*TAU_CR**2*TAU_A**2)) / ((MU_CR-M)**2 - z**2*TAU_CR**2)

def fD_star(Rel,M=0):
    A_star = fA_star(Rel,M=M)
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

    return X[0] - X[1]/A_c(d)

##################################################
# Design sweep
##################################################
Rel = PARAM[MYCASE][0]
Con = PARAM[MYCASE][1]
Pop = PARAM[MYCASE][2]

beta_cr = norm.ppf( Rel )
zc      = norm.ppf(Con)

d_s = fD_star(Rel)

D_pid = np.zeros((len(N_ALL),M)); D_pi = np.zeros((len(N_ALL),M))
R_pid = np.zeros((len(N_ALL),M)); R_pi = np.zeros((len(N_ALL),M))
M_pid = np.zeros((len(N_ALL),M)); M_pi = np.zeros((len(N_ALL),M))
D_bv  = np.zeros((len(N_ALL),M)); D_ri = np.zeros((len(N_ALL),M))
R_bv  = np.zeros((len(N_ALL),M)); R_ri = np.zeros((len(N_ALL),M))
M_bv  = np.zeros((len(N_ALL),M)); M_ri = np.zeros((len(N_ALL),M))

M_samp = np.zeros((len(N_ALL),M))

t0 = time.time()
ut.print_progress(0,len(N_ALL)-1,bar_length=60)
for jnd in range(len(N_ALL)):
    for ind in range(M):
        X_dat = np.random.multivariate_normal(mean=[MU_CR,MU_A],
                                              cov=np.diag([TAU_CR**2,TAU_A**2]),
                                              size=N_ALL[jnd]).T
        mu_dat = np.mean(X_dat[0])
        s2_dat = np.var(X_dat[0])
        T_dat  = [mu_dat,s2_dat]

        # Analytic BV approach
        B = mu_dat - fK_pc(Pop, Con, N_ALL[jnd]) * np.sqrt(s2_dat)
        D_bv[jnd,ind] = fD_bv(B,Rel)
        R_bv[jnd,ind] = fRel_eff(D_bv[jnd,ind])
        M_bv[jnd,ind] = (D_bv[jnd,ind]-d_s) / d_s

        # Analytic PI approach
        D_pi[jnd,ind] = bisect(lambda d: fRel_fcn(d,mu_dat,np.sqrt(s2_dat))-Rel,
                               D_LO, D_HI,
                               xtol=1e-14, maxiter=int(1e3))
        R_pi[jnd,ind] = fRel_eff(D_pi[jnd,ind])
        M_pi[jnd,ind] = (D_pi[jnd,ind]-d_s) / d_s

        # Monte Carlo PI + Delta approach
        # D  = del_rho(X_sim[0],T_dat)
        # H  = hes_rho(X_sim[0],T_dat)
        # Sg = np.diag([s2_dat/N_ALL[jnd],2*s2_dat**2/(N_ALL[jnd]-1)])

        # def objective(d):
        #     G   = g_lim(d,X_sim)
        #     h   = np.mean(G)
        #     dh  = np.mean(D*G,axis=1)
        #     d2f = np.mean(H*(G-h)**2,axis=2)
        #     d2h = np.mean(H*G,axis=2)

        #     m2 = np.dot(dh,np.dot(Sg,dh)) + 0.5*np.trace(np.dot(d2f,Sg))

        #     # No bias correction
        #     Pr_succ_p = np.mean( (G - zc*np.sqrt(m2)) > 0 )

        #     return Pr_succ_p - Rel

        # D_pid[jnd,ind] = bisect(objective, d_lo, d_hi,
        #                         xtol=1e-14, maxiter=int(1e3))
        # R_pid[jnd,ind] = Rel_eff(D_pid[jnd,ind])
        # M_pid[jnd,ind] = (D_pid[jnd,ind]-d_s)/d_s

        # # Store the realized sampling margin
        # G   = g_lim(D_pid[jnd,ind],X_sim)
        # h   = np.mean(G)
        # dh  = np.mean(D*G,axis=1)
        # d2f = np.mean(H*(G-h)**2,axis=2)
        # m2  = np.dot(dh,np.dot(Sg,dh)) + 0.5*np.trace(np.dot(d2f,Sg))

        # M_samp[jnd,ind] = np.sqrt(m2)

    ut.print_progress(jnd,len(N_ALL)-1,bar_length=60)
t1 = time.time()

##################################################
# Post-process
##################################################
D_pid.sort(axis=1); D_pi.sort(axis=1); D_bv.sort(axis=1)
R_pid.sort(axis=1); R_pi.sort(axis=1); R_bv.sort(axis=1)
M_pid.sort(axis=1); M_pi.sort(axis=1); M_bv.sort(axis=1)
M_samp.sort(axis=1)

D_pid_mu = np.mean(D_pid,axis=1); D_pi_mu = np.mean(D_pi,axis=1)
R_pid_mu = np.mean(R_pid,axis=1); R_pi_mu = np.mean(R_pi,axis=1)
M_pid_mu = np.mean(M_pid,axis=1); M_pi_mu = np.mean(M_pi,axis=1)
D_bv_mu  = np.mean(D_bv,axis=1);  D_ri_mu  = np.mean(D_ri,axis=1)
R_bv_mu  = np.mean(R_bv,axis=1);  R_ri_mu  = np.mean(R_ri,axis=1)
M_bv_mu  = np.mean(M_bv,axis=1);  M_ri_mu  = np.mean(M_ri,axis=1)
M_samp_mu = np.mean(M_samp,axis=1)

D_pid_lo = D_pid[:,int(M*((1-Con)/2))]; D_pi_lo = D_pi[:,int(M*((1-Con)/2))]
R_pid_lo = R_pid[:,int(M*((1-Con)/2))]; R_pi_lo = R_pi[:,int(M*((1-Con)/2))]
M_pid_lo = M_pid[:,int(M*((1-Con)/2))]; M_pi_lo = M_pi[:,int(M*((1-Con)/2))]
D_bv_lo = D_bv[:,int(M*((1-Con)/2))];   D_ri_lo = D_ri[:,int(M*((1-Con)/2))]
R_bv_lo = R_bv[:,int(M*((1-Con)/2))];   R_ri_lo = R_ri[:,int(M*((1-Con)/2))]
M_bv_lo = M_bv[:,int(M*((1-Con)/2))];   M_ri_lo = M_ri[:,int(M*((1-Con)/2))]
M_samp_lo = M_samp[:,int(M*((1-Con)/2))]

D_pid_hi = D_pid[:,int(M*(1-(1-Con)/2))]; D_pi_hi = D_pi[:,int(M*(1-(1-Con)/2))]
R_pid_hi = R_pid[:,int(M*(1-(1-Con)/2))]; R_pi_hi = R_pi[:,int(M*(1-(1-Con)/2))]
M_pid_hi = M_pid[:,int(M*(1-(1-Con)/2))]; M_pi_hi = M_pi[:,int(M*(1-(1-Con)/2))]
D_bv_hi = D_bv[:,int(M*(1-(1-Con)/2))];   D_ri_hi = D_ri[:,int(M*(1-(1-Con)/2))]
R_bv_hi = R_bv[:,int(M*(1-(1-Con)/2))];   R_ri_hi = R_ri[:,int(M*(1-(1-Con)/2))]
M_bv_hi = M_bv[:,int(M*(1-(1-Con)/2))];   M_ri_hi = M_ri[:,int(M*(1-(1-Con)/2))]
M_samp_hi = M_samp[:,int(M*(1-(1-Con)/2))]

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
        "R_pi_hi" : R_pi_hi
    }
)

df.to_csv("../../data/an_tension_c{0:}.csv".format(MYCASE))

##################################################
# Report
##################################################
print("Execution time: {} sec".format(t1-t0))

### Effective margin
plt.figure()
## Data
# Basis Value
plt.fill_between(N_ALL, 100*M_bv_lo, 100*M_bv_hi,
                 color='blue', alpha=0.1)
plt.plot(N_ALL,100*M_bv_mu,
         color='blue',label='AN+BV')
# Plug In
plt.fill_between(N_ALL, 100*M_pi_lo, 100*M_pi_hi,
                 color='red', alpha=0.1)
plt.plot(N_ALL,100*M_pi_mu,
         color='red',label='AN+PI')
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
plt.savefig('../../images/an_tension_Meff_c{0:}.png'.format(MYCASE))
plt.close()

### Effective reliability
plt.figure()
## Data
# Basis Value
plt.fill_between(N_ALL, 1 - R_bv_lo, 1 - R_bv_hi,
                 color='blue', alpha=0.1)
plt.plot(N_ALL, 1 - R_bv_mu,
         color='blue',label='AN+BV')
# Plug In
plt.fill_between(N_ALL, 1 - R_pi_lo, 1 - R_pi_hi,
                 color='red', alpha=0.1)
plt.plot(N_ALL, 1 - R_pi_mu,
         color='red',label='AN+PI')
## Annotation
plt.plot(N_ALL, [1 - Rel]*len(N_ALL), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Failure Chance (Pr)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
plt.yscale('log')
# Export
plt.savefig('../../images/an_tension_Feff_c{0:}.png'.format(MYCASE))
plt.close()

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

from scipy.stats import norm, nct, t, chi2
from scipy.optimize import bisect

plt.style.use('ggplot')
np.random.seed(101)
np.set_printoptions(precision=3)

##################################################
## Script parameters
##################################################
M     = int(1e3)           # Replications
L     = int(1e6)           # MC Samples
# N_all = [10,20,50,100,200,500,1000,2000] # Sample count sweep
N_all = [100,200,500,1000,2000] # Sample count sweep, for mycase==3

# DEBUG -- fix case
mycase = 3

# Design parameters
#                     Rel,  Con,  Pop
Param = np.array([[  0.90, 0.80, 0.90],  # C-basis; lax reliability
                  [  0.90, 0.95, 0.90],  #
                  [  0.90, 0.95, 0.99],  # A-basis; lax reliability
                  [1-1e-6, 0.95, 0.99]]) # A-basis; strict reliability

# Test parameters
sig_cr = 600.   * 1e+6 # mean, critical stress      [Pa]
F_a0   = 100.   * 1e+6 # mean, axial force          [N]
# Note: Scale parameters encoded in stan model; make sure these match!
sig    = sig_cr * 0.1  # std. dev., critical stress [Pa]
tau    = F_a0   * 0.1  # std. dev., axial force     [N]
# Fixed DV
R      = 1.0           # cylinder radius            [m]

d_lo = 1e-10; d_hi = R * 0.999 # thickness bounds for binary search [m]
dx   = np.sqrt(np.finfo(float).eps)

##################################################
# Helper functions
##################################################
A_c = lambda d: np.pi*(2*R*d + d**2)
sig_a0 = lambda d: F_a0 / A_c(d)
gam    = lambda d: tau / A_c(d)

def k_pc(p,c,n):
    return nct.ppf(c,n-1,-norm.ppf(1-p)*np.sqrt(n)) / np.sqrt(n)

def z_score(d,s_c,s):
    return (s_c-sig_a0(d)) / np.sqrt(s**2+gam(d)**2)

def Rel_fcn(d,s_c,s):
    return norm.cdf(z_score(d,s_c,s))

def Rel_eff(d):
    return norm.cdf(z_score(d,sig_cr,sig))

def d_bv(B,Rel):
    return np.sqrt((tau*norm.ppf(Rel)+F_a0)/np.pi/B+R) - R

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
    D[1] = (X-T[0])**2/T[1]**2
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
Rel = Param[mycase][0]
Con = Param[mycase][1]
Pop = Param[mycase][2]

beta_cr = norm.ppf( Rel )
zc      = norm.ppf(Con)

d_s = bisect(lambda d: Rel_eff(d)-Rel,
             d_lo, d_hi,
             xtol=1e-14, maxiter=int(1e3))

D_pid = np.zeros((len(N_all),M)); D_pi = np.zeros((len(N_all),M))
R_pid = np.zeros((len(N_all),M)); R_pi = np.zeros((len(N_all),M))
M_pid = np.zeros((len(N_all),M)); M_pi = np.zeros((len(N_all),M))
D_bv  = np.zeros((len(N_all),M)); D_ri = np.zeros((len(N_all),M))
R_bv  = np.zeros((len(N_all),M)); R_ri = np.zeros((len(N_all),M))
M_bv  = np.zeros((len(N_all),M)); M_ri = np.zeros((len(N_all),M))
D_nmc = np.zeros((len(N_all),M));
R_nmc = np.zeros((len(N_all),M));
M_nmc = np.zeros((len(N_all),M));

M_samp = np.zeros((len(N_all),M))

t0 = time.time()
ut.print_progress(0,len(N_all)-1,bar_length=60)
for jnd in range(len(N_all)):
    for ind in range(M):
        X_dat = np.random.multivariate_normal(mean=[sig_cr,F_a0],
                                              cov=np.diag([sig**2,tau**2]),
                                              size=N_all[jnd]).T
        mu_dat = np.mean(X_dat[0])
        s2_dat = np.var(X_dat[0])
        T_dat  = [mu_dat,s2_dat]

        Z_sim = np.random.multivariate_normal(np.zeros(2),cov=np.eye(2),size=L)
        X_sim = (np.array([mu_dat,F_a0]) \
              + np.dot(Z_sim,np.diag([np.sqrt(s2_dat),tau]))).T

        # Analytic BV approach
        B = mu_dat - k_pc(Pop, Con, N_all[jnd]) * np.sqrt(s2_dat)
        D_bv[jnd,ind] = d_bv(B,Rel)
        R_bv[jnd,ind] = Rel_eff(D_bv[jnd,ind])
        M_bv[jnd,ind] = (D_bv[jnd,ind]-d_s) / d_s

        # Analytic PI approach
        D_pi[jnd,ind] = bisect(lambda d: Rel_fcn(d,mu_dat,np.sqrt(s2_dat))-Rel,
                               d_lo, d_hi,
                               xtol=1e-14, maxiter=int(1e3))
        R_pi[jnd,ind] = Rel_eff(D_pi[jnd,ind])
        M_pi[jnd,ind] = (D_pi[jnd,ind]-d_s) / d_s

        # Monte Carlo PI + Delta approach
        D  = del_rho(X_sim[0],T_dat)
        H  = hes_rho(X_sim[0],T_dat)
        Sg = np.diag([s2_dat/N_all[jnd],2*s2_dat**2/(N_all[jnd]-1)])

        def objective(d):
            G   = g_lim(d,X_sim)
            h   = np.mean(G)
            dh  = np.mean(D*G,axis=1)
            d2f = np.mean(H*(G-h)**2,axis=2)
            d2h = np.mean(H*G,axis=2)

            # m2 = np.dot(dh,np.dot(Sg,dh))
            m2 = np.dot(dh,np.dot(Sg,dh)) + 0.5*np.trace(np.dot(d2f,Sg))

            # No bias correction
            Pr_succ_p = np.mean( (G - zc*np.sqrt(m2)) > 0 )

            # Bias-corrected version
            # b = 0.5*np.trace(np.dot(d2h,Sg))
            # Pr_succ_p = np.mean( (G + b - zc*np.sqrt(m2)) > 0 )

            return Pr_succ_p - Rel

        D_pid[jnd,ind] = bisect(objective, d_lo, d_hi,
                                xtol=1e-14, maxiter=int(1e3))
        R_pid[jnd,ind] = Rel_eff(D_pid[jnd,ind])
        M_pid[jnd,ind] = (D_pid[jnd,ind]-d_s)/d_s

        # Store the realized sampling margin
        G   = g_lim(D_pid[jnd,ind],X_sim)
        h   = np.mean(G)
        dh  = np.mean(D*G,axis=1)
        d2f = np.mean(H*(G-h)**2,axis=2)
        m2  = np.dot(dh,np.dot(Sg,dh)) + 0.5*np.trace(np.dot(d2f,Sg))

        M_samp[jnd,ind] = np.sqrt(m2)

    ut.print_progress(jnd,len(N_all)-1,bar_length=60)
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
# Report
##################################################
print("Execution time: {} sec".format(t1-t0))

### Effective margin
plt.figure()
## Data
# Basis Value
plt.fill_between(N_all, 100*M_bv_lo, 100*M_bv_hi,
                 color='red', alpha=0.1)
plt.plot(N_all,100*M_bv_mu,
         color='red',label='AN+BV')
# Plug In
plt.fill_between(N_all, 100*M_pi_lo, 100*M_pi_hi,
                 color='blue', alpha=0.1)
plt.plot(N_all,100*M_pi_mu,
         color='blue',label='AN+PI')
# Plug In + Delta
plt.fill_between(N_all, 100*M_pid_lo, 100*M_pid_hi,
                 color='green', alpha=0.1)
plt.plot(N_all, 100*M_pid_lo, 'k-')
plt.plot(N_all, 100*M_pid_hi, 'k-')
plt.plot(N_all, 100*M_pid_mu,
         color='green',label='MC+PI+Delta')
## Annotation
plt.plot(N_all,[0]*len(N_all), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Margin (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_all[0],N_all[-1]])
plt.xscale('log')
# Export
plt.savefig('../../images/tension_Meff_c{0:}.png'.format(mycase))
plt.close()

### Sampling margin
plt.figure()
## Data
plt.fill_between(N_all, 100*M_samp_lo, 100*M_samp_hi,
                 color='green', alpha=0.1)
plt.plot(N_all, 100*M_samp_lo, 'k-')
plt.plot(N_all, 100*M_samp_hi, 'k-')
plt.plot(N_all, 100*M_samp_mu, color='green')
# Annotation
plt.xlabel('Sample Count')
plt.ylabel('Sampling Margin ([g])')
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([N_all[0],N_all[-1]])
plt.yscale('log')
plt.xscale('log')
# Export
plt.savefig('../../images/tension_Msamp_c{0:}.png'.format(mycase))
plt.close()

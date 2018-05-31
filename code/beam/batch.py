"""
Run multiple optimizations for the cantilevered beam problem, where
distribution parameters are estimated from samples.

Zach del Rosario and Richard W. Fenrich, April 26, 2018
"""

import numpy as np 
import matplotlib.pyplot as plt

from optimize import optimize
from beam import stress, disp

import cProfile

def plotData(obj_list, pf_s_list, pf_d_list, m_list, nopt, confidence_interval,
    exact_obj, exact_prob, label="none", file_prefix="none"):
    """ Plot effective margin and effective reliability. """

    obj_mean = []
    obj_lo = []
    obj_hi = []
    pf_s_mean = []
    pf_s_lo = []
    pf_s_hi = []
    pf_d_mean = []
    pf_d_lo = []
    pf_d_hi = []
    for m in m_list:
        obj_mean.append(np.mean(obj_list[m]))
        obj_lo.append(sorted(obj_list[m])[int(nopt*(1-confidence_interval))])
        obj_hi.append(sorted(obj_list[m])[int(nopt*confidence_interval)])
        pf_s_mean.append(np.mean(pf_s_list[m]))
        pf_s_lo.append(sorted(pf_s_list[m])[int(nopt*(1-confidence_interval))])
        pf_s_hi.append(sorted(pf_s_list[m])[int(nopt*confidence_interval)])
        pf_d_mean.append(np.mean(pf_d_list[m]))
        pf_d_lo.append(sorted(pf_d_list[m])[int(nopt*(1-confidence_interval))])
        pf_d_hi.append(sorted(pf_d_list[m])[int(nopt*confidence_interval)])

    # Effective margin
    exact_val = exact_obj
    mu = [(i-exact_val)/exact_val*100 for i in obj_mean]
    lo = [(i-exact_val)/exact_val*100 for i in obj_lo]
    hi = [(i-exact_val)/exact_val*100 for i in obj_hi]

    plt.figure()
    plt.fill_between(m_list, lo, hi,
                    color = 'r', alpha = 0.1)
    plt.plot(m_list, lo,
            color = 'r', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi,
            color = 'r', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu,
            color ='r', linewidth = 2.0, label=label)
    ## Annotation
    plt.plot(m_list,[0]*len(m_list), 'k--', label='Requested')
    plt.xlabel('Sample Count')
    plt.ylabel('Effective Margin (%)')
    plt.tight_layout()
    plt.legend(loc=0)
    axes = plt.gca()
    axes.set_xlim([m_list[0],m_list[-1]])
    plt.xscale('log')
    # Export
    plt.savefig('%s_margin.png' % file_prefix)
    #plt.show()
    #plt.close()

    # Effective reliability
    exact_val = 100-exact_prob*100
    mu_s = [100-i*100 for i in pf_s_mean]
    lo_s = [100-i*100 for i in pf_s_lo]
    hi_s = [100-i*100 for i in pf_s_hi]
    mu_d = [100-i*100 for i in pf_d_mean]
    lo_d = [100-i*100 for i in pf_d_lo]
    hi_d = [100-i*100 for i in pf_d_hi]

    plt.figure()
    # Stress
    plt.fill_between(m_list, lo_s, hi_s,
                    color = 'r', alpha = 0.1)
    plt.plot(m_list, lo_s,
            color = 'r', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi_s,
            color = 'r', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu_s,
            color ='r', linewidth = 2.0, label='Stress: %s' % label)
    # Displacement
    plt.fill_between(m_list, lo_d, hi_d,
                    color = 'b', alpha = 0.1)
    plt.plot(m_list, lo_d,
            color = 'b', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi_d,
            color = 'b', linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu_d,
            color ='b', linewidth = 2.0, label='Displacement: %s' % label)
    ## Annotation
    plt.plot(m_list,[exact_val]*len(m_list), 'k--', label='Requested')
    plt.xlabel('Sample Count')
    plt.ylabel('Effective Reliability (%)')
    plt.tight_layout()
    plt.legend(loc=0)
    axes = plt.gca()
    axes.set_xlim([m_list[0],m_list[-1]])
    plt.xscale('log')
    # Export
    plt.savefig('%s_reliability.png' % file_prefix)
    #plt.close()

    # Save data used in plotting
    np.savetxt('plot.dat',np.array([m_list, lo, mu, hi, lo_s, mu_s, hi_s, lo_d, mu_d, hi_d]))

    # plt.show()

    return

def getSampleDistributionParams(rg1, mean, stddev, m):
    """ Sample from distributions for [H, V, E, S] and return sample mean and 
    sample standard deviation. """

    sample_mean = []
    sample_stddev = []

    # Exact distributions of lateral and vertical loads are known
    sample_mean.append(mean[0]) # H
    sample_stddev.append(stddev[0]) # H
    sample_mean.append(mean[1]) # V
    sample_stddev.append(stddev[1]) # V

    # Sample to estimate mean and stddev of E and S
    vale = rg1.multivariate_normal(mean=[mean[2]],
                                         cov=[[stddev[2]**2]],
                                         size=m)
    vals = rg1.multivariate_normal(mean=[mean[3]],
                                         cov=[[stddev[3]**2]],
                                         size=m)
    vale = np.squeeze(vale)
    vals = np.squeeze(vals)    
    e_sample_mean = np.mean(vale)
    e_sample_sd = np.sqrt((1./(m-1))*np.sum((vale - e_sample_mean)**2))
    s_sample_mean = np.mean(vals)
    s_sample_sd = np.sqrt((1./(m-1))*np.sum((vals - s_sample_mean)**2))
    sample_mean += [e_sample_mean, s_sample_mean]
    sample_stddev += [e_sample_sd, s_sample_sd]

    return np.array(sample_mean), np.array(sample_stddev)

def main():

    # Exact distribution data
    H_mean = 500. # 
    V_mean = 1000. # 
    E_mean = 2.9e7 #
    S_mean = 40000. # 
    H_sd = 100.
    V_sd = 100.
    E_sd = 1.45e6
    S_sd = 2000.
#    m_list = [20, 50, 100, 200, 500, 1000, 2000] # number of samples used to 
                                                 # determine distribution params 
                                                 # for E and S
    m_list = [100]
    
    # Samples for evaluating probability during post-processing
    ns = int(1e5)
    mean = [H_mean, V_mean, E_mean, S_mean]
    stddev = [H_sd, V_sd, E_sd, S_sd]
    rg0 = np.random.RandomState(seed=202)
    rg0_init_state = rg0.get_state()
    X = rg0.multivariate_normal(mean=mean,
                                cov=np.diag([j**2 for j in stddev]),
                                size=ns)

    # Setup plotting parameters
    nopt = 40 # number of optimizations to run
    confidence_interval = 0.95

    # Setup optimization parameters
    d0 = [2.5, 2.5] # t, w
    tol = 1e-6
    maxiter = 500
    bounds = [(1,4),(1,4)] # [(t_lower, t_upper),(w_lower, w_upper)]

    # Setup margin parameters
    info = {}
    # --- Data for constraints with basis values
    info["bv_p"] = 0.99
    info["bv_c"] = 0.95
    # --- Data for monte carlo estimates of reliability
    info["N"] = int(1e5) # number of samples
    info["pf"] = 0.00135 # desired probability of failure
    # --- Data for mean difference limit dispersion margin
    info["ldm_confidence_interval"] = 0.95

    # Calculate exact optimal solution (based on Monte Carlo with N = info["N"])
    rg2 = np.random.RandomState(seed=202)
    rg2_init_state = rg2.get_state()
    result = optimize(d0, np.array(mean), np.array(stddev), 0, formulation="pi", 
                info=info, tol=tol, maxiter=maxiter, bounds=bounds,
                random_number_generator=rg2)
    exact_obj = result.fun 
    exact_prob = info["pf"]
    print("Exact objective:   %f" % exact_obj)
    print("Exact probability: %e" % exact_prob)
    print("Exact optimal t:   %f" % result.x[0])
    print("Exact optimal w:   %f" % result.x[1])

    # Estimate probability of failure for exact solution
    gs = stress(result.x, X)
    gd = disp(result.x, X)
    pf_s = float(np.sum(gs <= 0))/ns 
    pf_d = float(np.sum(gd <= 0))/ns
    
    # Print results
    print(result)
    print('Estimated pf for Stress: %e' % pf_s)
    print('Estimated pf for Displa: %e' % pf_d)

    # Run optimizations
    # f_list = ["det","pi","bv","md_ldm","mip_ldm"]
    # leg_list = ["deterministic", "MC+PI; L=1e+05", "MC+BV; L=1e+05", 
    #     "MC+MD LDM; L=1e+05", "MC+PIM LDM; L=1e+05"]
    f_list = ["mip_ldm"]
    leg_list = ["MC+PI; L=1e+05"]
    data_name = [e+".txt" for e in f_list]
    pre_list = f_list

    rg1 = np.random.RandomState(seed=101)
    rg1_init_state = rg1.get_state()

    for fi, f in enumerate(f_list):

        formulation = f

        obj_list = {}
        pf_s_list = {}
        pf_d_list = {}
        t_list = {}
        w_list = {}
        for m in m_list:

            rg1.set_state(rg1_init_state)

            obj_list[m] = []
            pf_s_list[m] = []
            pf_d_list[m] = []
            t_list[m] = []
            w_list[m] = []

            x_mean_list = []
            x_stddev_list = []
            
            for i in range(nopt):
                
                rg2.set_state(rg2_init_state)

                # Get sample mean and variance of distributions
                X_mean, X_stddev = getSampleDistributionParams(rg1, mean, stddev, m)
                x_mean_list.append(X_mean)
                x_stddev_list.append(X_stddev)

                # Run optimization
                if i > 0:
                    d0 = result.x # warm start optimizer at previous solution
                result = optimize(d0, X_mean, X_stddev, m, formulation=formulation, 
                    info=info, tol=tol, maxiter=maxiter, bounds=bounds, 
                    random_number_generator=rg2)
                
                # Check results
                if not result.success:
                    print(result)
                    raise RuntimeError("Optimization unsuccessful.")
                
                # Extract results
                fopt = result.fun # objective
                topt, wopt = list(result.x)

                # Post-process results
                gs = stress(result.x, X)
                gd = disp(result.x, X)
                pf_s = float(np.sum(gs <= 0))/ns 
                pf_d = float(np.sum(gd <= 0))/ns
                
                # Print results
                print(result)
                print('Estimated pf for Stress: %e' % pf_s)
                print('Estimated pf for Displa: %e' % pf_d)

                # Save data
                obj_list[m].append(fopt)
                pf_s_list[m].append(pf_s)
                pf_d_list[m].append(pf_d)
                t_list[m].append(topt)
                w_list[m].append(wopt)

            # Print sampled variable characteristics
            print "Sample Mean"
            print np.sum(np.array(x_mean_list),axis=0)/nopt
            print "Sample Std Dev"
            print np.sum(np.array(x_stddev_list),axis=0)/nopt

        # Print data characteristics
        print("")
        print("Exact Objective: %0.8f" % exact_obj)
        for m in m_list:
            print("\nSample Count: %i" % m)
            mu_tmp = np.mean(obj_list[m])
            sig_tmp = np.sqrt(np.var(obj_list[m], ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Objective Mean: %0.8f" % mu_tmp)
            print("Objective SD:   %0.8f" % sig_tmp)
            print("Objective CoV:  %0.8f" % cov_tmp)
            mu_tmp = np.mean(1.-np.array(pf_s_list[m]))
            sig_tmp = np.sqrt(np.var(1.-np.array(pf_s_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Reliability (stress) Mean: %0.8f" % mu_tmp)
            print("Reliability (stress) SD:   %0.8f" % sig_tmp)
            print("Reliability (stress) CoV:  %0.8f" % cov_tmp)    
            mu_tmp = np.mean(np.array(pf_s_list[m]))
            sig_tmp = np.sqrt(np.var(np.array(pf_s_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Prob. of Failure (stress) Mean: %0.8f" % mu_tmp)
            print("Prob. of Failure (stress) SD:   %0.8f" % sig_tmp)
            print("Prob. of Failure (stress) CoV:  %0.8f" % cov_tmp)   
            mu_tmp = np.mean(1-np.array(pf_d_list[m]))
            sig_tmp = np.sqrt(np.var(1-np.array(pf_d_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Reliability (disp) Mean: %0.8f" % mu_tmp)
            print("Reliability (disp) SD:   %0.8f" % sig_tmp)
            print("Reliability (disp) CoV:  %0.8f" % cov_tmp)   
            mu_tmp = np.mean(np.array(pf_d_list[m]))
            sig_tmp = np.sqrt(np.var(np.array(pf_d_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Prob. of Failure (disp) Mean: %0.8f" % mu_tmp)
            print("Prob. of Failure (disp) SD:   %0.8f" % sig_tmp)
            print("Prob. of Failure (disp) CoV:  %0.8f" % cov_tmp)   
            mu_tmp = np.mean(t_list[m])
            sig_tmp = np.sqrt(np.var(1-np.array(t_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Thickness Mean: %0.8f" % mu_tmp)
            print("Thickness SD:   %0.8f" % sig_tmp)
            print("Thickness CoV:  %0.8f" % cov_tmp)               
            mu_tmp = np.mean(w_list[m])
            sig_tmp = np.sqrt(np.var(1-np.array(w_list[m]), ddof=1))
            cov_tmp = sig_tmp/mu_tmp
            print("Width Mean: %0.8f" % mu_tmp)
            print("Width SD:   %0.8f" % sig_tmp)
            print("Width CoV:  %0.8f" % cov_tmp) 

        # Save data
        with open(data_name[fi],'w') as fil: 
            for m in m_list:
                for k, obj in enumerate(obj_list[m]):
                    fil.write("%i %0.16f %0.16e %0.16e\n" % (m,obj,
                    pf_s_list[m][k],pf_d_list[m][k]))

        # Plot data
        plotData(obj_list, pf_s_list, pf_d_list, m_list, nopt, confidence_interval,
            exact_obj, exact_prob, label=leg_list[fi], file_prefix=pre_list[fi])



if __name__ == '__main__':
    main()
    #cProfile.run('main()')

"""
Compare margins and reliabilities for several optimizations.

Rick Fenrich 5/7/18
"""

import numpy as np
import matplotlib.pyplot as plt

filename = ['results/pi/plot.dat', 
            'results/bv/plot.dat',
            'results/md_ldm/plot.dat',
            'results/mip_ldm/plot.dat']
label = ['MC+PI', 'MC+BV', 'MC+MD LDM', 'MC+MIP LDM']
color = ['r', 'b', 'g', 'k']

# =============================================================================
# Plot effective margin
# =============================================================================
plt.figure()
for i, f in enumerate(filename):
    data = np.loadtxt(f)
    # Data format is:
    # np.savetxt('plot.dat',np.array([m_list, lo, mu, hi, lo_s, 
    #                                 mu_s, hi_s, lo_d, mu_d, hi_d]))
    m_list = data[0,:]
    lo = data[1,:]
    mu = data[2,:]
    hi = data[3,:]

    plt.fill_between(m_list, lo, hi,
                    color = color[i], alpha = 0.1)
    plt.plot(m_list, lo,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu,
            color =color[i], linewidth = 2.0, label=label[i])

# Annotation
plt.plot(m_list,[0]*len(m_list), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Margin (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([m_list[0],m_list[-1]])
plt.xscale('log')
# Export
plt.savefig('beam_margin_comparison.png')
#plt.close()

# =============================================================================
# Plot effective reliability for stress
# =============================================================================
plt.figure()
for i, f in enumerate(filename):
    data = np.loadtxt(f)
    # Data format is:
    # np.savetxt('plot.dat',np.array([m_list, lo, mu, hi, lo_s, 
    #                                 mu_s, hi_s, lo_d, mu_d, hi_d]))
    m_list = data[0,:]
    lo = data[4,:]
    mu = data[5,:]
    hi = data[6,:]

    plt.fill_between(m_list, lo, hi,
                    color = color[i], alpha = 0.1)
    plt.plot(m_list, lo,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu,
            color =color[i], linewidth = 2.0, label=label[i])

# Annotation
exact_val = 100-1.35e-3*100
plt.plot(m_list,[exact_val]*len(m_list), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Reliability w.r.t. Stress (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([m_list[0],m_list[-1]])
plt.xscale('log')
# Export
plt.savefig('beam_reliability_stress_comparison.png')
#plt.close()

# =============================================================================
# Plot effective reliability for stress
# =============================================================================
plt.figure()
for i, f in enumerate(filename):
    data = np.loadtxt(f)
    # Data format is:
    # np.savetxt('plot.dat',np.array([m_list, lo, mu, hi, lo_s, mu_s, hi_s, lo_d, mu_d, hi_d]))
    m_list = data[0,:]
    lo = data[7,:]
    mu = data[8,:]
    hi = data[9,:]

    plt.fill_between(m_list, lo, hi,
                    color = color[i], alpha = 0.1)
    plt.plot(m_list, lo,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, hi,
            color = color[i], linewidth = 0.5, linestyle = ":")
    plt.plot(m_list, mu,
            color =color[i], linewidth = 2.0, label=label[i])

# Annotation
exact_val = 100-1.35e-3*100
plt.plot(m_list,[exact_val]*len(m_list), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Reliability w.r.t. Displacement (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([m_list[0],m_list[-1]])
plt.xscale('log')
# Export
plt.savefig('beam_reliability_disp_comparison.png')
#plt.close()

plt.show()

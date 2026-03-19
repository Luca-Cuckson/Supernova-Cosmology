import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats
from getdist import plots, MCSamples
import dataconstants as dac

#######################################################################################################################################
# plot styling

# Set global font to Times New Roman
#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

def print_confidence_intervals(samples):
    stats = samples.getMargeStats()
    for p in samples.paramNames.names:
        par = stats.parWithName(p.name)
        lims = par.limits[0]
        print(f"{p.label}: {lims.lower:.4f} to {lims.upper:.4f} (68%)")


#######################################################################################################################################
# LCDM

#labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m"]
labels = [r"\mathcal{M}_B", r"\Omega_m"]
names = ["m", "Om"]

samples1 = np.loadtxt('LCDM_prior.txt')
samples2 = np.loadtxt('LCDM_no_prior.txt')

LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



g = plots.get_subplot_plotter(width_inch=3.5)
g.triangle_plot([LCDMnone], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
g.add_y_bands(dac.Omegam_mu, dac.Omegam_sigma, ax=[1,0])
g.add_y_marker(dac.Omegam_mu, ax=[1,0])
plt.savefig('report_LCDM.png', bbox_inches='tight', dpi=300)

print(LCDMnone.PCA(["Om", "m"]))
#print(LCDMprior.PCA(["Om", "m"]))

ndim = 2

# Calculate 16th, 50th (median), and 84th percentiles
# For each parameter (axis 0)
for i in range(ndim):
    lower, median, upper = np.percentile(samples2[:, i], [15.87, 50, 84.13])
    print(f"LCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")

for i in range(ndim):
    lower, median, upper = np.percentile(samples1[:, i], [15.87, 50, 84.13])
    print(f"Prior LCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")

#######################################################################################################################################
# wCDM

labels = [r"\mathcal{M}_B", r"\Omega_m", r"w"]
names = ["m", "Om", "w"]

samples1 = np.loadtxt('wCDM_prior.txt')
samples2 = np.loadtxt('wCDM_no_prior.txt')

wCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
wCDMnone = MCSamples(samples=samples2, names=names, labels=labels)




f = plots.get_subplot_plotter(width_inch=3.5)
f.triangle_plot([wCDMnone, wCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB + BAO'])

f2 = plots.get_subplot_plotter(width_inch=3.2)
f2.triangle_plot([wCDMnone, wCDMprior], ["Om", "w"], plot_3d_with_param="m", alpha=0.3, legend_labels = ['SNe Ia', 'SNe Ia + Planck']) #, legend_loc='upper left'
f2.add_2d_contours(wCDMnone, "Om", "w", ax=[1, 0], color='k', ls='-.')
f2.add_2d_contours(wCDMprior, "Om", "w", ax=[1, 0], color='r')

f2.add_y_bands(dac.w_mu, dac.w_sigma, ax=[1,0])
f2.add_y_marker(-1, ax=[1,0])
f2.add_x_bands(dac.w_mu, dac.w_sigma, ax=[1,1])
f2.add_x_marker(-1, ax=[1,1])

cb = f2.fig.axes[-1]  # colorbar is always the last axis
cb.set_position([1, 0.2, 0.03, 0.6])
cb.yaxis.tick_right()
cb.yaxis.set_label_position('right')

plt.savefig('report_wCDM.png', bbox_inches='tight', dpi=300)


stats = wCDMnone.getMargeStats()
lims0 = stats.parWithName("Om").limits
lims1 = stats.parWithName("w").limits
for conf, lim0, lim1 in zip(wCDMnone.contours, lims0, lims1):
    print("x0 %s%% lower: %.3f upper: %.3f (%s)" % (conf, lim0.lower, lim0.upper, lim0.limitType()))
    print("x1 %s%% lower: %.3f upper: %.3f (%s)" % (conf, lim1.lower, lim1.upper, lim1.limitType()))

ndim = 3

# Calculate 16th, 50th (median), and 84th percentiles
# For each parameter (axis 0)
for i in range(ndim):
    lower, median, upper = np.percentile(samples2[:, i], [15.87, 50, 84.13])
    print(f"wCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")

for i in range(ndim):
    lower, median, upper = np.percentile(samples1[:, i], [15.87, 50, 84.13])
    print(f"Prior wCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")

#print(wCDMnone.PCA(["Om", "w"]))
#print(wCDMprior.PCA(["Om", "w"]))



#######################################################################################################################################
# Curved-LCDM

labels = [r"\mathcal{M}_B", r"\Omega_m", r"\Omega_k"]
names = ["m", "Om", "Ok"]

samples1 = np.loadtxt('LCDM_curved_prior.txt')
samples2 = np.loadtxt('LCDM_curved_no_prior.txt')

Curved_LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
Curved_LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



h = plots.get_subplot_plotter(width_inch=3.5)
h.triangle_plot([Curved_LCDMnone, Curved_LCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
h.add_x_bands(dac.Omegak, dac.Omegak_err)

h2 = plots.get_subplot_plotter(width_inch=3.2)
h2.triangle_plot([Curved_LCDMnone, Curved_LCDMprior], ["Om", "Ok"], plot_3d_with_param="m", alpha=0.3, legend_labels = ['SNe Ia', 'SNe Ia + Planck'])
h2.add_2d_contours(Curved_LCDMnone, "Om", "Ok", ax=[1, 0], color='k', ls='-.')
h2.add_2d_contours(Curved_LCDMprior, "Om", "Ok", ax=[1, 0], color='r')
#h2.add_x_bands(dac.Omegam_mu, dac.Omegam_sigma, ax=[1,0])


h2.add_y_bands(dac.Omegak, dac.Omegak_err, ax=[1,0])
h2.add_y_marker(0, ax=[1,0])
h2.add_x_bands(dac.Omegak, dac.Omegak_err, ax=[1,1])
h2.add_x_marker(0, ax=[1,1])
#h2.add_y_marker(0.193, ax=[1,0])
#h2.add_y_marker(-0.19, ax=[1,0])

cb = h2.fig.axes[-1]  # colorbar is always the last axis
cb.set_position([1, 0.2, 0.03, 0.6])
cb.yaxis.tick_right()
cb.yaxis.set_label_position('right')

plt.savefig('report_oCDM.png', bbox_inches='tight', dpi=300)

ndim = 3

# Calculate 16th, 50th (median), and 84th percentiles
# For each parameter (axis 0)
for i in range(ndim):
    lower, median, upper = np.percentile(samples2[:, i], [15.87, 50, 84.13])
    print(f"oCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")

for i in range(ndim):
    lower, median, upper = np.percentile(samples1[:, i], [15.87, 50, 84.13])
    print(f"Prior oCDM Parameter {i}: {median:.4f} (+{upper-median:.4f} / -{median-lower:.4f})")


#print(Curved_LCDMnone.PCA(["Om", "Ok"]))
print(Curved_LCDMprior.PCA(["Om", "Ok"]))

#######################################################################################################################################

#plt.show()
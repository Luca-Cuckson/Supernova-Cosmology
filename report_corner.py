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


#######################################################################################################################################
# LCDM

labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m"]
names = ["m", "Om"]

samples1 = np.loadtxt('LCDM_prior.txt')
samples2 = np.loadtxt('LCDM_no_prior.txt')

LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



g = plots.get_subplot_plotter(width_inch=3.5)
g.triangle_plot([LCDMnone, LCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
g.add_y_bands(dac.Omegam_mu, dac.Omegam_sigma, ax=[1,0])
g.add_y_marker(dac.Omegam_mu, ax=[1,0])
plt.savefig('report_LCDM.png', bbox_inches='tight')

#print(LCDMnone.PCA(["Om", "m"]))
#print(LCDMprior.PCA(["Om", "m"]))

#######################################################################################################################################
# wCDM

labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m", r"w"]
names = ["m", "Om", "w"]

samples1 = np.loadtxt('wCDM_prior.txt')
samples2 = np.loadtxt('wCDM_no_prior.txt')

wCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
wCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



f = plots.get_subplot_plotter(width_inch=3.5)
f.triangle_plot([wCDMnone, wCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])

f2 = plots.get_subplot_plotter(width_inch=3.5)
f2.triangle_plot([wCDMnone, wCDMprior], ["Om", "w"], plot_3d_with_param="m", legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
#f2.add_2d_contours(wCDMnone, "Om", "w", ax=[1, 0])
f2.add_y_bands(dac.w_mu, dac.w_sigma, ax=[1,0])
plt.savefig('report_wCDM.png', bbox_inches='tight')


stats = wCDMnone.getMargeStats()
lims0 = stats.parWithName("Om").limits
lims1 = stats.parWithName("w").limits
for conf, lim0, lim1 in zip(wCDMnone.contours, lims0, lims1):
    print("x0 %s%% lower: %.3f upper: %.3f (%s)" % (conf, lim0.lower, lim0.upper, lim0.limitType()))
    print("x1 %s%% lower: %.3f upper: %.3f (%s)" % (conf, lim1.lower, lim1.upper, lim1.limitType()))

print(wCDMnone.PCA(["Om", "w"]))
print(wCDMprior.PCA(["Om", "w"]))



#######################################################################################################################################
# Curved-LCDM

labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m", r"\Omega_k"]
names = ["m", "Om", "Ok"]

samples1 = np.loadtxt('LCDM_curved_prior.txt')
samples2 = np.loadtxt('LCDM_curved_no_prior.txt')

Curved_LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
Curved_LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



h = plots.get_subplot_plotter(width_inch=3.5)
h.triangle_plot([Curved_LCDMnone, Curved_LCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
h.add_x_bands(dac.Omegak, dac.Omegak_err)

h2 = plots.get_subplot_plotter(width_inch=3.5)
h2.triangle_plot([Curved_LCDMnone, Curved_LCDMprior], ["Om", "Ok"], plot_3d_with_param="m", legend_labels = ['SNe Ia', 'SNe Ia + CMB'])
h2.add_y_bands(dac.Omegak, dac.Omegak_err, ax=[1,0])
#h2.add_x_bands(dac.Omegam_mu, dac.Omegam_sigma, ax=[1,0])
plt.savefig('report_oCDM.png', bbox_inches='tight')

#print(Curved_LCDMnone.PCA(["Om", "Ok"]))
#print(Curved_LCDMprior.PCA(["Om", "Ok"]))

#######################################################################################################################################

plt.show()
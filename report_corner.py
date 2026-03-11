import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats
from getdist import plots, MCSamples

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
names = ["M", "Om"]

samples1 = np.loadtxt('LCDM_prior.txt')
samples2 = np.loadtxt('LCDM_no_prior.txt')

LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



g = plots.get_subplot_plotter()
g.triangle_plot([LCDMnone, LCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])

#######################################################################################################################################
# wCDM

labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m", r"w"]
names = ["M", "Om", "w"]

samples1 = np.loadtxt('wCDM_prior.txt')
samples2 = np.loadtxt('wCDM_no_prior.txt')

wCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
wCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



f = plots.get_subplot_plotter()
f.triangle_plot([wCDMnone, wCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])

#######################################################################################################################################
# Curved-LCDM

labels = [r"M_B - 5\log_{10}(h)", r"\Omega_m", r"\Omega_k"]
names = ["M", "Om", "Ok"]

samples1 = np.loadtxt('LCDM_curved_prior.txt')
samples2 = np.loadtxt('LCDM_curved_no_prior.txt')

Curved_LCDMprior = MCSamples(samples=samples1, names=names, labels=labels)
Curved_LCDMnone = MCSamples(samples=samples2, names=names, labels=labels)



h = plots.get_subplot_plotter()
h.triangle_plot([Curved_LCDMnone, Curved_LCDMprior], filled=True, legend_labels = ['SNe Ia', 'SNe Ia + CMB'])

#######################################################################################################################################

plt.show()
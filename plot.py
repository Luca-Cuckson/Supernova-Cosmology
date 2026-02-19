import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats
import corner
import emcee


# constants
f_lambda_0 = 6.61 * (10 ** (-12)) #W m^-2 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = (75*1000)/(3.09*10**22) #s^-1

# extract data from text file
file = 'sn_data(1).txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)


#######################################################################################################################################
# Find values

def find_theoretical_m_eff(redshift, *params):
    L = params[1] * 10**32
    H = params[2] * 10**(-18)
    def integrand(x):
        return 1 / (np.sqrt((1 - params[0]) * (1+x)**3 + params[0]))
    I = []
    for i in range(0, len(redshift)):
        I.append(integrate.quad(integrand, 0, redshift[i]))
    I = np.array(I)
    I_value = I[:, 0]
    fracs = np.empty(len(redshift))
    for i in range(0, len(redshift)):
        fracs[i] = L * H**2 / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + float(redshift[i])) ** 2) * float(I_value[i])**2)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff


# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


#######################################################################################################################################

popt = [0.669732, 3.86701, 2.91125]

residuals = m_eff - find_theoretical_m_eff(z, *popt)
norm_residuals = residuals / m_err


gauss_xs = np.linspace(-1.2, 9, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)


smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_m_eff(smooth_x, *popt)



plt.rcParams["font.size"] = 18
# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('$M_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, 0.86])

#plt.text(0.5, 21, '$\chi^2_{min}=81.3$')
#plt.text(0.5, 20.5, 'DoF = 41')
#plt.text(0.5, 20, '$\chi^2_{reduced}=1.98$')

# Residuals
plt.figure(6).add_axes((0.1, 0.1, 0.74, 0.2))
plt.scatter(z, norm_residuals, color = 'r', marker='o', s=4)
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.xlabel('$Redshift (z)$')
plt.ylim([-1.2,9])
plt.xlim([0, 0.86])

# Residuals histrogram
plt.figure(6).add_axes((0.85, 0.1, 0.14, 0.2))
plt.hist(norm_residuals, bins = 12, orientation='horizontal', density=True, color='r')
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.plot(stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std), gauss_xs, color='k')
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
plt.xticks([0.2, 0.4])

plt.savefig('plot.svg', bbox_inches = 'tight')
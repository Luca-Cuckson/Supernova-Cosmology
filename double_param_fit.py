import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats

# constants
f_lambda_0 = 6.61 * (10 ** (-12)) #W m^-2 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = (75*1000)/(3.09*10**22) #s^-1


# extract data from text file
file = 'sn_data(1).txt'
redshift, m_effective, m_error = np.loadtxt(file, usecols=(1,2,3), unpack=True)

# taking only low redshift data
near_redshift = redshift[42:60]
near_m_effective = m_effective[42:60]
near_m_error = m_error[42:60]
far_redshift = redshift[0:42]
far_m_effective = m_effective[0:42]
far_m_error = m_error[0:42]


#######################################################################################################################################
# Find values

def find_theoretical_m_eff(redshift, *params):
    def integrand(x):
        return 1 / (np.sqrt((1 - params[0]) * (1+x)**3 + params[0]))
    I = []
    for i in range(0, len(redshift)):
        I.append(integrate.quad(integrand, 0, redshift[i]))
    I = np.array(I)
    I_value = I[:, 0]
    fracs = np.empty(len(redshift))
    for i in range(0, len(redshift)):
        fracs[i] = params[1] / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + float(redshift[i])) ** 2) * float(I_value[i])**2)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff

initials = np.array([0.65, 0.0017])
popt, cov = scipy.optimize.curve_fit(find_theoretical_m_eff, redshift, m_effective, sigma=m_error, absolute_sigma=True, p0=initials, check_finite=True)


popt_err = np.sqrt(np.diag(cov)) # double check this

print('Omega_Lambda_0 = ({} \u00B1 {})'.format(popt[0], popt_err[0]))
print('L_Lambda_peak * H_0^2 = ({} \u00B1 {})'.format(popt[1], popt_err[1]))
L_lambda_peak, L_lambda_peak_err = popt[1] / H_0**2, popt_err[1] / H_0**2
print('L_lambda_peak = ({} \u00B1 {})'.format(L_lambda_peak, L_lambda_peak_err))


#######################################################################################################################################


residuals = m_effective - find_theoretical_m_eff(redshift, *popt)
norm_residuals = residuals / m_error

gauss_xs = np.linspace(-5.5, 5.5, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)


# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


# check all this!!!
chi_squared_min = chi_squared(popt, find_theoretical_m_eff, redshift, m_effective, m_error)
print('Min chi^2 = {}'.format(chi_squared_min))
degrees_of_freedom_value = redshift.size - popt.size
print('Reduced chi^2 = {}'.format(chi_squared_min/degrees_of_freedom_value))


#######################################################################################################################################
# Plots

smooth_x = np.linspace(min(redshift), max(redshift), 1000)
ys = find_theoretical_m_eff(smooth_x, *popt)

#fig1, (ax1, ax2) = plt.subplots(2, 1, sharex='col', height_ratios=(4,1))
#ax1.errorbar(far_redshift, far_m_effective, yerr=far_m_error, marker='o', color = 'r', elinewidth=0.8, linestyle='none', ms = 2)
#ax1.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
#ax1.plot(smooth_x, ys, color = 'g', linewidth = 0.8)
#ax2.set_xlabel('$Redshift (z)$')
#ax1.set_ylabel('$M_{eff}$')

#fig2, ax2 = plt.subplots()
#ax2.scatter(redshift, norm_residuals, marker='.')
#ax2.axhline(y=0, color='k')
#ax2.set_ylim([-5,5])

#fig3, ax3 = plt.subplots()
#ax3.hist(norm_residuals, bins = 15, density=True)
#ax3.axvline(x=1, color='k', linestyle=':', alpha=0.5)
#ax3.axvline(x=-1, color='k', linestyle=':', alpha=0.5)
#ax3.plot(gauss_xs, stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std))

plt.rcParams["font.size"] = 16
# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(redshift, m_effective, yerr=m_error, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 2)
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('$M_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, 0.86])

#plt.text(0.5, 21, '$\chi^2_{min}=81.6$')
#plt.text(0.5, 20.5, 'DoF = 41')
#plt.text(0.5, 20, '$\chi^2_{reduced}=1.99$')

# Residuals
plt.figure(6).add_axes((0.1, 0.1, 0.74, 0.2))
plt.scatter(redshift, norm_residuals, color = 'r', marker='o', s=4)
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.xlabel('$Redshift (z)$')
plt.ylim([-5.5,5.5])
plt.xlim([0, 0.86])

# Residuals histrogram
plt.figure(6).add_axes((0.85, 0.1, 0.14, 0.2))
plt.hist(norm_residuals, bins = 10, orientation='horizontal', density=True, color='r')
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.plot(stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std), gauss_xs, color='k')
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
plt.xticks([0.2, 0.4])

plt.savefig('double_fit', bbox_inches = 'tight')

plt.show()
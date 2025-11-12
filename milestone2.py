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


# find d_L for nearby supernovae
def get_near_d_L(z):
    return c * z / (H_0*1000) # output in Mpc

# find L_lambda_peak * H_0^2 from redshift for nearby supernovae
def find_milestone_value(redshift, value):
    redshift = np.array(redshift)
    frac = value / (f_lambda_0 * 4 * np.pi * c**2 * redshift**2)
    m_eff = -2.5 * np.log10(frac)
    return m_eff

# optimise to find value of L_lambda_peak * H_0^2
popt1, cov1 = scipy.optimize.curve_fit(find_milestone_value, near_redshift, near_m_effective, sigma = near_m_error, absolute_sigma=True, p0=0.0016378, check_finite=True)
popt1_err = np.sqrt(np.diag(cov1)) # double check this

# print value
print('L_lambda_peak * H_0^2 = ({} \u00B1 {})'.format(popt1[0], popt1_err[0]))
value = float(popt1[0])
L_lambda_peak, L_lambda_peak_err = popt1 / H_0**2, popt1_err / H_0**2
print('L_lambda_peak = ({} \u00B1 {})'.format(L_lambda_peak[0], L_lambda_peak_err[0]))


# find the expected value of m_eff for a given Omega
def find_theoretical_m_eff(redshift, Omega):
    def integrand(x):
        return 1 / (np.sqrt((1 - Omega) * (1+x)**3 + Omega))
    I = []
    for i in range(0, len(redshift)):
        I.append(integrate.quad(integrand, 0, redshift[i]))
    I = np.array(I)
    I_value = I[:, 0]
    fracs = np.empty(len(redshift))
    for i in range(0, len(redshift)):
        fracs[i] = value / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + float(redshift[i])) ** 2) * float(I_value[i])**2)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff

# optimise to find value of Omega
popt2, cov2 = scipy.optimize.curve_fit(find_theoretical_m_eff, far_redshift, far_m_effective, sigma = far_m_error, absolute_sigma=True, p0=np.array([0.685]), check_finite=True)
popt2_err = np.sqrt(np.diag(cov2)) # double check this

print('Omega_Lambda_0 = ({} \u00B1 {})'.format(popt2[0], popt2_err[0]))



#######################################################################################################################################

# You've plotted residuals of the near redshift stuff to the far redshift function you silly goose
# calculate residuals
far_residuals = far_m_effective - find_theoretical_m_eff(far_redshift, popt2)
far_norm_residuals = far_residuals / far_m_error

far_gauss_xs = np.linspace(-5, 5, 1000)
far_norm_res_mean = np.mean(far_norm_residuals)
far_norm_res_std = np.std(far_norm_residuals)


#######################################################################################################################################


# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


# check all this!!!
chi_squared_min_LH_0 = chi_squared(popt1, find_milestone_value, near_redshift, near_m_effective, near_m_error)
print('Value min chi^2 = {}'.format(chi_squared_min_LH_0))
degrees_of_freedom_value = near_redshift.size - popt1.size
print('Value reduced chi^2 = {}'.format(chi_squared_min_LH_0/degrees_of_freedom_value))
print('Value DoF = {}'.format(degrees_of_freedom_value))
print(chi_squared(popt1+popt1_err, find_milestone_value, near_redshift, near_m_effective, near_m_error))
print(chi_squared(popt1-popt1_err, find_milestone_value, near_redshift, near_m_effective, near_m_error))

#-------------------------------------------------------------------------------------------------------------------

chi_squared_min_Omega = chi_squared(popt2, find_theoretical_m_eff, far_redshift, far_m_effective, far_m_error)
print('Omega min chi^2 = {}'.format(chi_squared_min_Omega))
degrees_of_freedom_Omega = far_redshift.size - popt2.size
print('Omega reduced chi^2 = {}'.format(chi_squared_min_Omega/degrees_of_freedom_Omega))
print('Omega DoF = {}'.format(degrees_of_freedom_Omega))



#######################################################################################################################################


# make smooth curve for theoretical value
smooth_x = np.linspace(min(redshift), max(redshift), 1000)
ys = find_theoretical_m_eff(smooth_x, popt2)
far_smooth_x = np.linspace(min(far_redshift), max(far_redshift), 1000)
far_ys = find_theoretical_m_eff(far_smooth_x, popt2)

fig1, (ax1, ax2) = plt.subplots(2, 1, sharex='col', height_ratios=(4,1))
ax1.errorbar(far_redshift, far_m_effective, yerr=far_m_error, marker='o', color = 'r', elinewidth=0.8, linestyle='none', ms = 2)
ax1.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
ax1.plot(smooth_x, ys, color = 'g', linewidth = 0.8)
ax2.set_xlabel('$Redshift (z)$')
ax1.set_ylabel('$M_{eff}$')

#fig2, ax2 = plt.subplots()
ax2.scatter(far_redshift, far_norm_residuals, marker='.')
ax2.axhline(y=0, color='k', linestyle=':', alpha=0.25)
ax2.set_ylim([-5,5])


#fig3, ax3 = plt.subplots()
#ax3.hist(far_norm_residuals, bins = 12, density=True)
#ax3.axvline(x=1, color='k', linestyle=':', alpha=0.5)
#ax3.axvline(x=-1, color='k', linestyle=':', alpha=0.5)
#ax3.plot(gauss_xs, stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std))


#######################################################################################################################################
# Poster Plots

# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(far_redshift, far_m_effective, yerr=far_m_error, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
plt.plot(far_smooth_x, far_ys, color = 'r', linewidth = 0.8)
plt.ylabel('$M_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0.1, 0.86])

# Residuals
plt.figure(6).add_axes((0.1, 0.1, 0.74, 0.2))
plt.scatter(far_redshift, far_norm_residuals, color = 'r', marker='o', s=4)
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.xlabel('$Redshift (z)$')
plt.ylim([-5.5,5.5])
plt.xlim([0.1, 0.86])

# Residuals histrogram
plt.figure(6).add_axes((0.85, 0.1, 0.14, 0.2))
plt.hist(far_norm_residuals, bins = 12, orientation='horizontal', density=True, color='r')
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.plot(stats.norm.pdf(far_gauss_xs, far_norm_res_mean, far_norm_res_std), far_gauss_xs, color='k')
plt.tick_params(axis='y', left=False, right=False, labelleft=False)


#######################################################################################################################################
# Contour plots!!!

extent = 3.5 # standard errors
n_points = 100 # mesh density 

chi_squared_Omegas = np.linspace(popt2-popt2_err, popt2+popt2_err, num=n_points)
chi_squareds2 = []
for i in range(0, len(chi_squared_Omegas)):
    chi_squareds2.append(chi_squared(chi_squared_Omegas[i], find_theoretical_m_eff, far_redshift, far_m_effective, far_m_error))

fig4, ax4 = plt.subplots()
ax4.plot(chi_squared_Omegas, chi_squareds2)


chi_squared_L = np.linspace(popt1-popt1_err*extent, popt1+popt1_err*extent, num=n_points)
chi_squareds1 = []
for i in range(0, len(chi_squared_L)):
    chi_squareds1.append(chi_squared(chi_squared_L[i], find_milestone_value, near_redshift, near_m_effective, near_m_error))

fig5, ax5 = plt.subplots()
ax5.plot(chi_squared_L/H_0**2, chi_squareds1, color='r') # watch out as I've changed the axis to be L, not L*H_0^2
ax5.axvline(x=popt1/H_0**2, color='k', linestyle = 'dashed', alpha=0.5)
ax5.axvline(x=(popt1+popt1_err)/H_0**2, color='k', linestyle=':', alpha=0.3)
ax5.axvline(x=(popt1-popt1_err)/H_0**2, color='k', linestyle=':', alpha=0.3)
ax5.axhline(y=chi_squared_min_LH_0, color='k', linestyle='dashed', alpha=0.5)
ax5.axhline(y=chi_squared_min_LH_0+1, color='k', linestyle=':', alpha=0.3)
ax5.set_xlabel('$L_{\lambda, peak}\ [W\ \AA^{-1}] $')
ax5.set_ylabel('$\chi ^2$')


#######################################################################################################################################

plt.show()
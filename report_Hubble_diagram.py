import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats

c = 3*10**5 # km / s

file = 'Union2.1_data3.txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(0,1,2), unpack=True)

def LCDM_theoretical_m_eff(z, *params):
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((params[1]) * (1+z_grid)**3 + (1-params[1])))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = (1 + z) * c * I_value
    return 5 * np.log10(fracs) + 15 + params[0]

def wCDM_theoretical_m_eff(z, *params):
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((params[1]) * (1+z_grid)**3 + (1-params[1]) * (1+z_grid)**(3*(1+params[2]))))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = (1 + z) * c * I_value
    return 5 * np.log10(fracs) + 15 + params[0]

def oCDM_theoretical_m_eff(z, *params):
    Omegak = params[2]
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((params[1]) * (1+z_grid)**3 + params[2] * (1+z_grid)**2 + (1-params[1]-params[2])))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)
    
    if Omegak>0:
        d_Lish = (1+z) * c * np.sinh(np.sqrt(params[2]) * I_value) / (np.sqrt(params[2]))
    if Omegak==0:
        d_Lish = (1+z) * c * I_value 
    if Omegak<0:
        d_Lish = (1+z) * c * np.sin(np.sqrt(-Omegak) * I_value) / (np.sqrt(-Omegak))
    return 5*np.log10(d_Lish) + 15 + params[0]




smooth_x = np.linspace(min(z), max(z), 1000)
ys1 = LCDM_theoretical_m_eff(smooth_x, -18.547, 0.28)
ys3 = wCDM_theoretical_m_eff(smooth_x, -18.548, 0.28, -1.03)
ys2 = oCDM_theoretical_m_eff(smooth_x, -18.547, 0.28, 0.0005)

residuals1 = m_eff - LCDM_theoretical_m_eff(z, -18.547, 0.28)
norm_residuals1 = residuals1 / m_err

residuals3 = m_eff - wCDM_theoretical_m_eff(z, -18.548, 0.28, -1.03)
norm_residuals3 = residuals3 / m_err

residuals2 = m_eff - oCDM_theoretical_m_eff(z, -18.547, 0.28, 0.0005)
norm_residuals2 = residuals2 / m_err


residuals = m_eff - LCDM_theoretical_m_eff(z, -18.547, 0.28)
norm_residuals = residuals / m_err

maxlim = np.max(z) + 0.1
uplim = np.max(norm_residuals) + 0.5
lowlim = np.min(norm_residuals) - 0.5

gauss_xs = np.linspace(lowlim, uplim, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)



plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'cm'


fig6 = plt.figure(1).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
plt.plot(smooth_x, ys1, color = 'r', linewidth = 0.8)
#plt.plot(smooth_x, ys2, color = 'g', linewidth = 0.8)
#plt.plot(smooth_x, ys3, color = 'b', linewidth = 0.8)
plt.ylabel('$m_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, maxlim])

# Residuals
plt.figure(1).add_axes((0.1, 0.1, 0.74, 0.2))
plt.scatter(z, norm_residuals1, color = 'r', marker='.', s=4)
#plt.scatter(z, norm_residuals2, color = 'g', marker='.', s=4)
#plt.scatter(z, norm_residuals3, color = 'b', marker='.', s=4)
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.xlabel('Redshift $z$')
plt.ylabel('Normalised\n Residuals')
#plt.ylim([lowlim, uplim])
plt.xlim([0, maxlim])

# Residuals histrogram
plt.figure(1).add_axes((0.85, 0.1, 0.14, 0.2))
plt.hist(norm_residuals, bins = 12, orientation='horizontal', density=True, color='r')
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.plot(stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std), gauss_xs, color='k')
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
plt.xticks([])

plt.savefig('report_zplot.png', bbox_inches = 'tight', dpi=300)


plt.show()
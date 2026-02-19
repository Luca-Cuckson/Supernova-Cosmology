import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats
import corner
import emcee
import dataconstants as dac


# constants
f_lambda_0 = 6.61 * (10 ** (-12)) #W m^-2 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = (75*1000)/(3.09*10**22) #s^-1

# extract data from text file
file = 'sn_data(1).txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)


#######################################################################################################################################
# Find values

def find_theoretical_m_eff(z, *params):
    L = params[1] * 10**32
    H = params[2] * 10**(-18)
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((1-params[0]) * (1+z_grid)**3 + params[0]))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = L * H**2 / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + z) ** 2) * I_value**2)
    return -2.5 * np.log10(fracs)


# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


#######################################################################################################################################
# MCMC


#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, m_eff, m_err):
    return -0.5 * chi_squared(p0, find_theoretical_m_eff, z, m_eff, m_err)

def lnprior(theta):
    Omega, L, H_0 = theta
    if 0<Omega<1 and 0<L<10 and 0<H_0:
        a = -0.5 * ((Omega - dac.DE_mu) / dac.DE_sigma)**2 - np.log(dac.DE_sigma * np.sqrt(2*np.pi))
        b = -0.5 * ((H_0 - dac.H0_mu) / dac.H0_sigma)**2 - np.log(dac.H0_sigma * np.sqrt(2*np.pi))
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


#######################################################################################################################################
# Running the MCMC

npar = 3 #number of parameters
nsteps = 3000
p0 = np.array([0.68, 3.2016, H_0 * 10**18]) #chi-squared best-fit
nwalkers = 24
stepwidth = np.array([0.03, 0.0001, 0.05]) #hopefully can figure this one out
burnin = 300

starting_guesses = p0 + stepwidth * np.random.randn(nwalkers, npar) #have different starting pos. for each walker


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, m_eff, m_err))

pos, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)


#######################################################################################################################################
# Getting the results

samples = sampler.flatchain
print(samples)


chain = sampler.get_chain()
#print(chain)
Omega_chain = chain[:,:,0] #now have array of arrays of values for all walkers at each step
#print(Omega_chain)
#print(chain[:,:,1])

print(sampler.acceptance_fraction)

def get_values(chain):
    Omega_chain = chain[:,:,0]
    Omega_chain = Omega_chain[burnin:]
    L_chain = chain[:,:,1] * 10 ** 32
    L_chain = L_chain[burnin:]
    H_chain = chain[:,:,2] * 30.9
    H_chain = H_chain[burnin:]
    print('Omega_Lambda_0 = ({} \u00B1 {})'.format(np.mean(Omega_chain), np.std(Omega_chain)))
    print('L = ({} \u00B1 {})'.format(np.mean(L_chain), np.std(L_chain)))
    print('H_0 = ({} \u00B1 {})'.format(np.mean(H_chain), np.std(H_chain)))
    return np.mean(Omega_chain), np.mean(L_chain), np.mean(H_chain), np.std(Omega_chain), np.std(L_chain), np.std(H_chain)


#######################################################################################################################################
# Plotting the results


def walker_plot(all_walker_chains):
    fig, ax = plt.subplots(npar, 1, sharex=True)
    Omega_chain = all_walker_chains[:,:,0]
    L_chain = all_walker_chains[:,:,1] * 10 ** 32
    H_chain = all_walker_chains[:,:,2] * 30.9
    ax0 = ax[0]
    ax0.set_ylim(0,1)
    for i in range(nwalkers):
        path = Omega_chain[:,i]
        ax0.plot(path)
        ax0.set_ylabel("Omega_Lambda")
    ax1 = ax[1]
    for i in range(nwalkers):
        path = L_chain[:,i]
        ax1.plot(path)
        ax1.set_ylabel("L")
    ax2 = ax[2]
    for i in range(nwalkers):
        path = H_chain[:,i]
        ax2.plot(path)
        ax2.set_ylabel("H_0")


walker_plot(chain)


##################################


Omega_Lambda, L, H_0, Omega_Lambda_err, L_err, H_0_err = get_values(chain)

labels = ["$\Omega_{\Lambda}$", "$L_{\lambda, peak}\ [10^{32}\ W\ \AA^{-1}]$", "$H_0$ [$km\ s^{-1}\ Mpc^{-1}$]"]
means = [Omega_Lambda, L, H_0]
stds = [Omega_Lambda_err, L_err, H_0_err]



flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
#flat_samples[:,1] = flat_samples[:,1] * 10 ** 32
flat_samples[:,2] = flat_samples[:,2] * 30.9
print(flat_samples.shape)

figure = corner.corner(
    flat_samples,
    labels=labels,
    #quantiles=[0.16, 0.5, 0.84],
    #show_titles=True,
    title_kwargs={"fontsize": 12},
)

def get_values2(chain):
    Omega_chain = chain[:,:,0]
    Omega_chain = Omega_chain[burnin:]
    L_chain = chain[:,:,1]
    L_chain = L_chain[burnin:]
    H_chain = chain[:,:,2] * 30.9
    H_chain = H_chain[burnin:]
    return np.mean(Omega_chain), np.mean(L_chain), np.mean(H_chain)

axes = np.array(figure.axes).reshape((npar, npar))
value1 = get_values2(chain)
print(value1)

for i in range(npar):
    ax = axes[i, i]
    ax.axvline(value1[i], color="g")
    #ax.set_title(f"{labels[i]} = {means[i]:.3f} ± {stds[i]:.3f}", fontsize=12)


# Loop over the histograms
for yi in range(npar):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(value1[xi], color="g")
        ax.axhline(value1[yi], color="g")
        ax.plot(value1[xi], value1[yi], "sg")


#######################################################################################################################################
# Plotting Hubble Diagram

L, L_err = L * 10 ** (-32), L_err * 10 ** (-32)
H_0, H_0_err = H_0 / 30.9, H_0_err / 30.9

residuals = m_eff - find_theoretical_m_eff(z, Omega_Lambda, L, H_0)
norm_residuals = residuals / m_err

gauss_xs = np.linspace(-5, 5, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)

# make smooth curve for theoretical value
smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_m_eff(smooth_x, Omega_Lambda, L, H_0)

plt.rcParams["font.size"] = 18

# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
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
plt.ylim([-5.5,5.5])
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

plt.savefig('H0_zplot.svg', bbox_inches = 'tight')





plt.show()

tau = sampler.get_autocorr_time()
print(tau)
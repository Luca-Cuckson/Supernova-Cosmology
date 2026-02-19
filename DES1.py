import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats
import corner
import emcee
import dataconstants as dac

c = 3*10**5 # km / s
H_0init = 70 # km s Mpc^-1


file = 'Des_Data.txt'
z, mudif, mu_err, muref = np.loadtxt(file, usecols=(4,5,6,7), unpack=True)
mu = muref + mudif


#######################################################################################################################################
# Def Equations

def find_theoretical_mu(z, *params):
    H = params[1]
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((1-params[0]) * (1+z_grid)**3 + params[0]))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = (1 + z) * c * I_value / H
    return 5 * np.log10(fracs) + 25



# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!



#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, mu, mu_err):
    return -0.5 * chi_squared(p0, find_theoretical_mu, z, mu, mu_err)

def lnprior(theta):
    Omega, H = theta
    if 0<Omega<1 and 0<H:
        a = -0.5 * ((Omega - dac.DE_mu) / (dac.DE_sigma))**2 - np.log(dac.DE_sigma * np.sqrt(2*np.pi))
        b = -0.5 * ((H - dac.H0_mu0) / (dac.H0_sigma0))**2 - np.log(dac.H0_sigma0 * np.sqrt(2*np.pi))
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

npar = 2 #number of parameters
nsteps = 3000
p0 = np.array([0.68, H_0init]) #chi-squared best-fit
nwalkers = 24
stepwidth = np.array([0.06, 1]) #hopefully can figure this one out

starting_guesses = p0 + stepwidth * np.random.randn(nwalkers, npar) #have different starting pos. for each walker


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, mu, mu_err))

pos, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)

#######################################################################################################################################
# Get results

samples = sampler.flatchain
chain = sampler.get_chain()

def walker_plot(all_walker_chains):
    fig, ax = plt.subplots(npar, 1, sharex=True)
    Omega_chain = all_walker_chains[:,:,0]
    value_chain = all_walker_chains[:,:,1]
    ax0 = ax[0]
    ax0.set_ylim(0,1)
    for i in range(nwalkers):
        path = Omega_chain[:,i]
        ax0.plot(path)
        ax0.set_ylabel("Omega_Lambda")
    ax1 = ax[1]
    for i in range(nwalkers):
        path = value_chain[:,i]
        ax1.plot(path)
        ax1.set_ylabel("H_0")


walker_plot(chain)


def get_values(chain):
    Omega_chain = chain[:,:,0]
    value_chain = chain[:,:,1]
    print('Omega_Lambda_0 = ({} \u00B1 {})'.format(np.mean(Omega_chain), np.std(Omega_chain)))
    print('H_0 = ({} \u00B1 {})'.format(np.mean(value_chain), np.std(value_chain)))
    return np.mean(Omega_chain), np.mean(value_chain)



Omega_Lambda, H_0 = get_values(chain)

labels = ["$\Omega_{\Lambda}$", "$H_0$, $km\ s^{-1}\ Mpc^{-1}$"]


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

#fig = corner.corner(flat_samples, labels=labels, truths=get_values(chain))

figure = corner.corner(
    flat_samples,
    labels=labels,
    #quantiles=[0.16, 0.5, 0.84],
    #show_titles=True,
    title_kwargs={"fontsize": 12},
)

axes = np.array(figure.axes).reshape((npar, npar))
value1 = get_values(chain)
print(value1)

for i in range(npar):
    ax = axes[i, i]
    ax.axvline(value1[i], color="c")

# Loop over the histograms
for yi in range(npar):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(value1[xi], color="c")
        ax.axhline(value1[yi], color="c")
        ax.plot(value1[xi], value1[yi], "sc")


print(sampler.acceptance_fraction)

#######################################################################################################################################
# Plot


smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_mu(smooth_x, Omega_Lambda, H_0)

residuals = mu - find_theoretical_mu(z, Omega_Lambda, H_0)
norm_residuals = residuals / mu_err

gauss_xs = np.linspace(min(norm_residuals)-0.5, max(norm_residuals)+0.5, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)

fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, mu, yerr=mu_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('Distance Modulus, $\mu$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, 0.86])

# Residuals
plt.figure(6).add_axes((0.1, 0.1, 0.74, 0.2))
plt.scatter(z, norm_residuals, color = 'r', marker='o', s=4)
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.xlabel('$Redshift (z)$')
plt.ylabel('Normalised \n Residuals')
plt.ylim([min(norm_residuals)-0.5,max(norm_residuals)+0.5])
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

plt.savefig('DES1_zplot.svg', bbox_inches = 'tight')

plt.show()



tau = sampler.get_autocorr_time()
print(tau)
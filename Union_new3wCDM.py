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
MBinit = -19.321

# extract data from text file
file = 'Union2.1_data3.txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(0,1,2), unpack=True)
#z, m_eff, m_err = z[551:], m_eff[551:], m_err[551:]


#######################################################################################################################################
# Find theoretical values

def find_theoretical_m_eff(z, *params):
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((params[1]) * (1+z_grid)**3 + (1-params[1]) * (1+z_grid)**(3*(1+params[2]))))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = (1 + z) * c * I_value
    return 5 * np.log10(fracs) + 15 + params[0]


# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


#######################################################################################################################################
# MCMC


#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, m_eff, m_err):
    return -0.5 * chi_squared(p0, find_theoretical_m_eff, z, m_eff, m_err)

def lnprior(theta):
    FunkyM, Omega, w  = theta
    if 0<Omega<1 and -30<FunkyM<-10 and -2<w<0:
#        a = -0.5 * ((Omega - dac.DE_mu) / dac.DE_sigma)**2 - np.log(dac.DE_sigma * np.sqrt(2*np.pi))
#        b = -0.5 * ((H_0 - dac.H0_mu0) / dac.H0_sigma0)**2 - np.log(dac.H0_sigma0 * np.sqrt(2*np.pi))
#        c = -0.5 * ((MB - dac.MB_mu) / dac.MB_sigma)**2 - np.log(dac.MB_sigma * np.sqrt(2*np.pi))
        f = -0.5 * ((w - dac.w_mu) / dac.w_sigma)**2 - np.log(dac.w_sigma * np.sqrt(2*np.pi))
        return f
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
nsteps = 4000
p0 = np.array([-18.5, 0.3, -1]) #chi-squared best-fit
nwalkers = 24
stepwidth = np.array([0.03, 0.06, 0.06]) #hopefully can figure this one out
burnin = 300

starting_guesses = p0 + stepwidth * np.random.randn(nwalkers, npar) #have different starting pos. for each walker


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, m_eff, m_err))

pos, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)


#######################################################################################################################################
# Getting the results

samples = sampler.flatchain
#print(samples)


chain = sampler.get_chain()
Omega_chain = chain[:,:,1] #now have array of arrays of values for all walkers at each step

print(sampler.acceptance_fraction)

def get_values(chain):
    Omega_chain = chain[:,:,1]
    Omega_chain = Omega_chain[burnin:]
    FunkyM_chain = chain[:,:,0]
    FunkyM_chain = FunkyM_chain[burnin:]
    w_chain = chain[:,:,2]
    w_chain = w_chain[burnin:]
    print('Omega_m_0 = ({} \u00B1 {})'.format(np.mean(Omega_chain), np.std(Omega_chain)))
    print('FunkyM = ({} \u00B1 {})'.format(np.mean(FunkyM_chain), np.std(FunkyM_chain)))
    print('w = ({} \u00B1 {})'.format(np.mean(w_chain), np.std(w_chain)))
    return np.mean(FunkyM_chain), np.mean(Omega_chain), np.mean(w_chain), np.std(FunkyM_chain), np.std(Omega_chain), np.std(w_chain)

#######################################################################################################################################
# Plotting the results


def walker_plot(all_walker_chains):
    fig, ax = plt.subplots(npar, 1, sharex=True)
    Omega_chain = all_walker_chains[:,:,1]
    FunkyM_chain = all_walker_chains[:,:,0]
    w_chain = all_walker_chains[:,:,2]
    ax0 = ax[1]
    ax0.set_ylim(0,1)
    for i in range(nwalkers):
        path = Omega_chain[:,i]
        ax0.plot(path)
        ax0.set_ylabel("$Omega_{m}$")
    ax1 = ax[0]
    for i in range(nwalkers):
        path = FunkyM_chain[:,i]
        ax1.plot(path)
        ax1.set_ylabel("$M_B - 5log(h)$")
    ax2 = ax[2]
    for i in range(nwalkers):
        path = w_chain[:,i]
        ax2.plot(path)
        ax2.set_ylabel("w")
    plt.savefig('Union2.3_walkerplot.svg', bbox_inches = 'tight')


walker_plot(chain)


##################################


FunkyM, Omega_Lambda, w, FunkyM_err, Omega_Lambda_err, w_err = get_values(chain)

labels = ["$M_B - 5log(h)$", "$\\Omega_{m}$", "$w$"]
means = [FunkyM, Omega_Lambda, w]
stds = [FunkyM_err, Omega_Lambda_err, w_err]



flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
print(flat_samples.shape)
np.savetxt('wCDM_prior.txt', flat_samples)

figure = corner.corner(
    flat_samples,
    labels=labels,
    #quantiles=[0.16, 0.5, 0.84],
    #show_titles=True,
    title_kwargs={"fontsize": 12},
)

def get_values2(chain):
    Omega_chain = chain[:,:,1]
    Omega_chain = Omega_chain[burnin:]
    FunkyM_chain = chain[:,:,0]
    FunkyM_chain = FunkyM_chain[burnin:]
    w_chain = chain[:,:,2]
    w_chain = w_chain[burnin:]
    return np.mean(FunkyM_chain), np.mean(Omega_chain), np.mean(w_chain)


axes = np.array(figure.axes).reshape((npar, npar))
value1 = get_values2(chain)
print('This is a check:', value1) #this was a check
chi2 = chi_squared(value1, find_theoretical_m_eff, z, m_eff, m_err)
chi2_reduced = chi2 / (len(m_eff) - npar)
print('Value $\chi^2$ = {}'.format(chi2))
print('Value Reduced $\chi^2$ = {}'.format(chi2_reduced))


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

plt.savefig('Union2.3_cornerplot.svg', bbox_inches = 'tight')


#######################################################################################################################################
# Plotting Hubble Diagram

residuals = m_eff - find_theoretical_m_eff(z, FunkyM, Omega_Lambda, w)
norm_residuals = residuals / m_err
#####
maxlim = np.max(z) + 0.1
uplim = np.max(norm_residuals) + 0.15
lowlim = np.min(norm_residuals) - 0.15
#####
gauss_xs = np.linspace(lowlim, uplim, 1000)
norm_res_mean = np.mean(norm_residuals)
norm_res_std = np.std(norm_residuals)

# make smooth curve for theoretical value
smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_m_eff(smooth_x, FunkyM, Omega_Lambda, w)

plt.rcParams["font.size"] = 18

# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('$m_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, maxlim])

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
plt.ylim([lowlim, uplim])
plt.xlim([0, maxlim])

# Residuals histrogram
plt.figure(6).add_axes((0.85, 0.1, 0.14, 0.2))
plt.hist(norm_residuals, bins = 12, orientation='horizontal', density=True, color='r')
plt.axhline(y=0, color='k', linestyle = 'dashed', alpha=0.5)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.3)
plt.axhline(y=-1, color='k', linestyle=':', alpha=0.3)
plt.plot(stats.norm.pdf(gauss_xs, norm_res_mean, norm_res_std), gauss_xs, color='k')
plt.tick_params(axis='y', left=False, right=False, labelleft=False)
plt.xticks([0.2, 0.4])

plt.savefig('Union2.3_zplot.svg', bbox_inches = 'tight')





plt.show()

tau = sampler.get_autocorr_time()
print(tau)
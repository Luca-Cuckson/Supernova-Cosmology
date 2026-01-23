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

def find_theoretical_m_eff(z, *params):
    def integrand(x):
        return 1 / (np.sqrt((1 - params[0]) * (1+x)**3 + params[0]))
    I = []
    for i in range(0, len(z)):
        I.append(integrate.quad(integrand, 0, z[i]))
    I = np.array(I)
    I_value = I[:, 0]
    fracs = np.empty(len(z))
    for i in range(0, len(z)):
        fracs[i] = params[1] / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + float(z[i])) ** 2) * float(I_value[i])**2)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff

# chi-squared time!!!!!
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2) # Note the `*model_params' here!


#######################################################################################################################################
# MCMC


#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, m_eff, m_err):
    return -0.5 * chi_squared(p0, find_theoretical_m_eff, z, m_eff, m_err)

def lnprior(theta):
    Omega, LHH = theta
    if 0<Omega<1 and 0<LHH:
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
p0 = np.array([0.68, 0.0019]) #chi-squared best-fit
nwalkers = 100
stepwidth = np.array([0.06, 0.00007]) #hopefully can figure this one out

starting_guesses = p0 + stepwidth * np.random.randn(nwalkers, npar) #have different starting pos. for each walker


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, m_eff, m_err))

pos, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)


#######################################################################################################################################
# Plotting the results

samples = sampler.flatchain
print(samples)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
chain = sampler.get_chain()
print(chain)
Omega_chain = chain[:,:,0] #now have array of arrays of values for all walkers at each step
print(Omega_chain)
print(chain[:,:,1])

def walker_plot(all_walker_chains):
    fig, ax = plt.subplots(1, npar)
    Omega_chain = all_walker_chains[:,:,0]
    value_chain = all_walker_chains[:,:,1]
    ax = axes[0]
    ax.set_ylim(0.2,1)
    for i in range(nwalkers):
        path = Omega_chain[:,i]
        ax.plot(path)
        ax.set_ylabel("Omega_Lambda")
    ax = axes[1]
    for i in range(nwalkers):
        path = value_chain[:,i]
        ax.plot(path)
        ax.set_ylabel("LHH")


walker_plot(chain)

fig1, ax1 = plt.subplots(npar,1)
ax1 = axes[0]
ax1.hist(Omega_chain)

#def get_walkers_means(chain):
#    Omega_chain = chain[:,:,0]
#    value_chain = chain[:,:,1]
#    means = np.empty((nwalkers,npar))
#    for i in range(nwalkers):
#        Omega_path = Omega_chain[:,i]
#        value_path = value_chain[:,i]
#        means[i,0] = np.mean(Omega_path)
#        means[i,1] = np.mean(value_path)
#    return means

#print(get_walkers_means(chain))


def get_values(chain):
    Omega_chain = chain[:,:,0]
    value_chain = chain[:,:,1]
    print('Omega_Lambda_0 = ({} \u00B1 {})'.format(np.mean(Omega_chain), np.std(Omega_chain)))
    print('LHH = ({} \u00B1 {})'.format(np.mean(value_chain), np.std(value_chain)))
    return np.mean(Omega_chain), np.mean(value_chain)



Omega_Lambda, LHH = get_values(chain)


smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_m_eff(smooth_x, Omega_Lambda, LHH)

fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 3)
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='s', color = 'orange', elinewidth=0.8, linestyle='none', ms = 2)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('$M_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, 0.86])






#fig, ax = plt.subplots(1,2,figsize=(16, 8))
labels = ["Omega_Lambda", "LHH"]
#for i in range(npar):
#    ax = axes[i]
#    ax.plot(chain[:,:, i], "k", alpha=0.3)
#    ax.set_xlim(0, len(chain))
#    ax.set_ylabel(labels[i])
#    ax.yaxis.set_label_coords(-0.1, 0.5)

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples, labels=labels)

#from IPython.display import display, Math

#for i in range(npar):
#    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#    q = np.diff(mcmc)
#    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
#    display(Math(txt))




plt.show()






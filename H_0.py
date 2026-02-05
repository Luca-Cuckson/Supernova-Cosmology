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
# MCMC


#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, m_eff, m_err):
    return -0.5 * chi_squared(p0, find_theoretical_m_eff, z, m_eff, m_err)

def lnprior(theta):
    Omega, L, H_0 = theta
    if 0<Omega<1 and 0<L<10 and 0<H_0:
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
nsteps = 20000
p0 = np.array([0.68, 3.2016, H_0 * 10**18]) #chi-squared best-fit
nwalkers = 36
stepwidth = np.array([0.03, 0.05, 0.05]) #hopefully can figure this one out

starting_guesses = p0 + stepwidth * np.random.randn(nwalkers, npar) #have different starting pos. for each walker


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, m_eff, m_err))

pos, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)


#######################################################################################################################################
# Getting the results

samples = sampler.flatchain
print(samples)

fig, axes = plt.subplots(npar, figsize=(10, 7), sharex=True)
chain = sampler.get_chain()
print(chain)
Omega_chain = chain[:,:,0] #now have array of arrays of values for all walkers at each step
print(Omega_chain)
print(chain[:,:,1])

print(sampler.acceptance_fraction)

def get_values(chain):
    Omega_chain = chain[:,:,0]
    L_chain = chain[:,:,1]
    H_chain = chain[:,:,2]
    print('Omega_Lambda_0 = ({} \u00B1 {})'.format(np.mean(Omega_chain), np.std(Omega_chain)))
    print('L = ({} \u00B1 {})'.format(np.mean(L_chain), np.std(L_chain)))
    print('H_0 = ({} \u00B1 {})'.format(np.mean(H_chain), np.std(H_chain)))
    return np.mean(Omega_chain), np.mean(L_chain), np.mean(H_chain)


#######################################################################################################################################
# Plotting the results


def walker_plot(all_walker_chains):
    fig, ax = plt.subplots(1, npar)
    Omega_chain = all_walker_chains[:,:,0]
    L_chain = all_walker_chains[:,:,1]
    H_chain = all_walker_chains[:,:,2]
    ax = axes[0]
    ax.set_ylim(0.2,1)
    for i in range(nwalkers):
        path = Omega_chain[:,i]
        ax.plot(path)
        ax.set_ylabel("Omega_Lambda")
    ax = axes[1]
    for i in range(nwalkers):
        path = L_chain[:,i]
        ax.plot(path)
        ax.set_ylabel("L")
    ax = axes[2]
    for i in range(nwalkers):
        path = H_chain[:,i]
        ax.plot(path)
        ax.set_ylabel("H_0")


walker_plot(chain)

plt.show()

##################################


Omega_Lambda, L, H_0 = get_values(chain)



labels = ["Omega_Lambda", "L", "H_0"]

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

fig = corner.corner(flat_samples, labels=labels)





plt.show()
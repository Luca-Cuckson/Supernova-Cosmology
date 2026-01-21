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

#number of parameters
npar = 2
#number of steps
nsteps = 2000
p0 = np.array([0.68, 0.0019])
nwalkers = 11


stepwidth = np.array([0.06, 0.00007]) #hopefully can figure this one out

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


#starting_guesses = np.random.randn(nwalkers, npar)
starting_guesses = np.array([0.6, 0.0001] * nwalkers)


sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=(z, m_eff, m_err))

sampler.run_mcmc(starting_guesses, 5000, progress=True)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
chain = sampler.get_chain()


fig, ax = plt.subplots(1,2,figsize=(16, 8))
ax[0].set_ylabel('Omega_Lambda')
ax[0].set_xlabel('Step')
ax[1].set_ylabel('L_lambda,peak * H_0^2')
ax[1].set_xlabel('Step')
ax[0].plot(chain[:,0]);
ax[1].plot(chain[:,1]);


bchain = chain[200:,:] #let's set burn-in to 100 for now

fig, ax = plt.subplots(1,2,figsize=(16, 8))
ax[0].set_ylabel('Number of samples')
ax[0].set_xlabel('Omega_Lambda')
ax[1].set_ylabel('Number of samples')
ax[1].set_xlabel('L_lambda,peak * H_0^2')
ax[0].hist(bchain[:,0], bins = 30);
ax[1].hist(bchain[:,1], bins = 30);
print('Omega_Lambda: ',round(np.median(bchain[:,0]), 3), '+/-', round(np.std(bchain[:,0]),3))
print('Omega_Lambda: ',round(np.mean(bchain[:,0]), 3), '+/-', round(np.std(bchain[:,0]),3))
print('L_lambda,peak * H_0^2: ',round(np.median(bchain[:,1]), 6), '+/-', round(np.std(bchain[:,1]),6))
print('L_lambda,peak * H_0^2: ',round(np.mean(bchain[:,1]), 6), '+/-', round(np.std(bchain[:,1]),6))



fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(bchain[:,0], bchain[:,1], 'o',alpha = .1, markersize = 10, markeredgewidth=0, label= 'MCMC Samples')
ax.set_ylabel('L_lambda,peak * H_0^2')
ax.set_xlabel('Omega_Lambda')
l=ax.legend()


import corner
#corner.corner(bchain,labels=['Omega_Lambda', 'L_lambda,peak * H_0^2'])
plt.show()






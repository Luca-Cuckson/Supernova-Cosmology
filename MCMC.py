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
z, m_eff, m_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)

# taking only low redshift data
nz = z[42:60]
nm_eff = m_eff[42:60]
nm_err = m_err[42:60]
fz = z[0:42]
fm_eff = m_eff[0:42]
fm_err = m_err[0:42]


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
nsteps = 5000

#create arrays to store walker locations and likelihoods
chain = np.empty((nsteps, npar))
likelihoods = np.empty((nsteps, npar))
stepwidth = np.array([0.06, 0.00007]) #hopefully can figure this one out

#find log of likelihood, p0 is initial parameters
def log_likelihood(p0, z, m_eff, m_err):
    return -0.5 * chi_squared(p0, find_theoretical_m_eff, z, m_eff, m_err)



p0 = np.array([0.68, 0.0019])
all_log_likelihoods = np.empty(nsteps)
acceptance = np.empty(nsteps)
for i in range(nsteps): #h for here, l for last, t for trial, ll for log_likelihood, #p for position

    if i==0: #starting point is special
        hp = p0
        hll = log_likelihood(hp, z, m_eff, m_err)

    else: #from now on
        lp = chain[i-1,:] # get the previous position
        tp = lp + np.random.normal(0, 1, npar) * stepwidth # get trial position that's moved by stepwidth times by
                                                            #2D (for two parameters) normally distributed random numbers
        tll = log_likelihood(tp, z, m_eff, m_err)
        ll_ratio = np.exp(tll - lll) # ratio of trial ll to last ll, lll defined at end of loop (saved for each i)

        if ll_ratio >= np.random.uniform():
            # always move somewhere with better likelihood
            # sometimes move somewhere with worse likelihood
            hp = tp
            hll = tll
            acceptance[i] = 1

        else: #repeat same place (stay) if trial rejected
            hp = lp
            hll = lll
            acceptance[i] = 0

    chain[i,:] = hp #save current position in history of steps
    all_log_likelihoods[i] = hll #save current log_likelihood in case handy later? tutorial says?
    lll = hll #set previous log_likelihood for next step to allow loop to work :)

print(np.average(acceptance))
#######################################################################################################################################
# Let's see what's up!!


#currently tutorial code, change it to what you like in a bit. just really wanna see if works
fig, ax = plt.subplots(1,2,figsize=(16, 8))
ax[0].set_ylabel('Omega_Lambda')
ax[0].set_xlabel('Step')
ax[1].set_ylabel('L_lambda,peak * H_0^2')
ax[1].set_xlabel('Step')
ax[0].plot(chain[:,0]);
ax[1].plot(chain[:,1]);


bchain = chain[150:,:] #let's set burn-in to 100 for now

fig, ax = plt.subplots(1,2,figsize=(16, 8))
ax[0].set_ylabel('Number of samples')
ax[0].set_xlabel('Omega_Lambda')
ax[1].set_ylabel('Number of samples')
ax[1].set_xlabel('L_lambda,peak * H_0^2')
ax[0].hist(bchain[:,0], bins = 30);
ax[1].hist(bchain[:,1], bins = 30);
print('Omega_Lambda: ',round(np.median(bchain[:,0]), 3), '+/-', round(np.std(bchain[:,0]),3))
print('L_lambda,peak * H_0^2: ',round(np.median(bchain[:,1]), 6), '+/-', round(np.std(bchain[:,1]),6))



fig, ax = plt.subplots(figsize=(16, 8))
plt.plot(bchain[:,0], bchain[:,1], 'o',alpha = .1, markersize = 10, markeredgewidth=0, label= 'MCMC Samples')
ax.set_ylabel('L_lambda,peak * H_0^2')
ax.set_xlabel('Omega_Lambda')
l=ax.legend()


import corner
corner.corner(bchain,labels=['Omega_Lambda', 'L_lambda,peak * H_0^2'])
plt.show()




best = np.argmax(all_log_likelihoods)
bestpar = chain[best,:]

smooth_x = np.linspace(min(z), max(z), 1000)
ys = find_theoretical_m_eff(smooth_x, *bestpar)

plt.rcParams["font.size"] = 16
# Main plot
fig6 = plt.figure(6).add_axes((0.1,0.32,0.74,0.68))
plt.errorbar(z, m_eff, yerr=m_err, marker='o', color = 'k', elinewidth=1, ecolor='gray', linestyle='none', ms = 2)
plt.plot(smooth_x, ys, color = 'r', linewidth = 0.8)
plt.ylabel('$M_{eff}$')
plt.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
plt.xlim([0, 0.86])



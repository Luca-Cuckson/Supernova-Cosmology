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


def new_m_effs(z, *params):
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((1-params[0]) * (1+z_grid)**3 + params[0]))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = params[1] / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + z) ** 2) * I_value**2)
    return -2.5 * np.log10(fracs)

print(find_theoretical_m_eff(z, *[0.6818, 0.001886]))
print(new_m_effs(z, *[0.6818, 0.001886]))

print(new_m_effs(z, *[0.6818, 0.001886]) - find_theoretical_m_eff(z, *[0.6818, 0.001886]))
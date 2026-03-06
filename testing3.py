import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats

H = 69.6
c = 3*10**5 # km / s

def find_theoretical_m_eff(z, *params):
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
    return 5*np.log10(d_Lish) + 25 + params[0]



print(find_theoretical_m_eff(1, -29, 0.286, 0))
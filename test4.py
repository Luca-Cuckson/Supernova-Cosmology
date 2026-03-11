import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate
import scipy.stats as stats

c = 3*10**5 # km / s



def find_theoretical_m_eff(z, *params):
    z_grid = np.linspace(0, np.max(z), 2000)
    integrand = 1 / (np.sqrt((params[1]) * (1+z_grid)**3 + (1-params[1])))
    Integrals = integrate.cumulative_trapezoid(integrand, z_grid, initial=0)
    I_interp = scipy.interpolate.make_interp_spline(z_grid, Integrals, k=3)
    I_value = I_interp(z)

    fracs = (1 + z) * c * I_value
    return 5 * np.log10(fracs) + 15 + params[0]


def test_model(z, M, Omega):
    def integrand(x):
        return 1 / np.sqrt(Omega * (1 + x)**3 + (1 - Omega))

    I = []
    for i in range(len(z)):
        I.append(integrate.quad(integrand, 0, z[i]))

    I = np.array(I)
    I_value = I[:, 0]
    fracs = (1 + z) * c * I_value
    return 5 * np.log10(fracs) + 15 + M



z = np.array([0.015, 1, 1.5])
M, Omega = np.array([-18.547, 0.28])

print(find_theoretical_m_eff(z, M, Omega))
print(test_model(z, M, Omega))
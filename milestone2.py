import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate

# constants
f_lambda_0 = 6.61 * (10 ** (-12)) #W m^-2 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = (75*1000)/(3.09*10**22) #km s^-1 Mpc^-1 


# extract data from text file
file = 'sn_data(1).txt'
redshift, m_effective, m_error = np.loadtxt(file, usecols=(1,2,3), unpack=True)

# taking only low redshift data
near_redshift = redshift[42:60]
near_m_effective = m_effective[42:60]
near_m_error = m_error[42:60]
far_redshift = redshift[0:42]
far_m_effective = m_effective[0:42]
far_m_error = m_error[0:42]

# find d_L for nearby supernovae
def get_near_d_L(z):
    return c * z / (H_0*1000) # output in Mpc

def find_milestone_value(redshift, value):
    redshift = np.array(redshift)
    frac = value / (f_lambda_0 * 4 * np.pi * c**2 * redshift**2)
    m_eff = -2.5 * np.log10(frac)
    return m_eff

print(find_milestone_value(near_redshift, 0.0017492))

popt1, cov1 = scipy.optimize.curve_fit(find_milestone_value, near_redshift, near_m_effective, sigma = near_m_error, absolute_sigma=True, p0=0.0016378, check_finite=True)

print(popt1)
value = float(popt1[0])


def find_Omega_Lambda(redshift, Omega):
    def integrand(x):
        return 1 / (np.sqrt((1 - Omega) * (1+x)**3 + Omega))
    I = []
    for i in range(0, len(redshift)):
        I.append(integrate.quad(integrand, 0, redshift[i]))
    I = np.array(I)
    I_value = I[:, 0]
    fracs = np.empty(len(redshift))
    for i in range(0, len(redshift)):
        fracs[i] = value / ((c**2) * (f_lambda_0) * 4 * np.pi * ((1 + float(redshift[i])) ** 2) * float(I_value[i])**2)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff


popt2, cov2 = scipy.optimize.curve_fit(find_Omega_Lambda, far_redshift, far_m_effective, sigma = far_m_error, absolute_sigma=True, p0=np.array([0.685]), check_finite=True)

print(popt2)






smooth_x = np.linspace(min(redshift), max(redshift), 1000)
ys = find_Omega_Lambda(smooth_x, popt2)

fig3, ax3 = plt.subplots()
ax3.errorbar(redshift, m_effective, yerr=m_error, marker='.', linestyle='none')
ax3.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='.', linestyle='none')
ax3.plot(smooth_x, ys, color = 'g', linewidth = 0.5)

plt.show()
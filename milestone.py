import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize 
import scipy.integrate as integrate

# constants
f_lambda_0 = 6.61 * (10 ** (-9)) #erg cm^-2 s^-1 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = 75 #km s^-1 Mpc^-1 


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

# get f_lambda from magnitude
def get_band_pass(m_eff, z): # input effective magnitudes and redshifts
    return (10 ** (-0.4 * m_eff) * f_lambda_0) / (1 + z) # output in erg cm^-2 s^-1 A^-1

# find d_L for nearby supernovae
def get_near_d_L(z):
    return c * z / (H_0*1000) # output in Mpc

# find L_lambda_peak
def get_peak_L(d_L, f_lambda, z):
    return 4 * np.pi * ((d_L * 3.09 * 10**24)**2) * (1 + z) * (f_lambda * 10**(-7)) # output W A^-1

def weighted_mean(x, sigma):
    w = 1 / (sigma**2)
    mean = np.sum(x * w) / np.sum(w)
    std = np.std(x) # double check this is correct standard deviation (population vs sample)
    N = len(x)
    error = std / np.sqrt(N)
    return np.array([mean, error])


#Omega_Lambda = 0.685
#def integrand(x):
#    return 1 / (np.sqrt((1 - Omega_Lambda) * (1+x)**3 + Omega_Lambda))

#I = np.array(integrate.quad(integrand, 0, near_redshift[0])) * c / (H_0 * 1000)
#print(I)

band_passes = get_band_pass(near_m_effective, near_redshift)
band_passes_err = band_passes - get_band_pass(near_m_effective + near_m_error, near_redshift)
near_distances = get_near_d_L(near_redshift)
peak_Ls = get_peak_L(near_distances, band_passes, near_redshift)
peak_Ls_err = get_peak_L(near_distances, band_passes + band_passes_err, near_redshift) - peak_Ls

peak_L = weighted_mean(peak_Ls, peak_Ls_err) # W A^-1 ?
print(peak_L)

peak_L = [3e+32, 1e+31] # just to see


def model_function(redshift, Omega):  ### Is currently ignoring the errors in the theoretical value!!! Dangerous!!!
    #params = 0.685
    def integrand(x):
        return 1 / (np.sqrt((1 - Omega) * (1+x)**3 + Omega))
    fracs = []
    I = []
    for i in range(0, len(redshift)):
        I.append(integrate.quad(integrand, 0, redshift[i]))
    I = np.array(I)
    I_value = I[:, 0]
    for i in range(0, len(redshift)):
        frac = (peak_L[0] * ((H_0*1000 / (3.09 * 10**22)) ** 2)) / ((c**2) * (f_lambda_0 / 1000) * 4 * np.pi * ((1 + redshift[i]) ** 2) * I_value[i]**2)
        fracs.append(frac)
    m_eff = -2.5 * np.log10(np.array(fracs))
    return m_eff

popt, cov = scipy.optimize.curve_fit(model_function, far_redshift, far_m_effective, sigma = far_m_error, absolute_sigma=True, p0=np.array([0.685]), check_finite=True)

print(popt)

print(model_function(far_redshift, 0.685))


smooth_x = np.linspace(min(redshift), max(redshift), 1000)
ys = model_function(smooth_x, popt)




#a = np.average(peak_Ls)
#def model_L_peak(x, *params):
#    return params[0]
#popt, cov = scipy.optimize.curve_fit(model_L_peak, near_redshift, peak_Ls, sigma=peak_Ls_err, absolute_sigma=True, p0=a, check_finite=True)


fig1, ax1 = plt.subplots()
ax1.errorbar(near_redshift, band_passes, yerr=band_passes_err, marker='.', linestyle='none')

fig2, ax2 = plt.subplots()
ax2.errorbar(near_redshift, peak_Ls, yerr = peak_Ls_err, marker='.', linestyle='none')
ax2.set_ylim([0, 1.2 * np.max(peak_Ls)])
ax2.axhline(peak_L[0])

fig3, ax3 = plt.subplots()
ax3.errorbar(redshift, m_effective, yerr=m_error, marker='.', linestyle='none')
ax3.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='.', linestyle='none')
ax3.plot(smooth_x, ys, color = 'g', linewidth = 0.5)

# checks
#print(redshift)
#print(m_effective)
#print(near_redshift)
#print(near_m_effective)
#print(get_band_pass(near_m_effective, near_redshift))
#print(band_passes_err)
#print(get_near_d_L(near_redshift))
#print(near_distances)

# plot
#plt.errorbar(redshift, m_effective, yerr=m_error, marker='.', linestyle='none')
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='.', linestyle='none')
plt.show()
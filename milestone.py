import numpy as np
import matplotlib.pyplot as plt

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

# get f_lambda from magnitude
def get_band_pass(m_eff, z): # input effective magnitudes and redshifts
    return (10 ** (-0.4 * m_eff) * f_lambda_0) / (1 + z) # output in erg cm^-2 s^-1 A^-1

# find d_L for nearby supernovae
def get_near_d_L(z):
    return c * z / (H_0*1000) # output in Mpc

# find L_lambda_peak
def get_peak_L(d_L, f_lambda, z):
    return 4 * np.pi * (d_L**2) * (1 + z) * f_lambda

band_passes = get_band_pass(near_m_effective, near_redshift)
distances = get_near_d_L(near_redshift)
peak_Ls = get_peak_L(distances, band_passes, near_redshift)
plt.scatter(near_redshift, peak_Ls)
plt.gca().set_ylim([0, 1.2 * np.max(peak_Ls)])

# checks
#print(redshift)
#print(m_effective)
print(near_redshift)
#print(near_m_effective)
print(get_band_pass(near_m_effective, near_redshift))
print(get_near_d_L(near_redshift))

# plot
#plt.errorbar(redshift, m_effective, yerr=m_error, marker='.', linestyle='none')
#plt.errorbar(near_redshift, near_m_effective, yerr = near_m_error, marker='.', linestyle='none')
plt.show()
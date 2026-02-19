import numpy as np

# constants
f_lambda_0 = 6.61 * (10 ** (-12)) #W m^-2 A^-1 
c = 3.00 * (10**8) #m s^-1
H_0 = (75*1000)/(3.09*10**22) #s^-1

# extract data from text file
file = 'sn_data(1).txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)


# CMB constraints
DE_mu = 0.6889
DE_sigma = 0.0056 * 10 
H0_mu = (67.66*1000)/(3.09*10**22) * 10**18
H0_sigma = (0.42*1000)/(3.09*10**22) * 10**18 * 10

H0_mu0 = 67.66
H0_sigma0 = 0.42 * 10

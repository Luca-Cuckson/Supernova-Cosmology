import numpy as np

file = 'Union2.1_data.txt'
z, mu, mu_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)

m_eff = mu - 19.321
m_err = np.sqrt(mu_err**2 + 0.03**2)

np.savetxt('Union2.1_data2.txt', np.transpose([z, m_eff, m_err]))
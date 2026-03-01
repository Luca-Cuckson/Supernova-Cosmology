import numpy as np


# Stuff to get m_eff from mu

#file = 'Union2.1_data.txt'
#z, mu, mu_err = np.loadtxt(file, usecols=(1,2,3), unpack=True)

#m_eff = mu - 19.321
#m_err = np.sqrt(mu_err**2 + 0.03**2)

#np.savetxt('Union2.1_data2.txt', np.transpose([z, m_eff, m_err]))

##################################################################################################
# Stuff to sort out redshifts
file = 'Union2.1_data2.txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(0,1,2), unpack=True)

#za = z[:10]

#a = []

#np.append(a, 6)
#print(a)
#print(za)


#x = [5, -6, 3, 11, -4, 2] 
#y = [1, 9, 2, 8, 12, -5]

#pairs = [(a, b) for a, b in zip(x, y) if 0 <= a <= 10]
#print(pairs)

triples1 = [(a, b, c) for a, b, c in zip(z, m_eff, m_err) if 0 <= a < 0.1]
#print(triples1)

triples2 = [(a, b, c) for a, b, c in zip(z, m_eff, m_err) if 0.1 <= a < 0.25]
triples3 = [(a, b, c) for a, b, c in zip(z, m_eff, m_err) if 0.25 <= a < 0.5]
triples4 = [(a, b, c) for a, b, c in zip(z, m_eff, m_err) if 0.5 <= a < 1]
triples5 = [(a, b, c) for a, b, c in zip(z, m_eff, m_err) if 1 <= a]



triples = np.append(triples1, triples2, axis=0)
triples = np.append(triples, triples3, axis=0)
triples = np.append(triples, triples4, axis=0)
triples = np.append(triples, triples5, axis=0)

#print(triples)
print(len(triples))
print(len(z))
print(len(triples1), len(triples2), len(triples3), len(triples4), len(triples5))



#np.savetxt('Union2.1_data3.txt', X=triples, header= "z, m_eff, m_err")

file = 'Union2.1_data3.txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(0,1,2), unpack=True)

#print(z[0:175])

#print(triples2)
#print(z[175:257])

#print(triples3)
#print(z[257:412])

#print(triples4)
#print(z[412:551])

print(triples5)
print(z[551:])
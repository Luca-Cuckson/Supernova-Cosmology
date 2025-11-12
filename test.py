print('hello world')
print("change")
print("I'm testing")

def square(x):
    return x ** 2

print(square(4))


import scipy.integrate as integrate
import numpy as np

Omega_Lambda = 0.685
def integrand(x):
    return 1 / (np.sqrt((1 - Omega_Lambda) * (1+x)**3 + Omega_Lambda))

c = 3 * (10**8)
H_0 = 75

I = integrate.quad(integrand, 0, 0.03)
print(I)

a = [0.03, 0.3, 0.4]
I = []
for i in range(0, len(a)):
    I.append((integrate.quad(integrand, 0, a[i])))
I = np.array(I)
frac = 2 * I[:, 0]
print(frac)



#Omega_Lambda = 0.685
#def integrand(x):
#    return 1 / (np.sqrt((1 - Omega_Lambda) * (1+x)**3 + Omega_Lambda))


#I = np.array(integrate.quad(integrand, 0, near_redshift[0])) * c / (H_0 * 1000)

#print(I)


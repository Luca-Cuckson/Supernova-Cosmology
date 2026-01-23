import numpy as np
import matplotlib.pyplot as plt

nwalkers = 5
nsteps = 3
npar = 2
a = np.empty((nsteps, nwalkers, npar))
for i in range(nsteps):
    a[i,:,0] = i
    a[i,:,1] = i+1


print(a) #is (nsteps, nwalkers, npar)

b = a[:,:, 0]
d = a[:,:,1]
print(b)
print(d)

c = a[0,:]
print(c)

print(np.mean(b))
print(np.std(b))
print(np.mean(d))
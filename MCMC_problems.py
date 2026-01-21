import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#def gauss(x):
#    return np.exp(-(x-2)**2 / 4)

def func(x):
    if 3<x<7:
        a = 0.25
    else:
        a = 0
    return a

#def prop_dist(x, y):
#    return np.exp(-(x-y)**2)

nsteps = 100000
chain = np.empty(nsteps)
p0 = 4

for i in range(nsteps):
    if i==0:
        chain[i]=p0
    
    else:
        prev = chain[i-1]
        prop = np.random.normal() + prev
        r = np.random.uniform()
        if func(prev) == 0:
            chain[i] = prev
        else:
            ratio = func(prop)/func(prev)
        
            if ratio > r:
                chain[i] = prop

            else:
                chain[i]=prev


gauss_xs = np.linspace(-3, 6, 1000)
xs = np.linspace(0,8,1000)
ys = np.ones(1000) * 0.25

fig1 = plt.figure(1).add_axes((0.1,0.1,0.8,0.8))
plt.hist(chain, bins=50, density=True)
#plt.plot(gauss_xs, stats.norm.pdf(gauss_xs, 2, np.sqrt(2)))
plt.plot(xs, ys)
plt.show()



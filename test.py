import numpy as np

print(np.ones((2,3,4)))

print(np.average(np.ones((2,3,4))))

def func(n):
    return np.ones((2,n))

def func2(n, m):
    thing = np.emty((n, m))
    for i in range(m):
        thing



a = [1,2,3,4,5,6,7,8,9]
b = a[4:]
print(b)


print(np.array(a)+3)

z = [0.01, 0.3, 0.4, 0.5]

minlim = np.min([np.min(z) - 0.1, 0])

print(minlim)

file = 'Union2.1_data3.txt'
z, m_eff, m_err = np.loadtxt(file, usecols=(0,1,2), unpack=True)

print(np.max(z))
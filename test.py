import numpy as np

print(np.ones((2,3,4)))

print(np.average(np.ones((2,3,4))))

def func(n):
    return np.ones((2,n))

def func2(n, m):
    thing = np.emty((n, m))
    for i in range(m):
        thing
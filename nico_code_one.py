import numpy as np

Lim = [[25,75],[20000,30000],[98,102],[0.15,0.25]]
print(len(Lim))
list = [np.random.uniform(Lim[idx][0],Lim[idx][1]) for idx in range(len(Lim))]
print(list)
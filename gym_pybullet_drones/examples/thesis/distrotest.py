import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(2)
norml1 = np.random.normal(size=5000000)
norml2 = np.random.normal(size=5000000)
norml3 = np.random.normal(size=5000000)
norml4 = np.random.normal(size=5000000)

exp = 4
plt.figure()
plt.subplot(221)
plt.hist(norml1[0:10**exp],500)
plt.subplot(222)
plt.hist(norml1[10**exp:2*10**exp],500)
plt.subplot(223)
plt.hist(norml1[2*10**exp:3*10**exp],500)
plt.subplot(224)
plt.hist(norml1[3*10**exp:4*10**exp],500)
plt.show()



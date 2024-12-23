import matplotlib.pyplot as plt
import numpy as np
dict = np.load("dump.npz")
print(dict.keys())

l_force = dict["l_force"]
l_orien = dict["l_orien"]

plt.plot(l_orien[:, 0])
plt.plot(l_orien[:, 1])
plt.plot(l_orien[:, 2])

#plt.plot(l_force[:, 0])
#plt.plot(l_force[:, 1])
#plt.plot(l_force[:, 2])
#plt.plot(l_force[:, 3])
#plt.plot(l_force[:, 4])
#plt.plot(l_force[:, 5])

plt.show()
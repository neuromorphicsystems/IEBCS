import numpy as np
import matplotlib.pyplot as plt


# This script is to compare the 2nd order and delayed 1st order response for Figure 2 in the paper.
# Not used in the simulator.

tau1 = 500 # us
tau2 = 200  # us

time = np.arange(0, 2000, 1)

rep = 1 - (tau1 / (tau1 - tau2) * np.exp(-time / tau1) - tau2 / (tau1 - tau2) * np.exp(-time / tau2))
rep2 = 1 - np.exp(-(time - 200) / tau1)
ind = np.where(rep2 < 0)
rep2[ind] = 0

plt.figure()
plt.plot(time, rep, label="2nd order")
plt.plot(time, rep2, label="Delayed 1st order")
plt.legend()
plt.xlabel(" time (us) ")
plt.ylabel(" Response ")
plt.show()


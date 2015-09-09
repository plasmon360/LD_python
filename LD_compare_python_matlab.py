from LD import LD # Make sure this file is visible to PYTHONPATH or keep it in the same directory of file which is trying to call it.
import numpy as np
import matplotlib.pyplot as plt
import os.path

lamda = np.linspace(200E-9,2000E-9,200) # Creates a wavelength vector from 300 nm to 1000 nm of length 100
gold = LD(lamda, material = 'Au',model = 'LD')

print gold.epsilon_real
print gold.epsilon_imag
print gold.n
print gold.k

data = np.loadtxt(os.path.join(os.path.dirname(__file__), r'LD_bora_ung_code_comparision\eps_Au_ld_bora.dat'))
f,ax = plt.subplots(nrows = 1, ncols = 2)

ax[0].plot(1E9*lamda, gold.epsilon_real, 'bs', label = 'LD.py')
ax[0].plot(1E9*data[:, 0], data[:, 1], '-r', label = 'LD.m')

ax[1].plot(1E9*lamda, gold.epsilon_imag, 'bs', label = 'LD.py')
ax[1].plot(1E9*data[:, 0], data[:, 2], '-r', label = 'LD.m')
plt.legend()
plt.show()
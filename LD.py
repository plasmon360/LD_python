"""
 This module calculates the real and imaginary part of the dielectric function,
 real and imaginary part of the refractive index for different metals using either
 Drude model (D) and Lorentz-Drude model (LD). The parameters are obtained from
 Rakic et al. This module is inspired by LD.m
 http://www.mathworks.com/matlabcentral/fileexchange/18040-drude-lorentz-and-debye-lorentz-models-for-the-dielectric-constant-of-metals-and-water

    Example:
    To use in other python files

    from LD import LD # Make sure the file is accessible to PYTHONPATH or in the same directory of file which is trying to import
    import numpy as np
    lamda = np.linspace(300E-9,1000E-9,100) # Creates a wavelength vector from 300 nm to 1000 nm of length 100
    gold = LD(lamda, material = 'Au',model = 'LD') # Creates gold object with dielectric function of LD model
    print gold.epsilon_real
    print gold.epsilon_imag
    print gold.n
    print gold.k
    gold.plot_epsilon()
    gold.plot_n_k()

%   INPUT PARAMETERS:
%
%       lambda   ==> wavelength (meters) of light excitation on material. Numpy array
%
%       material ==>    'Ag'  = silver
%                       'Al'  = aluminum
%                       'Au'  = gold
%                       'Cu'  = copper
%                       'Cr'  = chromium
%                       'Ni'  = nickel
%                       'W'   = tungsten
%                       'Ti'  = titanium
%                       'Be'  = beryllium
%                       'Pd'  = palladium
%                       'Pt'  = platinum
%
%       model    ==> Choose 'LD' or 'D' for Lorentz-Drude or Drude model.
%
%       Reference:
%       Rakic et al., Optical properties of metallic films for vertical-
%       cavity optoelectronic devices, Applied Optics (1998)

"""

import numpy as np


class LD():
    def __init__(self, lamda, material, model='LD'):


        self.lamda = lamda
        self.material = material
        self.model = model

        # ***********************************************************************
        # Physical constants
        #***********************************************************************
        twopic = 1.883651567308853e+09  # twopic=2*pi*c where c is speed of light
        omega_light = twopic / self.lamda;  # angular frequency of light (rad/s)
        invsqrt2 = 0.707106781186547  # 1/sqrt(2)
        ehbar = 1.519250349719305e+15  # e/hbar where hbar=h/(2*pi) and e=1.6e-19

        if self.material == 'Ag':
            # Plasma frequency
            omega_p = 9.01 * ehbar
            # Oscillators' strengh
            f = [0.845, 0.065, 0.124, 0.011, 0.840, 5.646]
            # Damping frequency of each oscillator
            Gamma = [0.048, 3.886, 0.452, 0.065, 0.916, 2.419]
            # Resonant frequency of each oscillator
            omega = [0.000, 0.816, 4.481, 8.185, 9.083, 20.29]
            # Number of resonances
        elif self.material == 'Al':
            omega_p = 14.98 * ehbar
            f = [0.523, 0.227, 0.050, 0.166, 0.030]
            Gamma = [0.047, 0.333, 0.312, 1.351, 3.382]
            omega = [0.000, 0.162, 1.544, 1.808, 3.473]
        elif self.material == 'Au':
            omega_p = 9.03 * ehbar
            f = [0.760, 0.024, 0.010, 0.071, 0.601, 4.384]
            Gamma = [0.053, 0.241, 0.345, 0.870, 2.494, 2.214]
            omega = [0.000, 0.415, 0.830, 2.969, 4.304, 13.32]
        elif self.material == 'Cu':
            omega_p = 10.83 * ehbar
            f = [0.575, 0.061, 0.104, 0.723, 0.638]
            Gamma = [0.030, 0.378, 1.056, 3.213, 4.305]
            omega = [0.000, 0.291, 2.957, 5.300, 11.18]
        elif self.material == 'Cr':
            omega_p = 10.75 * ehbar
            f = [0.168, 0.151, 0.150, 1.149, 0.825]
            Gamma = [0.047, 3.175, 1.305, 2.676, 1.335]
            omega = [0.000, 0.121, 0.543, 1.970, 8.775]
        elif self.material == 'Ni':
            omega_p = 15.92 * ehbar
            f = [0.096, 0.100, 0.135, 0.106, 0.729]
            Gamma = [0.048, 4.511, 1.334, 2.178, 6.292]
            omega = [0.000, 0.174, 0.582, 1.597, 6.089]
        elif self.material == 'W':
            omega_p = 13.22 * ehbar
            f = [0.206, 0.054, 0.166, 0.706, 2.590]
            Gamma = [0.064, 0.530, 1.281, 3.332, 5.836]
            omega = [0.000, 1.004, 1.917, 3.580, 7.498]
        elif self.material == 'Ti':
            omega_p = 7.29 * ehbar
            f = [0.148, 0.899, 0.393, 0.187, 0.001]
            Gamma = [0.082, 2.276, 2.518, 1.663, 1.762]
            omega = [0.000, 0.777, 1.545, 2.509, 1.943]
        elif self.material == 'Be':
            omega_p = 18.51 * ehbar
            f = [0.084, 0.031, 0.140, 0.530, 0.130]
            Gamma = [0.035, 1.664, 3.395, 4.454, 1.802]
            omega = [0.000, 0.100, 1.032, 3.183, 4.604]
        elif self.material == 'Pd':
            omega_p = 9.72 * ehbar
            f = [0.330, 0.649, 0.121, 0.638, 0.453]
            Gamma = [0.008, 2.950, 0.555, 4.621, 3.236]
            omega = [0.000, 0.336, 0.501, 1.659, 5.715]
        elif self.material == 'Pt':
            omega_p = 9.59 * ehbar
            f = [0.333, 0.191, 0.659, 0.547, 3.576]
            Gamma = [0.080, 0.517, 1.838, 3.668, 8.517]
            omega = [0.000, 0.780, 1.314, 3.141, 9.249]
        else:
            print('Not a Valid Material')

        order = len(omega)
        Gamma = [_ * ehbar for _ in Gamma]
        omega = [_ * ehbar for _ in omega]

        if self.model == 'D':

            epsilon_D = np.zeros(len(omega_light), dtype=complex)
            for i, w in enumerate(omega_light):
                epsilon_D[i] = 1 - (f[0] * omega_p ** 2 / (w ** 2 + 1j * (Gamma[0]) * w))
            self.epsilon = epsilon_D

        elif self.model == 'LD':

            epsilon_D = np.zeros(len(omega_light), dtype=complex)
            for i, w in enumerate(omega_light):
                epsilon_D[i] = 1 - (f[0] * omega_p ** 2 / (w ** 2 + 1j * (Gamma[0]) * w))
            epsilon_L = np.zeros(len(omega_light), dtype=complex)

            for i, w in enumerate(omega_light):
                for k in xrange(1, order):
                    epsilon_L[i] += (f[k] * omega_p ** 2) / (omega[k] ** 2 - w ** 2 - 1j * Gamma[k] * w)
            self.epsilon = epsilon_D + epsilon_L

        self.refractive_index = np.sqrt(self.epsilon)
        self.epsilon_real = self.epsilon.real
        self.epsilon_imag = self.epsilon.imag
        self.n = self.refractive_index.real
        self.k = self.refractive_index.imag

    def plot_epsilon(self):
        import matplotlib.pyplot as plt

        self.fig_eps, self.ax_eps = plt.subplots(1, 2, figsize=(15, 6))
        self.ax_eps[0].plot(1E9 * self.lamda, self.epsilon_real, '-o')
        self.ax_eps[0].set_xlabel('Wavelength(nm)')
        self.ax_eps[0].set_ylabel('Real (Epsilon)')

        self.ax_eps[1].plot(1E9 * self.lamda, self.epsilon_imag, '-s')
        self.ax_eps[1].set_xlabel('Wavelength(nm)')
        self.ax_eps[1].set_ylabel('Imag (Epsilon)')
        self.fig_eps.suptitle('Epsilon of {0}: {1} model'.format(self.material, self.model))

        plt.show()

    def plot_n_k(self):
        import matplotlib.pyplot as plt

        self.fig_nk, self.ax_nk = plt.subplots(1, 2, figsize=(15, 6))
        self.ax_nk[0].plot(1E9 * self.lamda, self.n, '-o')
        self.ax_nk[0].set_xlabel('Wavelength(nm)')
        self.ax_nk[0].set_ylabel('n')

        self.ax_nk[1].plot(1E9 * self.lamda, self.k, '-s')
        self.ax_nk[1].set_xlabel('Wavelength(nm)')
        self.ax_nk[1].set_ylabel('k')
        self.fig_nk.suptitle('n+ik of {0}: {1} model'.format(self.material, self.model))

        plt.show()


if __name__ == '__main__':
    import numpy as np

    lamda = np.linspace(200E-9, 2000E-9, 300)  # Creates a wavelength vector from 300 nm to 1000 nm of length 100
    silver = LD(lamda, material='Ag', model='LD')
    print silver.epsilon_real
    print silver.epsilon_imag
    print silver.n
    print silver.k
    silver.plot_epsilon()
    silver.plot_n_k()
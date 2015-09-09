## Description:

 This module calculates the real and imaginary part of the dielectric function,
 real and imaginary part of the refractive index for different metals using either
 Drude model (D) and Lorentz-Drude model (LD). The parameters are obtained from
 Rakic et al. This module is inspired by LD.m
 http://www.mathworks.com/matlabcentral/fileexchange/18040-drude-lorentz-and-debye-lorentz-models-for-the-dielectric-constant-of-metals-and-water

##Reference:

 Rakic et al., Optical properties of metallic films for vertical-
 cavity optoelectronic devices, Applied Optics (1998)


##Example:

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

##INPUT PARAMETERS for LD:

       lambda   ==> wavelength (meters) of light excitation on material. Numpy array

       material ==>    'Ag'  = silver
                       'Al'  = aluminum
                       'Au'  = gold
                       'Cu'  = copper
                       'Cr'  = chromium
                       'Ni'  = nickel
                       'W'   = tungsten
                       'Ti'  = titanium
                       'Be'  = beryllium
                       'Pd'  = palladium
                       'Pt'  = platinum

       model    ==> Choose 'LD' or 'D' for Lorentz-Drude or Drude model.

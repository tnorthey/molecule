import numpy as np
import os
import sys
# import my own modules
import molecule

# create class objects
sp = molecule.Structure_pool_method()

title = "nmm"
atomlist = np.array(['C', 'C', 'H', 'N', 'H', 'C', 'C', 'H', 'H', 'H', 'C', 'H', 'O', 'H', 'H', 'H', 'H', 'H'])
excitation_factor = 0.057
nstructures = 20000
modes = np.array([0, 6, 13, 14, 15, 18, 20])
nmodes = len(modes)
displacement_factors = 1.0 * np.ones(nmodes)
displacement_factors[0] = 6.0

subtitle = "modes0-6-13-14-15-18-20"
chi2_time_avg = sp.generate(
        title,
        subtitle,
        atomlist,
        excitation_factor,
        nstructures,
        modes,
        displacement_factors,
    )

print(chi2_time_avg)

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
nstructures = 100
modes = np.array([0, 1])
nmodes = len(modes)
displacement_factors = 1.0 * np.ones(nmodes)
#displacement_factors[0] = 6.0

chi2_time_avg = np.zeros((48, 48))
for i in range(48):
    if i==0:
        displacement_factors[0] = 6.0
    if i > 24:
        displacement_factors[0] = 0.2
    print(i)
    for j in range(i, 48):
    #for j in range(48):
        if j==0:
            displacement_factors[1] = 6.0
        if j > 24:
            displacement_factors[1] = 0.2
        modes = np.array([i, j])
        subtitle = "modes%i_%i" % (i, j)
        chi2_time_avg[i, j] = sp.generate(
                title,
                subtitle,
                atomlist,
                excitation_factor,
                nstructures,
                modes,
                displacement_factors,
            )

np.savez('chi2_2mode_time_avg.npz', chi2_time_avg=chi2_time_avg)

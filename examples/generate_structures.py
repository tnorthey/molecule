#!/bin/python3

import numpy as np
# import my own modules
import molecule

# create class object
m = molecule.Molecule()
nm = molecule.Normal_modes()

# starting coordinates
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/chlorobenzene.xyz")

# read normal modes
nmfile = "nm/chlorobenzene_normalmodes.txt"
natom = 12
nmodes = 30
modes = list(range(0, nmodes))
a = 0.2
displacements = nm.read_nm_displacements(nmfile, natom)
linear_dist = False
normal_dist = True

# generate random structures
nstructures = 10000
for i in range(nstructures):
    #mass = mi[ int( (i - 1) / 3 ) ]  # int rounds down by default
    #displacement_constant = (mass**.5 * 0.172*freqcm1**.5)**-1
    if linear_dist:
        factors = np.random.rand(nmodes)*2*a - a  # random factors in range [-a, a]
    elif normal_dist:
        mu, sigma = 0, a # mean and standard deviation
        factors = np.random.normal(mu, sigma, nmodes) # random factors in normal distribution with standard deviation = a
    #print(factors)

    displaced_xyz = nm.nm_displacer(xyz, displacements, modes, factors)

    fname = "xyz/generated/%s.xyz" % str(i).zfill(4)
    comment = "generated: %s" % str(i).zfill(4)
    m.write_xyz(fname, comment, atomlist, displaced_xyz)


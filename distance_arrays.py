#!/bin/python3

import numpy as np

# import my own modules
import molecule

# create class object
m = molecule.Molecule()
nm = molecule.Normal_modes()

# starting coordinates
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/chlorobenzene.xyz")

# read random structures
nstructures = 10000
ClC1, ClC4, C2H1, C1C2, C1C4 = (
    np.zeros(nstructures),
    np.zeros(nstructures),
    np.zeros(nstructures),
    np.zeros(nstructures),
    np.zeros(nstructures)
)

for i in range(nstructures):
    fname = "xyz/generated/10000_16/%s.xyz" % str(i).zfill(4)
    header, comment, atomlist, xyz = m.read_xyz(fname)
    dist_array = m.distances_array(xyz)
    ClC1[i] = dist_array[0, 1]  # bonded Cl-C distance
    ClC4[i] = dist_array[0, 4]  # Cl-C distance across the ring
    C2H1[i] = dist_array[2, 1 + 6]  # C-H bond
    C1C2[i] = dist_array[1, 2]      # C-C bond
    C1C4[i] = dist_array[1, 4]      # C-C bond

np.savez(
    "distances_10000_16.npz",
    ClC1=ClC1,
    ClC4=ClC4,
    C2H1=C2H1,
    C1C2=C1C2,
    C1C4=C1C4
)

"""
atom order:
Cl
C 
C 
C 
C 
C 
C 
H 
H 
H 
H 
H 
"""

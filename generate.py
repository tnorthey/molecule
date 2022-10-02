import numpy as np
import os
import sys

# import my own modules
import molecule

# create class objects
nm = molecule.Normal_modes()
sp = molecule.Structure_pool_method()

# title
title = "nmm"
# the xyz file "xyz/title.xyz" has to exist
# and the normal mode displacements file "nm/title_normalmodes.txt"

starting_xyzfile = "xyz/%s.xyz" % title
nmfile = "nm/%s_normalmodes.txt" % title
nmodes = 48
#modes = list(range(0, nmodes))
mode_low_range = int(sys.argv[6])
#modes = list(range(mode_low_range, mode_low_range + 1))
#modes = np.array([0, 14, 18, 28, 38])
modes = np.array([14, 18, 28, 38])
#modes = list(range(47, 48))
# define displacement factors array
displacement_factors = 0.5 * np.ones(len(modes))
displacement_factors[0] = 8.0
displacement_factors[1] = 2.0
displacement_factors[2] = 1.0
# =================================
# nstructures = 10000
nstructures = int(sys.argv[1])
#option = "linear"
option = "normal"
subtitle = sys.argv[5]
directory = "xyz/generated/%s_%s_%i" % (title, subtitle, nstructures)
os.makedirs(directory, exist_ok=True)  # create directory if doesn't exist
dist_arrays = True
iam_arrays = True

gen_structs = False
if sys.argv[2] == "true":
    gen_structs = True
if gen_structs:
    nm.generate_structures(
        starting_xyzfile,
        nmfile,
        modes,
        displacement_factors,
        nstructures,
        option,
        directory,
        dist_arrays,
        iam_arrays,
        subtitle
    )

# create chi2 array
chi2_bool = False
if sys.argv[3] == "true":
    chi2_bool = True
if chi2_bool:
    excitation_factor = 0.057
    iam_array_file = "iam_arrays_%i_%s.npz" % (nstructures, subtitle)
    sp.chi2_(iam_array_file, nstructures, excitation_factor, subtitle)

argmin_bool = False
if sys.argv[4] == "true":
    argmin_bool = True
# create argmin trajectory
if argmin_bool:
    chi2_file = "chi2_%i_%s.npz" % (nstructures, subtitle)
    atomlist = np.array(['C', 'C', 'H', 'N', 'H', 'C', 'C', 'H', 'H', 'H', 'C', 'H', 'O', 'H', 'H', 'H', 'H', 'H'])
    xyz_array_file = "xyz_array_%i_%s.npz" % (nstructures, subtitle)
    sp.xyz_trajectory(atomlist, xyz_array_file, chi2_file, nstructures, subtitle)


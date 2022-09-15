import numpy as np
import os
import sys
# import my own modules
import molecule

# create class object
nm = molecule.Normal_modes()

# title
title = "nmm"
# the xyz file "xyz/title.xyz" has to exist
# and the normal mode displacements file "nm/title_normalmodes.txt"

starting_xyzfile = "xyz/%s.xyz" % title
nmfile = "nm/%s_normalmodes.txt" % title
nmodes = 48
modes = list(range(0, nmodes))
displacement_factor = 0.2
#nstructures = 10000
nstructures = int(sys.argv[1])
option = 'normal'
directory = "xyz/generated/%s_%i" % (title, nstructures)
os.makedirs(directory, exist_ok=True)  # create directory if doesn't exist
dist_arrays = True
iam_arrays = True

nm.generate_structures(    
        starting_xyzfile,
        nmfile,
        modes,
        displacement_factor,
        nstructures,
        option,
        directory,
        dist_arrays,
        iam_arrays
    )

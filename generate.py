import numpy as np
import os
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
nmodes = 30
modes = list(range(0, nmodes))
displacement_factor = 0.2
nstructures = 1000
option = 'normal'
directory = "xyz/generated/%s_%i" % (title, nstructures)
os.makedirs(directory, exist_ok=True)  # create directory if doesn't exist

nm.generate_structures(    
        starting_xyzfile,
        nmfile,
        modes,
        displacement_factor,
        nstructures,
        option,
        directory,
    )
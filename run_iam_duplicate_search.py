from time import time
START=time()

import molecule

x = molecule.Xray()

# starting coordinates
starting_xyzfile = "xyz/chlorobenzene.xyz"

# read normal modes
nmfile = "nm/chlorobenzene_normalmodes.txt"
#modes = list(range(0, 20))
modes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 15, 16, 17, 19, 21)
nmodes = len(modes)
print("number of modes = %i" % nmodes)
displacement_factor = 0.2

# generate random structures
niterations = 10

x.iam_duplicate_search(starting_xyzfile, nmfile, modes, displacement_factor, niterations)

END=time()-START
print('Total time %f s' % END)
time_per_iter = END / niterations
print('time per iteraion = %f s' % time_per_iter)

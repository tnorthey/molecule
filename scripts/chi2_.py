import sys
# import my own modules
import molecule

# create class object
sp = molecule.Structure_pool_method()

N = int(sys.argv[1])
excitation_factor = 0.057

sp.chi2_(N, excitation_factor)

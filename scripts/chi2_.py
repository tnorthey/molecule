import sys
# import my own modules
import molecule

# create class object
sp = molecule.Structure_pool_method()

N = int(sys.argv[1])
excitation_factor = 0.057

iam_array_file = "iam_arrays_2pt5_100000.npz"
sp.chi2_(iam_array_file, N, excitation_factor)

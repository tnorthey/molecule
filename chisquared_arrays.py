import numpy as np
import sys
from scipy.stats import chisquare
# import my own modules
import molecule

# create class object
#nm = molecule.Normal_modes()

# read iam array
N = int(sys.argv[1])
array_file = 'iam_arrays_%i.npz' % N
f = np.load(array_file)
q = f['q']
nq = len(q)
data = f['iam']

chi2 = np.zeros((N, N))
for i in range(N):
    refi = data[:, i]
    for j in range(N):
        refj = data[:, j]
        chi2[i, j] = np.sum((refi - refj)**2)

chi2 /= nq # normalise by len(q)
np.save('chi2.npy', chi2)

print(chi2)

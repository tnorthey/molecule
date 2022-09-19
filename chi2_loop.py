import numpy as np
import sys
import scipy.io
from random import random

# from scipy.stats import chisquare
# import my own modules
import molecule

# create class object
m = molecule.Molecule()
x = molecule.Xray()

# read iam array
N = int(sys.argv[1])
array_file = "iam_arrays_%i.npz" % N
f = np.load(array_file)
q = f["q"]
nq = len(q)
pcd = f["pcd"]
print(pcd.shape)

# load experiment pcd
datafile = "data/NMM_exp_dataset.mat"
mat = scipy.io.loadmat(datafile)
t_exp = mat["t"]
q_exp = np.squeeze(mat["q"])
pcd_exp = mat["iso"]
errors_exp = mat["iso_stdx"]

# chi2 loop
nt = len(t_exp)
chi2 = np.zeros((N, nt))
a_factor = np.ones((N, nt))
for t in range(nt):
    print(t)
    y = np.squeeze(pcd_exp[:, t])
    maxy = np.mean(y)
    for i in range(N):
        x = pcd[:, i]
        maxx = np.mean(x)
        factor = maxy / maxx
        x = factor * x
        chi2[i, t] = np.sum((x - y) ** 2)
        for j in range(20):
            a = random()
            chi2_tmp = np.sum((a * x - y) ** 2)
            if chi2_tmp < chi2[i, t]:
                chi2[i, t] = chi2_tmp
                a_factor[i, t] = factor * a
                # if chi2_tmp < 0.001:
                #    break

chi2 /= nq  # normalise by len(q)
np.savez("chi2_%i.npz" % N, chi2=chi2, pcd=pcd, a_factor=a_factor)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# load chi2 file
chi2_file = np.load('../chi2_10000.npz')
chi2_array = chi2_file['chi2']
nt = 99

# load distances file
f100 = np.load('../distances_10000.npy')
c0c1 = f100[0, 1, :]

# load IAM file
f = np.load('../iam_arrays_10000.npz')
q = f['q']
pcd = f['pcd']

# load experiment data
datafile = '../data/NMM_exp_dataset.mat'
mat = scipy.io.loadmat(datafile)
q_exp = np.squeeze(mat['q'])
t_exp = np.squeeze(mat['t'])
pcd_exp = mat['iso']
errors = mat['iso_stdx']


factor = 0.057
theory = factor * pcd[:, i]
experiment = pcd_exp[:, time_step]

argmin_array = np.argmin(chi2_array[:, :], axis=0)
print(argmin_array)


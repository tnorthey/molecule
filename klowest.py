import numpy as np
import scipy.io
import sys
import molecule
m = molecule.Molecule()

# usage: python klowest.py Nstructures subtitle k time_step
N = int(sys.argv[1])
subtitle = sys.argv[2]
k = int(sys.argv[3])
time_step = int(sys.argv[4])

chi2_file = np.load('data/chi2_%i_%s.npz' % (N, subtitle))
chi2_array = chi2_file['chi2']

# load IAM
f = np.load('data/iam_arrays_%i_%s.npz' % (N, subtitle))
q = f['q']
pcd = f['pcd']

# load experiment data
datafile = 'data/NMM_exp_dataset.mat'
mat = scipy.io.loadmat(datafile)
q_exp = np.squeeze(mat['q'])
t_exp = np.squeeze(mat['t'])
pcd_exp = mat['iso']
errors = mat['iso_stdx']
i_pre_t0 = 13
t_exp = t_exp[i_pre_t0:]  # remove before t = 0
pcd_exp = pcd_exp[:, i_pre_t0:]
errors = errors[:, i_pre_t0:]

# indices of k lowest values
idx = np.argpartition(chi2_array[:, time_step], k)
print(idx[:k])
print('%i lowest chi2 values:' % k)
print(chi2_array[[idx[:k]], time_step])
print('in order:')
print(np.sort(chi2_array[[idx[:k]], time_step]))

# load xyz array
xyz_array_file = np.load('data/xyz_array_%i_%s.npz' % (N, subtitle))
_, _, atoms, _ = m.read_xyz('xyz/nmm.xyz')

for j in range(k):
    xyz = xyz_array_file['xyz'][:, :, idx[j]]
    print(xyz)
    fname = 'out_t_%s_%i.xyz' % (str(time_step).zfill(2), j)
    m.write_xyz(fname, 'k = %i' % j, atoms, xyz)



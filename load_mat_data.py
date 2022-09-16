import scipy.io
import numpy as np

datafile = 'data/NMM_exp_dataset.mat'
mat = scipy.io.loadmat(datafile)

print('t :')
print(mat['t'])
q = np.squeeze(mat['q'])
print('len(q) :')
print(len(q))
print('q :')
print(q)
print('iso :')
print(mat['iso'])
print('iso_stdx :')
print(mat['iso_stdx'])

# for v7.3 mat files (which are hdf5 datasets):
#import numpy as np
#import h5py
#f = h5py.File('somefile.mat','r')
#data = f.get('data/variable1')
#data = np.array(data) # For converting to a NumPy array


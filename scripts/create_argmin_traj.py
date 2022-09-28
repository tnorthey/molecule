import molecule

ms = molecule.Structure_pool_method()

N = 100000
directory = '../xyz/generated/nmm_24modes_2pt5_%i' % N
chi2_file = 'chi2_100000.npz'
ms.xyz_trajectory(directory, chi2_file, N)

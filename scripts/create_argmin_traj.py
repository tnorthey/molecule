import molecule

ms = molecule.Structure_pool_method()

N = 100000
directory = '../xyz/generated/nmm_24modes_%i' % N
chi2_file = 'chi2_100000_24modes.npz'
ms.xyz_trajectory(directory, chi2_file, N)

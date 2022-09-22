import molecule

ms = molecule.Mil_structure_method()

N = 10000
directory = 'xyz/generated/nmm_24modes_%i' % N
ms.xyz_trajectory(directory, N)

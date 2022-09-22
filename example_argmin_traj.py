import molecule

ms = molecule.Mil_structure_method()

directory = 'xyz/generated/nmm_1000'
N = 1000
ms.xyz_trajectory(directory, N)

import numpy as np
# my own modules
import molecule

# create class object
m = molecule.Molecule()
# define stuff
natom = 9

# normal mode definitions
nmfile = "nm/furan_normalmodes.txt"
displacements = m.read_nm_displacements(nmfile, natom)
displacement = displacements[20, :, :]  # 1st mode displacements

print(displacement)


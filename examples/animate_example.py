import molecule

m = molecule.Molecule()

xyz_start_file = 'xyz/furan.xyz'
nmfile = 'nm/furan_normalmodes.txt'
natoms = 9
modes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
for i in range(len(modes)):
    m.animate_mode(modes[i], xyz_start_file, nmfile, natoms)

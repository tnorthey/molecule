import numpy as np
import time
start_time=time.time()

# my own modules
import molecule

# create class objects
m = molecule.Molecule()
# define stuff
natom = 3
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
chargelist = [m.periodic_table(symbol) for symbol in atomlist]

# xray testing
x = molecule.Xray()
qlen = 80
qvector = np.linspace(0, 10, qlen, endpoint=True)  # q probably in a.u.
for k in range(4000):
    iam = x.iam_calc(chargelist, xyz, qvector)

end_time=time.time()-start_time
print(end_time)

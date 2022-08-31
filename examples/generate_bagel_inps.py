import numpy as np
# import my own modules
import molecule

# create class object
m = molecule.Molecule()

# generate random structures
nstructures = 10000
for i in range(nstructures):
    title = str(i).zfill(4)
    xyzfile = "xyz/generated/%s.xyz" % title
    jsonfile = "bagel/generated/%s.json" % title
    m.write_bagel_dyson(xyzfile, jsonfile)


import numpy as np
import molecule
import sys

m = molecule.Molecule()
nm = molecule.Normal_modes()

factor = float(sys.argv[1])

title = 'chd'
xyz_start_file = "../xyz/%s.xyz" % title
nmfile = "../nm/%s_normalmodes.txt" % title
natoms = 14
modes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
]
#    36,
#    37,
#    38,
#    39,
#    40,
#    41,
#    42,
#    43,
#    44,
#    45,
#    46,
#    47,
#]
for i in range(len(modes)):
    nm.animate_mode(modes[i], xyz_start_file, nmfile, natoms, factor)

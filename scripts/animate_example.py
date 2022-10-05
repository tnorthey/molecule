import numpy as np
import molecule

m = molecule.Molecule()
nm = molecule.Normal_modes()

factor = 3.0

xyz_start_file = "../xyz/nmm.xyz"
nmfile = "../nm/nmm_normalmodes.txt"
natoms = 18
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
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
]
for i in range(len(modes)):
    nm.animate_mode(modes[i], xyz_start_file, nmfile, natoms, factor)

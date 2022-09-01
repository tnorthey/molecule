[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# molecule

### Molecule subclass
- read/write xyz
- distance matrix, Coulomb matrix

Read xyz file example,

```
from molecule import Molecule

m = Molecule()  # initialise class object

xyzheader, comment, atomarray, xyzmatrix = m.read_xyz('xyz/test.xyz')
```

### Quantum subclass
- read/write quantum chemistry output files (only Bagel currently)
```
from molecule import Quantum

q = Quantum()

```

### Normal modes subclass
- displace molecules along their normal modes
- animate a normal mode (to visualise in VMD etc.)



### Spectra subclass
- Debye (IAM) x-ray molecule scattering, Lorentzian broaden spectra 

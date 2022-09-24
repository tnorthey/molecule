[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# molecule

## Usage

Setup virtual environment,
```bash
python -m venv .venv
```
then activate the venv,
``` source .venv/bin/activate ```
install requirements
```pip install -r requirements.txt```
To exit the venv,
```deactivate```

Add full path of base directory to the python path (so scripts work correctly), 
create a .pth file in this directory containing the full path e.g.,
```
echo "/home/thomas/molecule" > .venv/lib/python3.8/site-packages/fullpath.pth"
```

It uses pytest (unit tests are in `test_functions.py`).
```pytest -v```
to confirm all tests pass.

## Description
### Molecule subclass
- read/write xyz
- distance matrix, Coulomb matrix

Read xyz file example,

```python
from molecule import Molecule

m = Molecule()  # initialise class object

xyzheader, comment, atomarray, xyzmatrix = m.read_xyz('xyz/test.xyz')
```

### Quantum subclass
- read/write quantum chemistry output files (only Bagel currently)
```python
from molecule import Quantum

q = Quantum()
```

### Normal modes subclass
- displace molecules along their normal modes
- animate a normal mode (to visualise in VMD etc.)
```python
from molecule import Normal_modes

nm = Normal_modes()
```




### Spectra subclass
- Debye (IAM) x-ray molecule scattering
- Lorentzian broaden spectra 

Lorentzian broadening example,
```python
from molecule import Spectra

s = Spectra()

x = energies 	# x-values of input data
y = intensities # y-values of input data
xmin = 0	# minimum value of broadened data
xmax = 1	# maximum value of broadened data
n = 100   	# length of broadened data
fwhm = 0.2	# FWHM of the Lorentzian function

x_new, y_new = s.lorenzian_broaden(x, y, xmin, xmax, n, fwhm)
```

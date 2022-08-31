#!/bin/python

import numpy as np
import sys
import os

class Spectra:
    """Manipulate spectra data; apply broadening etc."""

    def __init__(self):
        pass

    def lorenzian_broaden(self, x, y, xmin, xmax, n, fwhm):
        """Apply Lorenzian broadening (w. FWHM = fwhm) to data y(x),
        outputs new data with length n and min, max = xmin, xmax"""
        x_new = np.linspace(xmin, xmax, n, endpoint=True)
        y_broadened = np.zeros(n)

        g = (0.5 * fwhm) ** 2  # Factor in Lorentzian function
        for j in range(len(y)):  # loop over original data length
            y_val = y[j]
            x_val = x[j]
            for i in range(n):  # loop over new data size
                lorentz = (
                    y_val * g / ((x_new[i] - x_val) ** 2 + g)
                )  # Lorentzian broadening
                y_broadened[i] += lorentz
        return x_new, y_broadened

    def read_bagel_dyson(self, bagel_dyson_output, max_rows):
        """read dyson norms and ionisation energies from a bagel dyson output file"""
        str_find = "Norms^2 of Dyson orbitals approximately indicate the strength of an inization transitions."
        energy, norm = [], []  # define here to avoid return error if str_find isn't found
        with open(bagel_dyson_output, "r") as f:
            for line in f:
                if str_find in line:    # go to line containing str
                    out_array = np.loadtxt(     # numpy loadtxt into an array
                        f,
                        dtype={
                            "names": ("from", "-", "to", "energy", "norm"),
                            "formats": ("i4", "a2", "i4", "f4", "f4"),
                        },
                        skiprows = 4,
                        max_rows = max_rows
                    )
                    energy = out_array["energy"]
                    norm = out_array["norm"]
        return energy, norm


s = Spectra()

fwhm = 0.02
npoints = 200
max_rows = 6
xmin = 0
xmax = 1
outfile = sys.argv[1]
split_tup = os.path.splitext(outfile)
file_name = split_tup[0]  # file name with extension removed
energy, norm = s.read_bagel_dyson(outfile, max_rows)
if len(energy) != 0:
    x_new, y_new = s.lorenzian_broaden(energy, norm**2, xmin, xmax, npoints, fwhm)
    data = np.append(np.transpose([x_new]), np.transpose([y_new]), axis=1)
    # save to csv
    csvfile = '%s.csv' % file_name
    np.savetxt(csvfile, data, delimiter=' ')


import numpy as np
# import my own modules
import molecule
import spectra

# create class object
m = molecule.Molecule()
s = spectra.Spectra()

# generate random structures
nstructures = 1000
max_rows = 6
fwhm = 0.02
npoints = 200
xmin = 0
xmax = 1
total_data = np.zeros((npoints, 2))
for i in range(nstructures):
    title = str(i).zfill(4)
    outfile = "bagel/generated/%s.out" % title
    energy, norm = m.read_bagel_dyson(outfile, max_rows)
    if len(energy) != 0:
        x_new, y_new = s.lorenzian_broaden(energy, norm**2, xmin, xmax, npoints, fwhm)
        data = np.append(np.transpose([x_new]), np.transpose([y_new]), axis=1)
        total_data[:, 0] = data[:, 0]
        total_data[:, 1] += data[:, 1]
        # save to csv
        csvfile = 'csv/%s.csv' % title
        np.savetxt(csvfile, data, delimiter=' ')
csvfile = 'csv/total.csv'
np.savetxt(csvfile, total_data, delimiter=' ')


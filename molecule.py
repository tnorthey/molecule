"""
molecule, T. Northey, 2022
### Read & write xyz files, convert to Z-matrix, Coulomb matrix,
    transform / sample by normal modes, Z-matrix manipulation
### Goals (done: -, not done: x)
- read xyz
- write xyz
x convert to Z-matrix
- convert to Coulomb matrix (CM)
- ability to sort atomlist and xyz by charge (consistently ordered CM)
- reduced CM
- normal mode displacement
x normal mode sampling
x Z-matrix displacement
x Z-matrix sampling
"""
######
import os
import numpy as np
import pandas as pd

######
class Molecule:
    """methods to manipulate molecules"""

    def __init__(self):
        pass

    def periodic_table(self, element):
        """Outputs atomic number for each element in the periodic table"""
        with open("pt.txt") as pt_file:
            for line in pt_file:
                if line.split()[0] == element:
                    return int(line.split()[1])

    def atomic_mass(self, element):
        """Outputs atomic mass for each element in the periodic table"""
        with open("atomic_masses.txt") as am_file:
            for line in am_file:
                if line.split()[0] == element:
                    return line.split()[1]

    # read/write xyz files

    def read_xyz(self, fname):
        """Read a .xyz file"""
        with open(fname, "r") as xyzfile:
            xyzheader = int(xyzfile.readline())
            comment = xyzfile.readline()
        xyzmatrix = np.loadtxt(fname, skiprows=2, usecols=[1, 2, 3])
        atomarray = np.loadtxt(fname, skiprows=2, dtype=str, usecols=[0])
        return xyzheader, comment, atomarray, xyzmatrix

    def write_xyz(self, fname, comment, atoms, xyz):
        """Write .xyz file"""
        natom = len(atoms)
        xyz = np.transpose(xyz)
        atoms_xyz = np.transpose(np.append([atoms], xyz, axis=0))
        with open(fname, "w") as xyzfile:
            np.savetxt(
                fname,
                atoms_xyz,
                fmt="%s",
                delimiter=" ",
                header=str(natom) + "\n" + comment,
                footer="",
                comments="",
            )
        return

    def sort_array(self, tosort, sortbyarray):
        """sort tosort by sortbyarray (have to be same size)"""
        indices = np.argsort(sortbyarray)
        indices = indices[::-1]
        sorted_array = tosort[indices]
        return sorted_array

    ### distances array

    def distances_array(self, xyz):
        """Computes matrix of distances from xyz"""
        natom = xyz.shape[0]  # number of atoms
        dist_array = np.zeros((natom, natom))  # the array of distances
        for i in range(natom):
            dist_array[i, i] = 0
            for j in range(i + 1, natom):
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                dist_array[i, j] = dist
                dist_array[j, i] = dist  # opposite elements are equal
        return dist_array

    # Coulomb matrix

    def triangle_cm(self, charges, xyz, dim):
        """Computes the triangle Coulomb matrix from charges and xyz arrays"""

        tcm = np.zeros((dim, dim))  # the CM of size dim**2
        fcm = np.zeros((dim, dim))  # must make sure to np.zeros; fcm=tcm doesn't work.
        natom = len(charges)  # number of atoms

        for i in range(natom):
            diag_element = 0.5 * charges[i] ** 2.4  # diagonal elements
            tcm[i, i] = diag_element
            fcm[i, i] = diag_element
            for j in range(i + 1, natom):
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                reps = charges[i] * charges[j] / dist  # Pair-wise repulsion
                tcm[i, j] = reps
                fcm[i, j] = reps
                fcm[j, i] = reps  # opposite elements are equal
        return tcm, fcm

    def reduced_cm(self, cm, size):
        """change CM to reduced CM"""
        # only 1st row of CM
        rcm = cm[0:size, 0]
        return rcm


m = Molecule()
### End Molecule class section


class Quantum:
    def __init__(self):
        pass

    # Bagel stuff
    def write_bagel_dyson(self, xyzfile, outfile="bagel_inp.json"):
        """writes a bagel dyson norm input file based on
        bagel_dyson.template with atoms and geometry from xyzfile"""
        _, _, atoms, _, xyzmatrix = self.read_xyz(xyzfile)
        bagel_df = pd.read_json(
            "templates/bagel_dyson.template"
        )  # read bagel input template as pandas dataframe
        for k in range(len(atoms)):
            bagel_df["bagel"][0]["geometry"][k]["atom"] = atoms[k]
            bagel_df["bagel"][0]["geometry"][k]["xyz"] = xyzmatrix[k, :]
        bagel_df.to_json(outfile, indent=4)  # this runs in bagel!
        return

    def read_bagel_dyson(self, bagel_dyson_output, max_rows):
        """read dyson norms and ionisation energies from a bagel dyson output file"""
        str_find = "Norms^2 of Dyson orbitals approximately indicate the strength of an inization transitions."
        energy, norm = (
            [],
            [],
        )  # define here to avoid return error if str_find isn't found
        with open(bagel_dyson_output, "r") as f:
            for line in f:
                if str_find in line:  # go to line containing str
                    out_array = np.loadtxt(  # numpy loadtxt into an array
                        f,
                        dtype={
                            "names": ("from", "-", "to", "energy", "norm"),
                            "formats": ("i4", "a2", "i4", "f4", "f4"),
                        },
                        skiprows=4,
                        max_rows=max_rows,
                    )
                    energy = out_array["energy"]
                    norm = out_array["norm"]
        return energy, norm

    ### End Bagel stuff


class Normal_modes:
    def __init__(self):
        pass

    ### Normal modes section
    def read_nm_displacements(self, fname, natoms):
        """read_nm_displacements: Reads displacement vector from file=fname e.g. 'normalmodes.txt'
        Inputs: 	natoms (int), total number of atoms
        Outputs:	displacements, array of displacements, size: (nmodes, natoms, 3)"""
        if natoms == 2:
            nmodes = 1
        elif natoms > 2:
            nmodes = 3 * natoms - 6
        else:
            print("ERROR: natoms. Are there < 2 atoms?")
            return False
        with open(fname, "r") as xyzfile:
            tmp = np.loadtxt(fname)
        displacements = np.zeros((nmodes, natoms, 3))
        for i in range(3 * natoms):
            for j in range(nmodes):
                if i % 3 == 0:  # Indices 0,3,6,...
                    dindex = int(i / 3)
                    displacements[j, dindex, 0] = tmp[i, j]  # x coordinates
                elif (i - 1) % 3 == 0:  # Indices 1,4,7,...
                    displacements[j, dindex, 1] = tmp[i, j]  # y coordinates
                elif (i - 2) % 3 == 0:  # Indices 2,5,8,...
                    displacements[j, dindex, 2] = tmp[i, j]  # z coordinates
        return displacements

    def displace_xyz(self, xyz, displacement, factor):
        """displace xyz by displacement * factor
        xyz and displacement should be same size"""
        return xyz + displacement * factor

    def nm_displacer(self, xyz, displacements, modes, factors):
        """displace xyz along all displacements by factors array"""
        summed_displacement = np.zeros(displacements[0, :, :].shape)
        modes_array = np.squeeze(np.array([modes]))  # convert to arrays for iteration
        nmodes = len(modes_array)
        factors_array = np.multiply(factors, np.ones(nmodes))
        for i in range(nmodes):
            summed_displacement += (
                displacements[modes_array[i], :, :] * factors_array[i]
            )
        displaced_xyz = self.displace_xyz(xyz, summed_displacement, 1.0)
        return displaced_xyz

    def animate_mode(self, mode, xyz_start_file, nmfile, natoms):
        """make xyz file animation along normal mode"""
        displacements = self.read_nm_displacements(nmfile, natoms)
        a = 0.4
        factor = np.linspace(-a, a, 20, endpoint=True)
        factor = np.append(factor, np.linspace(a, -a, 20, endpoint=True))
        _, _, atoms, xyz_start = m.read_xyz(xyz_start_file)
        for k in range(len(factor)):
            xyz = self.nm_displacer(xyz_start, displacements, mode, factor[k])
            xyzfile_out = "animate/mode%i_%s.xyz" % (mode, str(k).zfill(2))
            m.write_xyz(xyzfile_out, str(factor[k]), atoms, xyz)

    def generate_structures(
        self,
        starting_xyzfile,
        nmfile,
        modes,
        displacement_factor,
        nstructures,
        option,
        directory,
        dist_arrays
    ):
        """generate xyz files by normal mode displacements"""
        nmodes = len(modes)
        xyzheader, comment, atomlist, xyz = m.read_xyz(starting_xyzfile)
        natoms = len(atomlist)
        # starting coordinates
        a = displacement_factor
        # read normal modes
        displacements = self.read_nm_displacements(nmfile, natoms)
        if option == "linear":
            linear_dist, normal_dist = True, False
        elif option == "normal":
            linear_dist, normal_dist = False, True
        # generate random structures
        n_zfill = len(str(nstructures))
        if dist_arrays:
            dist_array = np.zeros((natoms, natoms, nstructures))
            dist_save_bool = True
        for i in range(nstructures):
            if linear_dist:
                factors = (
                    np.random.rand(nmodes) * 2 * a - a
                )  # random factors in range [-a, a]
            elif normal_dist:
                mu, sigma = 0, a  # mean and standard deviation
                factors = np.random.normal(
                    mu, sigma, nmodes
                )  # random factors in normal distribution with standard deviation = a
            displaced_xyz = self.nm_displacer(xyz, displacements, modes, factors)
            if dist_save_bool:
                dist_array[:,:,i] = m.distances_array(displaced_xyz)
            fname = "%s/%s.xyz" % (directory, str(i).zfill(n_zfill))
            comment = "generated: %s" % str(i).zfill(n_zfill)
            m.write_xyz(fname, comment, atomlist, displaced_xyz)
        # file saves
        if dist_save_bool:
            outfile = 'distances.npy'
            np.save(outfile, dist_array)


nm = Normal_modes()
### End normal modes section


class Spectra:
    """Manipulate spectra data; apply broadening etc."""

    def __init__(self):
        pass

    def lorenzian_broaden(self, x, y, xmin, xmax, n, fwhm):
        """Apply Lorenzian broadening (FWHM = fwhm) to data y(x),
        outputs new data with length n and min, max = xmin, xmax"""
        x_new = np.linspace(xmin, xmax, n, endpoint=True)
        y_new = np.zeros(n)
        g = (0.5 * fwhm) ** 2  # Factor in Lorentzian function
        for j in range(len(y)):  # loop over original data length
            y_val = y[j]
            x_val = x[j]
            for i in range(n):  # loop over new data size
                lorentz = (
                    y_val * g / ((x_new[i] - x_val) ** 2 + g)
                )  # Lorentzian broadening
                y_new[i] += lorentz
        return x_new, y_new


class Xray:
    def __init__(self):
        pass

    def atomic_factor(self, atom_number, qvector):
        """returns atomic x-ray scattering factor for atom_number, and qvector"""
        # coeffs hard coded here (maybe move to separate file later.)
        aa = np.array(
            [
                [0.489918, 0.262003, 0.196767, 0.049879],  # hydrogen
                [0.8734, 0.6309, 0.3112, 0.1780],  # helium
                [1.1282, 0.7508, 0.6175, 0.4653],  # lithium
                [1.5919, 1.1278, 0.5391, 0.7029],  # berylium
                [2.0545, 1.3326, 1.0979, 0.7068],  # boron
                [2.3100, 1.0200, 1.5886, 0.8650],  # carbon
                [12.2126, 3.1322, 2.0125, 1.1663],  # nitrogen
                [3.0485, 2.2868, 1.5463, 0.8670],  # oxygen
                [3.5392, 2.6412, 1.5170, 1.0243],  # fluorine
                [3.9553, 3.1125, 1.4546, 1.1251],  # neon
                [4.7626, 3.1736, 1.2674, 1.1128],  # sodium
                [5.4204, 2.1735, 1.2269, 2.3073],  # magnesium
                [6.4202, 1.9002, 1.5936, 1.9646],  # aluminium
                [6.2915, 3.0353, 1.9891, 1.5410],  # Siv
                [6.4345, 4.1791, 1.7800, 1.4908],  # phosphorus
                [6.9053, 5.2034, 1.4379, 1.5863],  # sulphur
                [11.4604, 7.1964, 6.2556, 1.6455],  # chlorine
            ]
        )

        bb = np.array(
            [
                [20.6593, 7.74039, 49.5519, 2.20159],  # hydrogen
                [9.1037, 3.3568, 22.9276, 0.9821],  # helium
                [3.9546, 1.0524, 85.3905, 168.261],  # lithium
                [43.6427, 1.8623, 103.483, 0.5420],  # berylium
                [23.2185, 1.0210, 60.3498, 0.1403],  # boron
                [20.8439, 10.2075, 0.5687, 51.6512],  # carbon
                [0.00570, 9.8933, 28.9975, 0.5826],  # nitrogen
                [13.2771, 5.7011, 0.3239, 32.9089],  # oxygen
                [10.2825, 4.2944, 0.2615, 26.1476],  # fluorine
                [8.4042, 3.4262, 0.2306, 21.7184],  # Ne
                [3.2850, 8.8422, 0.3136, 129.424],  # Na
                [2.8275, 79.2611, 0.3808, 7.1937],  # Mg
                [3.0387, 0.7426, 31.5472, 85.0886],  # Al
                [2.4386, 32.3337, 0.6785, 81.6937],  # Siv
                [1.9067, 27.1570, 0.5260, 68.1645],  # P
                [1.4679, 22.2151, 0.2536, 56.1720],  # S
                [0.0104, 1.1662, 18.5194, 47.7784],  # Cl
            ]
        )

        cc = np.array(
            [
                0.001305,  # hydrogen
                0.0064,  # helium
                0.0377,  # lithium
                0.0385,  # berylium
                -0.1932,  # boron
                0.2156,  # carbon
                -11.529,  # nitrogen
                0.2508,  # oxygen
                0.2776,  # fluorine
                0.3515,  # Ne
                0.6760,  # Na
                0.8584,  # Mg
                1.1151,  # Al
                1.1407,  # Si
                1.1149,  # P
                0.8669,  # S
                -9.5574,  # Cl
            ]
        )

        qlen = len(qvector)
        atomfactor = np.zeros(qlen)
        for j in range(qlen):
            for i in range(4):
                atomfactor[j] += aa[atom_number - 1, i] * np.exp(
                    -bb[atom_number - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                )
        atomfactor += cc[atom_number - 1]
        return atomfactor

    def iam_calc(self, atomic_numbers, xyz, qvector):
        """calculate IAM molecular scattering curve for atoms, xyz, qvector"""
        natom = len(atomic_numbers)
        qlen = len(qvector)
        atomic = np.zeros(qlen)
        molecular = np.zeros(qlen)
        for i in range(natom):
            atomic += self.atomic_factor(atomic_numbers[i], qvector) ** 2
            for j in range(i + 1, natom):  # j > i
                fij = np.multiply(
                    self.atomic_factor(atomic_numbers[i], qvector),
                    self.atomic_factor(atomic_numbers[j], qvector),
                )
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                molecular += 2 * fij * np.sinc(qvector * dist / np.pi)
        iam = atomic + molecular
        return iam

    def iam_duplicate_search(
        self, starting_xyzfile, nmfile, modes, displacement_factor, niterations
    ):
        # from scipy.stats import chisquare
        # starting coordinates
        xyzheader, comment, atomlist, xyz = m.read_xyz(starting_xyzfile)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        natoms = len(atomlist)
        # read normal modes
        displacements = nm.read_nm_displacements(nmfile, natoms)
        a = displacement_factor
        nmodes = len(modes)
        qlen = 101
        qvector = np.linspace(0, 10, qlen, endpoint=True)
        # generate random structures
        thresh_chi_dist = 0.01
        # thresh_chi_iams = 0.001
        thresh_chi_iams = 0.01
        c = 0
        for i in range(niterations):
            factors = (
                np.random.rand(nmodes) * 2 * a - a
            )  # random factors in range [-a, a]
            xyz_1 = nm.nm_displacer(xyz, displacements, modes, factors)
            factors = (
                np.random.rand(nmodes) * 2 * a - a
            )  # random factors in range [-a, a]
            xyz_2 = nm.nm_displacer(xyz, displacements, modes, factors)
            dist_array_1 = m.distances_array(xyz_1)
            dist_array_2 = m.distances_array(xyz_2)
            iam_1 = self.iam_calc(atomic_numbers, xyz_1, qvector)
            iam_2 = self.iam_calc(atomic_numbers, xyz_2, qvector)
            chi_dists = (
                abs(np.sum(dist_array_1.flatten() - dist_array_2.flatten()))
                / natoms**2
            )
            chi_iams = 100 * abs(np.sum(iam_1 / iam_2 - 1)) / qlen
            if chi_dists > thresh_chi_dist and chi_iams < thresh_chi_iams:
                c += 1
                print(c)
                m.write_xyz("xyz/%d_found_1.xyz" % c, "found %d" % c, atomlist, xyz_1)
                m.write_xyz("xyz/%d_found_2.xyz" % c, "found %d" % c, atomlist, xyz_2)
                # save IAMs to csv
                csvfile = "xyz/%d_found_1.csv" % c
                np.savetxt(csvfile, np.column_stack((qvector, iam_1)), delimiter=" ")
                csvfile = "xyz/%d_found_2.csv" % c
                np.savetxt(csvfile, np.column_stack((qvector, iam_2)), delimiter=" ")
        return

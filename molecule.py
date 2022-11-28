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
from numpy import random
import pandas as pd
import scipy.io
from scipy import interpolate
from scipy.spatial.transform import Rotation as R

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
        xyz = xyz.astype("|S10")  # convert to string array (max length 10)
        atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
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

    def read_xyz_traj(self, fname, ntsteps):
        """Read a .xyz trajectory file"""
        with open(fname, "r") as xyzfile:
            natoms = int(xyzfile.readline())
            comment = xyzfile.readline()
        atomarray = np.loadtxt(fname, skiprows=2, max_rows=14, dtype=str, usecols=[0])
        xyztraj = np.zeros((natoms, 3, ntsteps))
        for t in range(ntsteps):
            xyztraj[:, :, t] = np.loadtxt(
                fname, skiprows=14 * t + 2 * (t + 1), max_rows=14, usecols=[1, 2, 3]
            )
        return natoms, comment, atomarray, xyztraj

    def write_xyz_traj(self, fname, atoms, xyz_traj):
        """converts xyz_traj array to traj.xyz"""
        natom = len(atoms)
        atoms_xyz_traj = np.empty((1, 4))
        for t in range(xyz_traj.shape[2]):
            comment = "iteration: %i" % t
            xyz = xyz_traj[:, :, t]
            xyz = xyz.astype("|S14")  # convert to string array (max length 14)
            tmp = np.array([[str(natom), "", "", ""], [comment, "", "", ""]])
            atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
            atoms_xyz = np.append(tmp, atoms_xyz, axis=0)
            atoms_xyz_traj = np.append(atoms_xyz_traj, atoms_xyz, axis=0)
        atoms_xyz_traj = atoms_xyz_traj[1:, :]  # remove 1st line of array
        print("writing %s..." % fname)
        np.savetxt(
            fname,
            atoms_xyz_traj,
            fmt="%s",
            delimiter=" ",
            header="",
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
        if not hasattr(modes, "__len__"):  # create array if not
            modes = np.array([modes])  # convert to arrays for iteration
        nmodes = len(modes)
        factors_array = np.multiply(factors, np.ones(nmodes))
        for i in range(nmodes):
            summed_displacement += displacements[modes[i], :, :] * factors_array[i]
        return xyz + summed_displacement

    def animate_mode(self, mode, xyz_start_file, nmfile, natoms, factor):
        """make xyz file animation along normal mode"""
        displacements = self.read_nm_displacements(nmfile, natoms)
        a = factor
        factor = np.linspace(-a, a, 20, endpoint=True)
        factor = np.append(factor, np.linspace(a, -a, 20, endpoint=True))
        _, _, atoms, xyz_start = m.read_xyz(xyz_start_file)
        for k in range(len(factor)):
            xyz = self.nm_displacer(xyz_start, displacements, mode, factor[k])
            xyzfile_out = "../animate/mode%i_%s.xyz" % (mode, str(k).zfill(2))
            m.write_xyz(xyzfile_out, str(factor[k]), atoms, xyz)

    def generate_structures(
        self,
        starting_xyzfile,
        nmfile,
        modes,
        displacement_factors,
        nstructures,
        option,
        directory,
        dist_arrays,
        iam_arrays,
        subtitle,
    ):
        """generate xyz files by normal mode displacements"""
        nmodes = len(modes)
        xyzheader, comment, atomlist, xyz = m.read_xyz(starting_xyzfile)
        natoms = len(atomlist)
        # read normal modes
        displacements = self.read_nm_displacements(nmfile, natoms)
        displaced_xyz_array = np.zeros((natoms, 3, nstructures))
        if option == "linear":
            linear_dist, normal_dist = True, False
        elif option == "normal":
            linear_dist, normal_dist = False, True
        dist_save_bool, iam_save_bool = False, False
        write_xyz_files = True
        # generate random structures
        n_zfill = len(str(nstructures))
        if dist_arrays:
            dist_array = np.zeros((natoms, natoms, nstructures))
            dist_save_bool = True
        if iam_arrays:
            nq = 39
            qstart = 0.33226583
            qend = 4.37267671
            qvector = np.linspace(qstart, qend, nq, endpoint=True)
            atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
            iam_array = np.zeros((nq, nstructures))
            pcd_array = np.zeros((nq, nstructures))
            iam_save_bool = True
            # refence IAM curve
            reference_xyz = "xyz/nmm.xyz"
            xyzheader, comment, atomlist, xyz = m.read_xyz(reference_xyz)
            reference_iam = x.iam_calc(atomic_numbers, xyz, qvector)
        for i in range(nstructures):
            print(i)
            if linear_dist:
                factors = np.zeros(nmodes)
                for j in range(nmodes):
                    a = displacement_factors[j]
                    factors[j] = (
                        random.random_sample() * 2 * a - a
                    )  # random factors in range [-a, a]
            elif normal_dist:
                mu, sigma = 0, displacement_factors  # mean and standard deviation
                factors = random.normal(
                    mu, sigma, nmodes
                )  # random factors in normal distribution with standard deviation = a
            displaced_xyz = self.nm_displacer(xyz, displacements, modes, factors)
            displaced_xyz_array[:, :, i] = displaced_xyz
            if dist_save_bool:
                dist_array[:, :, i] = m.distances_array(displaced_xyz)
            if iam_save_bool:
                iam_array[:, i] = x.iam_calc(atomic_numbers, displaced_xyz, qvector)
                pcd_array[:, i] = 100 * (iam_array[:, i] / reference_iam - 1)
            if write_xyz_files:
                fname = "%s/%s.xyz" % (directory, str(i).zfill(n_zfill))
                comment = "generated: %s" % str(i).zfill(n_zfill)
                m.write_xyz(fname, comment, atomlist, displaced_xyz)
        # file saves
        outfile = "data/xyz_array_%i_%s" % (nstructures, subtitle)
        np.savez(outfile, xyz=displaced_xyz_array)
        if dist_save_bool:
            outfile = "data/distances_%i_%s.npy" % (nstructures, subtitle)
            np.save(outfile, dist_array)
        if iam_save_bool:
            outfile = "data/iam_arrays_%i_%s.npz" % (nstructures, subtitle)
            # print("saving %s..." % outfile)
            np.savez(outfile, q=qvector, iam=iam_array, pcd=pcd_array)


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

    def iam_calc_slow(self, atomic_numbers, xyz, qvector):
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

    def compton_spline(self, atomic_numbers, qvector):
        """spline the compton factors to correct qvector, outputs array (atoms, qvector)"""
        natom = len(atomic_numbers)
        compton_array = np.zeros(
            (natom, len(qvector))
        )  # inelastic component for each atom
        tmp = np.load("Compton_Scattering_Intensities.npz")  # compton factors
        q_compton, arr = tmp["q_compton"], tmp["compton"]
        for i in range(natom):
            tck = interpolate.splrep(q_compton, arr[atomic_numbers[i] - 1, :], s=0)
            compton_array[i, :] = interpolate.splev(qvector, tck, der=0)
        return compton_array

    def iam_calc_compton(self, atomic_numbers, xyz, qvector, compton_array):
        """calculate IAM molecular scattering curve for atoms, xyz, qvector"""
        natom = len(atomic_numbers)
        qlen = len(qvector)
        atomic = np.zeros(qlen)  # total atomic factor
        molecular = np.zeros(qlen)  # total molecular factor
        compton = np.zeros(qlen)  # total compton factor
        atomic_factor_array = np.zeros((natom, qlen))  # array of atomic factors
        for i in range(natom):
            tmp = self.atomic_factor(atomic_numbers[i], qvector)
            atomic_factor_array[i, :] = tmp
            atomic += tmp**2
            compton += compton_array[i, :]
        for i in range(natom):
            for j in range(i + 1, natom):  # j > i
                molecular += (
                    2
                    * np.multiply(atomic_factor_array[i, :], atomic_factor_array[j, :])
                    * np.sinc(qvector * np.linalg.norm(xyz[i, :] - xyz[j, :]) / np.pi)
                )
        iam = atomic + molecular
        return iam, compton

    def iam_calc(self, atomic_numbers, xyz, qvector):
        """calculate IAM molecular scattering curve for atoms, xyz, qvector"""
        natom = len(atomic_numbers)
        qlen = len(qvector)
        atomic = np.zeros(qlen)  # total atomic factor
        molecular = np.zeros(qlen)  # total molecular factor
        atomic_factor_array = np.zeros((natom, qlen))  # array of atomic factors
        for i in range(natom):
            tmp = self.atomic_factor(atomic_numbers[i], qvector)
            atomic_factor_array[i, :] = tmp
            atomic += tmp**2
        qpi = qvector / np.pi
        for i in range(natom):
            for j in range(i + 1, natom):  # j > i
                molecular += np.multiply(
                    atomic_factor_array[i, :], atomic_factor_array[j, :]
                ) * np.sinc(qpi * np.linalg.norm(xyz[i, :] - xyz[j, :]))
        return atomic + 2 * molecular

    def iam_duplicate_search(
        self, starting_xyzfile, nmfile, modes, displacement_factor, niterations
    ):
        """finds very similar IAM curves for different geometries"""
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
            factors = random.rand(nmodes) * 2 * a - a  # random factors in range [-a, a]
            xyz_1 = nm.nm_displacer(xyz, displacements, modes, factors)
            factors = random.rand(nmodes) * 2 * a - a  # random factors in range [-a, a]
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


x = Xray()


class Structure_pool_method:
    """functions related to the 1m structure method"""

    def __init__(self):
        pass

    def chi2_(self, iam_array_file, N, excitation_factor, subtitle):
        """loops over iam_array and compares to exp. outputs chi2 array"""
        # read iam array
        array_file = "data/%s" % iam_array_file
        # print('reading %s...' % array_file)
        f = np.load(array_file, allow_pickle=True)
        q = f["q"]
        nq = len(q)
        pcd = f["pcd"]
        # print(pcd.shape)
        # load experiment pcd
        datafile = "data/NMM_exp_dataset.mat"
        mat = scipy.io.loadmat(datafile)
        t_exp = mat["t"]
        q_exp = np.squeeze(mat["q"])
        pcd_exp = mat["iso"]
        errors_exp = mat["iso_stdx"]
        # optionally "clean up" data
        pre_t0_bool, combine_bool = True, False
        if pre_t0_bool:
            i_pre_t0 = 13
            t_exp = t_exp[i_pre_t0:]  # remove before t = 0
            pcd_exp = pcd_exp[:, i_pre_t0:]
        if combine_bool:
            # also combine every 2nd lineout
            t_exp = t_exp[0::2]
            pcd_exp = 0.5 * (pcd_exp[:, 0::2] + pcd_exp[:, 1::2])
        # print(len(t_exp))
        # print(pcd_exp.shape)
        nt = len(t_exp)
        # chi2 loop
        chi2 = np.zeros((N, nt))
        for t in range(nt):
            # print(t)
            y = np.squeeze(pcd_exp[:, t])
            errors = np.squeeze(errors_exp[:, t])
            for i in range(N):
                x = excitation_factor * pcd[:, i]
                # chi2[i, t] = np.sum(((x - y) / errors) ** 2)
                # chi2[i, t] = np.sum(((x - y) / y) ** 2 )
                chi2[i, t] = np.sum((x - y) ** 2)
        chi2 /= nq  # normalise by len(q)
        np.savez("data/chi2_%i_%s.npz" % (N, subtitle), chi2=chi2)
        return

    def chi2_arrays(self, N):
        """theory-theory chi2 values for all combinations of IAM curve in iam_arrays.npz"""
        # read iam array
        array_file = "iam_arrays_%i.npz" % N
        f = np.load(array_file)
        q = f["q"]
        nq = len(q)
        data = f["iam"]
        chi2 = np.zeros((N, N))
        for i in range(N):
            x = data[:, i]
            for j in range(N):
                y = data[:, j]
                chi2[i, j] = np.sum((x - y) ** 2)
        chi2 /= nq  # normalise by len(q)
        np.save("chi2_array.npy", chi2)
        return

    def xyz_trajectory(self, atoms, xyz_array_file, chi2_file, N, subtitle):
        """outputs xyz trajectory based on chi2 array"""
        n_zfill = len(str(N))
        natom = len(atoms)
        # load chi2 file
        chi2_file = np.load("data/%s" % chi2_file)
        chi2_array = chi2_file["chi2"]
        argmin_array = np.argmin(chi2_array[:, :], axis=0)
        chi2_time_avg = 0
        for t in range(len(argmin_array)):
            chi2_time_avg += chi2_array[argmin_array[t], t]
        chi2_time_avg /= len(argmin_array)
        # print('time-avg chi2: %f' % chi2_time_avg)
        atoms_xyz_traj = np.empty((1, 4))
        # load xyz array
        xyz_array = np.load("data/%s" % xyz_array_file)["xyz"]
        for j in argmin_array:
            xyz = xyz_array[:, :, j]
            xyz = xyz.astype("|S10")  # convert to string array (max length 10)
            comment = str(j).zfill(n_zfill)
            tmp = np.array([[str(natom), "", "", ""], [comment, "", "", ""]])
            atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
            atoms_xyz = np.append(tmp, atoms_xyz, axis=0)
            atoms_xyz_traj = np.append(atoms_xyz_traj, atoms_xyz, axis=0)
        atoms_xyz_traj = atoms_xyz_traj[1:, :]  # remove 1st line of array
        fname = "data/argmin_traj_%i_%s.xyz" % (N, subtitle)
        # print('writing %s...' % fname)
        np.savetxt(
            fname,
            atoms_xyz_traj,
            fmt="%s",
            delimiter=" ",
            header="",
            footer="",
            comments="",
        )
        return chi2_time_avg

    def generate(
        self,
        title,
        subtitle,
        atomlist,
        excitation_factor,
        nstructures,
        modes,
        displacement_factors,
    ):
        """wrapper function:
        generate structures, xyz array, IAM array,
        chi2 array, argmin_trajectory, time-averaged chi2"""
        # the xyz file "xyz/title.xyz" and
        # the normal mode displacements file "nm/title_normalmodes.txt" have to exist
        starting_xyzfile = "xyz/%s.xyz" % title
        nmfile = "nm/%s_normalmodes.txt" % title
        # generate structure pool
        dist_arrays = True
        iam_arrays = True
        option = "linear"  # uniform random distribution
        directory = "xyz/generated/%s_%s_%i" % (title, subtitle, nstructures)
        os.makedirs(directory, exist_ok=True)  # create directory if doesn't exist
        nm.generate_structures(
            starting_xyzfile,
            nmfile,
            modes,
            displacement_factors,
            nstructures,
            option,
            directory,
            dist_arrays,
            iam_arrays,
            subtitle,
        )
        # create chi2 array
        iam_array_file = "iam_arrays_%i_%s.npz" % (nstructures, subtitle)
        self.chi2_(iam_array_file, nstructures, excitation_factor, subtitle)
        # create argmin trajectory
        chi2_file = "chi2_%i_%s.npz" % (nstructures, subtitle)
        xyz_array_file = "xyz_array_%i_%s.npz" % (nstructures, subtitle)
        chi2_time_avg = self.xyz_trajectory(
            atomlist, xyz_array_file, chi2_file, nstructures, subtitle
        )
        return chi2_time_avg

    def uniform_factors(self, nmodes, displacement_factors):
        """uniformly random displacement step along each mode"""
        factors = np.zeros(nmodes)
        for j in range(nmodes):
            # random factors in range [-a, a]
            a = displacement_factors[j]
            factors[j] = 2 * a * random.random_sample() - a
        return factors

    def displacements_from_wavenumbers(self, wavenumbers, step_size, exponential=False):
        nmodes = len(wavenumbers)
        displacement_factors = np.zeros(nmodes)
        for i in range(nmodes):  # initial factors are inv. prop. to wavenumber
            if wavenumbers[i] > 0:
                if exponential:
                    displacement_factors[i] = np.exp(wavenumbers[0] / wavenumbers[i])
                else:
                    displacement_factors[i] = wavenumbers[0] / wavenumbers[i]
            else:
                displacement_factors[i] = 0.0
        displacement_factors *= step_size  # adjust max size of displacement step
        return displacement_factors

    def simulate_trajectory(
        self, starting_xyz, displacements, wavenumbers, nsteps, step_size
    ):
        """creates a simulated trajectory by randomly moving along normal modes"""
        natom = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        xyz = starting_xyz  # start at starting xyz
        xyz_traj = np.zeros((natom, 3, nsteps))
        for i in range(nsteps):
            factors = self.uniform_factors(
                nmodes, displacement_factors
            )  # random factors
            xyz = nm.nm_displacer(xyz, displacements, modes, factors)
            xyz_traj[:, :, i] = xyz
        return xyz_traj

    def xyz_traj_to_iam(self, xyz_traj, qvector, reference_xyz_file="xyz/nmm.xyz"):
        """creates iam_array from xyz_traj"""
        nsteps = xyz_traj.shape[2]
        qlen = len(qvector)
        # refence IAM curve
        xyzheader, comment, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
        reference_iam = x.iam_calc(atomic_numbers, xyz, qvector)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        pcd_array = np.zeros((qlen, nsteps))
        for i in range(nsteps):
            iam = x.iam_calc(atomic_numbers, xyz, qvector)
            pcd_array[:, i] = 100 * (iam / reference_iam - 1)
        return pcd_array

    def pcd_iam(self, xyz, atomic_numbers, qvector, reference_iam):
        iam = x.iam_calc(atomic_numbers, xyz, qvector)
        return 100 * (iam / reference_iam - 1)

    def chi2_value(self, x, y):
        chi2 = np.sum((x - y) ** 2) / len(x)
        return chi2

    def rmsd_atoms(self, xyz, xyz_, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        natoms = len(indices)
        rmsd = 0.0
        for i in range(natoms):
            rmsd += np.sum((xyz[indices[i], :] - xyz_[indices[i], :]) ** 2)
        rmsd = (rmsd / natoms) ** 0.5
        return rmsd

    def rmsd_kabsch(self, xyz, xyz_, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        # first rotate xyz to have max coincidence with xyz_
        estimated_rotation, rmsd = R.align_vectors(xyz[indices, :], xyz_[indices, :])
        return rmsd, estimated_rotation

    def simulated_annealing(
        self,
        title,
        starting_xyz,
        displacements,
        wavenumbers,
        target_pcd_array,
        qvector,
        nsteps=10000,
        nruns=10,
        cutoff_value=1e-6,
        step_size=0.1,
        starting_temp=0.5,
        print_values=False,
        target_xyz_array=[],
        max_restarts=10,
        max_reverts=10,
    ):
        """simulated annealing minimisation to experiment,
        displace along each mode at each step"""
        # q vector
        qmax = qvector[-1]
        qlen = len(qvector)
        ntimesteps = target_pcd_array.shape[1]
        # create a string to define the run
        run_name_string = "%s_qmax_%2.1f_T0_%2.1f_ds_%3.2f_N_%i_Nruns_%i_nt_%i" % (
            title,
            qmax,
            starting_temp,
            step_size,
            nsteps,
            nruns,
            ntimesteps,
        )
        natoms = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        # testing: remove wavenumber damping
        w_damping = False
        h_mode_clamping = False
        if not w_damping:
            wavenumbers = np.ones(len(wavenumbers))
        if h_mode_clamping:
            wavenumbers[29:35] = 0
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        # reference IAM curve
        reference_xyz_file = "xyz/%s.xyz" % title
        _, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)
        # reference IAM up to q = 8, for comparison
        # qvector8 = np.linspace(0, 8, 79, endpoint=True)
        # reference_iam8 = x.iam_calc(atomic_numbers, reference_xyz8, qvector8)

        def displace_xyz(xyz, step_size, uniform_random=True):
            a = step_size
            if uniform_random:
                factors = self.uniform_factors(
                    nmodes, displacement_factors
                )  # random factors
            else:
                factors = np.random.choice([-1, 1], size=nmodes)
            xyz = nm.nm_displacer(xyz, displacements, modes, a * factors)
            return xyz, factors

        # start loop
        final_xyz_array = np.zeros((natoms, 3, ntimesteps))
        final_sum_distances_array = np.zeros(ntimesteps)
        final_chi2_array = np.zeros(ntimesteps)
        final_pcd_array = np.zeros((qlen, ntimesteps))
        final_pcd8_array = np.zeros((qlen, ntimesteps))
        factor_array = np.zeros(nmodes)
        for t in range(ntimesteps):
            print("time-step: %i/%i" % (t + 1, ntimesteps))
            final_chi2 = 1e9  # arbitrarily large start value
            # now we start at the final_xyz of the previous step
            target_pcd = target_pcd_array[:, t]
            run = 0
            while run < nruns:
                run += 1
                print("run: %i/%i" % (run, nruns))
                xyz = starting_xyz
                temp = starting_temp
                factor_array_ = np.zeros(nmodes)
                chi2, chi2_best = 1e9, 1e9  # arbitrarily large start value
                nrestarts, c, i = 0, 0, 0  # initiate counters
                nreverts = 0
                starting_temp_ = starting_temp
                step_size_ = step_size
                nsteps_ = nsteps
                while i < nsteps_:
                    i += 1  # count steps
                    xyz_, factors_ = displace_xyz(xyz, step_size_)
                    pcd_ = self.pcd_iam(xyz_, atomic_numbers, qvector, reference_iam)
                    chi2_ = np.sum((pcd_ - target_pcd) ** 2) / qlen
                    temp = starting_temp_ * (1 - i / nsteps_)  # decrease temperature
                    if chi2_ < chi2 or temp > random.rand():  # acceptance criteria
                        c += 1  # count acceptances
                        chi2, pcd, xyz = chi2_, pcd_, xyz_  # update values
                        factor_array_ += factors_  # check how far along each mode
                        if chi2_ < chi2_best:
                            # update values
                            chi2_best, pcd_best, xyz_best = (
                                chi2_,
                                pcd_,
                                xyz_,
                            )
                        if print_values:
                            print("temp = %f" % temp)
                            print("chi2 = %f" % chi2)
                        if chi2 < cutoff_value:
                            print("reached convergence value!")
                            nreverts = 0
                            break
                        # criteria for restarting
                        if chi2 >= cutoff_value and i > int(nsteps_ / 2):
                            nrestarts += 1
                            print(
                                "restarting (%i) run (%8.7f > %8.7f)"
                                % (nrestarts, chi2, cutoff_value)
                            )
                            i, c = 0, 0
                            starting_temp_ *= 0.5
                            print("T decreased: %8.7f" % starting_temp_)
                            if nrestarts == max_restarts:
                                step_size_ *= 0.75
                                nsteps_ = int(nsteps_ * 0.75)
                                print("Step-size decreased: %8.7f" % step_size_)
                                print("Steps decreased: %i" % nsteps_)
                                nrestarts = 0
                                nreverts += 1
                                starting_temp_ = starting_temp
                            if nreverts > max_reverts:
                                nreverts = 0
                                print("reached max_restarts. Break.")
                                break
                    elif i == nsteps_ and chi2 >= cutoff_value:
                        if c == 0:
                            print("no downhill move.")
                            step_size_ *= 0.75
                            nsteps_ = int(nsteps_ * 0.75)
                            print("Step-size decreased: %8.7f" % step_size_)
                            print("Steps decreased: %i" % nsteps_)
                        nrestarts += 1
                        print(
                            "i = N restart (%i) run (%8.7f > %8.7f)"
                            % (nrestarts, chi2, cutoff_value)
                        )
                        i, c = 0, 0
                        starting_temp_ *= 0.5
                        print("T decreased: %8.7f" % starting_temp_)
                        if nrestarts == max_restarts:
                            step_size_ *= 0.75
                            nsteps_ = int(nsteps_ * 0.75)
                            print("Step-size decreased: %8.7f" % step_size_)
                            print("Steps decreased: %i" % nsteps_)
                            nrestarts = 0
                            nreverts += 1
                            starting_temp_ = starting_temp
                        if nreverts > max_reverts:
                            nreverts = 0
                            print("reached max_restarts. Break.")
                            break
                    # end iterations loop
                # if the run improves on the previous run,
                if chi2 < final_chi2:
                    final_chi2, final_xyz, final_pcd = chi2_best, xyz_best, pcd_best
                    final_temp = temp
                    factor_array = factor_array_
                    print("chi2 best: %8.7f" % final_chi2)
                # end run loop
            # time-step loop updates:
            starting_xyz = final_xyz  # time-step t starts at geometry t-1
            # final_xyz_array[:, :, t] = final_xyz
            non_h_indices = [0, 1, 2, 3, 4, 5]
            _, r = self.rmsd_kabsch(final_xyz, target_xyz_array[:, :, t], non_h_indices)
            final_xyz_array[:, :, t] = np.dot(final_xyz, r.as_matrix())
            final_chi2_array[t] = final_chi2
            final_pcd_array[:, t] = final_pcd
            final_sum_distances_array[t] = np.sum(
                m.distances_array(final_xyz[non_h_indices]) ** 1.0
            )
            # print('sum(sqrt(distances)) = %4.3f' % final_sum_sqrt_distances_array[t])
            # final_pcd8_array[:, t] = self.pcd_iam(final_xyz, atomic_numbers, qvector8, reference_iam8)
            # end time loop
        write_final_xyz = True
        if write_final_xyz:
            m.write_xyz_traj(
                "final_%s.xyz" % run_name_string, atomlist, final_xyz_array
            )
        return (
            run_name_string,
            final_xyz_array,
            final_pcd_array,
            final_chi2_array,
            factor_array,
            final_sum_distances_array,
        )

    def simulated_annealing_v2(
        self,
        title,
        starting_xyz,
        displacements,
        wavenumbers,
        target_pcd_array,
        qvector,
        nsteps=10000,
        max_restarts=10,
        cutoff_value=1e-6,
        step_size=0.1,
        print_values=False,
        target_xyz_array=[],
        max_reverts=2,
        save_chi2_path=False,
        restart_xyz_bool=False,
        stochastic=True,
        starting_temp=0.2,
    ):
        """simulated annealing minimisation to experiment,
        displace along each mode at each step"""
        # q vector
        qmax = qvector[-1]
        qlen = len(qvector)
        ntimesteps = target_pcd_array.shape[1]
        # create a string to define the run
        run_name_string = "%s_qmax_%2.1f_ds_%3.2f_N_%i_nt_%i" % (
            title,
            qmax,
            step_size,
            nsteps,
            ntimesteps,
        )
        natoms = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        # testing: remove wavenumber damping
        w_damping = True
        h_mode_clamping = False
        if not w_damping:
            wavenumbers = np.ones(len(wavenumbers))
        if h_mode_clamping:
            wavenumbers[29:35] = 0
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        # reference IAM curve
        reference_xyz_file = "xyz/%s.xyz" % title
        _, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)

        def displace_xyz(xyz, step_size):
            return xyz, factors

        # hard coded IAM coeffs!

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
        atomic_total = np.zeros(qlen)  # total atomic factor
        atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
        for k in range(natoms):
            atomfactor = np.zeros(qlen)
            for j in range(qlen):
                for i in range(4):
                    atomfactor[j] += aa[atomic_numbers[k] - 1, i] * np.exp(
                        -bb[atomic_numbers[k] - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                    )
            atomfactor += cc[atomic_numbers[k] - 1]
            atomic_factor_array[k, :] = atomfactor
            atomic_total += atomfactor**2
        pre_molecular = np.zeros((natoms, natoms, qlen))
        for i in range(natoms):
            for j in range(natoms):
                pre_molecular[i, j, :] = np.multiply(
                    atomic_factor_array[i, :], atomic_factor_array[j, :]
                )

        # start loop
        final_xyz_array = np.zeros((natoms, 3, ntimesteps))
        final_sum_distances_array = np.zeros(ntimesteps)
        final_chi2_array = np.zeros(ntimesteps)
        final_pcd_array = np.zeros((qlen, ntimesteps))
        factor_array = np.zeros(nmodes)
        chi2_path = np.zeros(
            (nsteps * (max_restarts + 1) * (max_reverts + 1), ntimesteps)
        )
        qpi = qvector / np.pi
        for t in range(ntimesteps):
            print("time-step: %i/%i" % (t + 1, ntimesteps))
            final_chi2 = 1e9  # arbitrarily large start value
            target_pcd = target_pcd_array[:, t]
            run = 0
            xyz = starting_xyz
            factor_array_ = np.zeros(nmodes)
            chi2, chi2_best = 1e9, 1e9  # arbitrarily large start value
            nrestarts, c, i = 0, 0, 0  # initiate counters
            nreverts = 0
            starting_temp_ = starting_temp
            # step_size_ = step_size
            step_size_ = (t + 1) * step_size / ntimesteps
            nsteps_ = nsteps
            restart_xyz_bool_ = restart_xyz_bool
            stochastic_ = stochastic
            while i < nsteps_:
                i += 1  # count steps

                temp = starting_temp_ * (1 - i / nsteps_)  # decrease temperature

                # calculate random displacement factors
                factors_ = np.zeros(nmodes)
                for n in range(nmodes):
                    a = displacement_factors[n]
                    factors_[n] = temp * (2 * a * random.random_sample() - a)

                # displace the xyz coordinates
                summed_displacement = np.zeros(displacements[0, :, :].shape)
                for n in range(nmodes):
                    summed_displacement += displacements[modes[n], :, :] * factors_[n]
                xyz_ = xyz + summed_displacement

                # IAM calculation
                molecular = np.zeros(qlen)  # total molecular factor
                for ii in range(natoms):
                    for jj in range(ii + 1, natoms):  # j > i
                        molecular += pre_molecular[ii, jj, :] * np.sinc(
                            qpi * np.linalg.norm(xyz_[ii, :] - xyz_[jj, :])
                        )
                iam_ = atomic_total + 2 * molecular

                # PCD, chi2 calculations
                pcd_ = 100 * (iam_ / reference_iam - 1)
                chi2_ = np.sum((pcd_ - target_pcd) ** 2) / qlen
                if chi2_ < chi2 or temp > random.rand():
                    # accept the step
                    if abs(chi2 - chi2_) < 1e-9:
                        print(
                            "local minimum (%12.10f, %12.10f). skipping."
                            % (chi2, chi2_)
                        )
                        i = nsteps_ - 1  # go to last step
                    c += 1  # count acceptances
                    chi2, pcd, xyz = chi2_, pcd_, xyz_  # update values
                    factor_array_ += factors_  # check how far along each mode
                    if chi2_ < chi2_best:
                        # store lowest chi2 values
                        chi2_best, pcd_best, xyz_best = chi2_, pcd_, xyz_
                    if print_values:
                        print("temp = %f" % temp)
                        print("chi2 = %f" % chi2)
                    if save_chi2_path:
                        chi2_path[c - 1, t] = chi2_
                    if chi2_ < cutoff_value:
                        print("reached convergence value!")
                        break
                # criteria for restarting
                if i == nsteps_:
                    # don't restart criteria
                    if nreverts > max_reverts:
                        nreverts = 0
                        print("Reached max_reverts. End of t-step...")
                        break
                    else:
                        i, c = 0, 0
                        print(
                            "restarting (%i) run (%8.7f > %8.7f)"
                            % (nrestarts, chi2, cutoff_value)
                        )
                        if not stochastic_:
                            nrestarts = max_restarts
                            stochastic_ = True
                        if nrestarts < max_restarts:
                            nrestarts += 1
                            chi2 = 1e9  # arbitrarily large start value
                            if restart_xyz_bool_:
                                print("starting from starting_xyz...")
                                xyz = starting_xyz
                        else:
                            nrestarts = 0
                            nreverts += 1
                            restart_xyz_bool_ = False
                            if nreverts <= max_reverts:
                                starting_temp_ = 0.0
                                xyz = xyz_best
                                print("starting from xyz_best!")
                                print(
                                    "Max_restarts reached. Trying again with T0 = %3.2f"
                                    % starting_temp_
                                )
                # end iterations loop
            # if the run improves on the previous run,
            if chi2 < final_chi2:
                final_chi2, final_xyz, final_pcd = chi2_best, xyz_best, pcd_best
                final_temp = temp
                factor_array = factor_array_
                print("chi2 best: %8.7f" % final_chi2)
            # end run loop
            # time-step loop updates:
            starting_xyz = final_xyz  # time-step t starts at geometry t-1
            non_h_indices = [0, 1, 2, 3, 4, 5]
            # Kabsch rotation to target
            _, r = self.rmsd_kabsch(final_xyz, target_xyz_array[:, :, t], non_h_indices)
            final_xyz_array[:, :, t] = np.dot(final_xyz, r.as_matrix())
            final_chi2_array[t] = final_chi2
            final_pcd_array[:, t] = final_pcd
            final_sum_distances_array[t] = np.sum(
                m.distances_array(final_xyz[non_h_indices])
            )
            # end time loop
        write_final_xyz = True
        if write_final_xyz:
            m.write_xyz_traj(
                "final_%s.xyz" % run_name_string, atomlist, final_xyz_array
            )
        return (
            run_name_string,
            final_xyz_array,
            final_pcd_array,
            final_chi2_array,
            factor_array,
            final_sum_distances_array,
            chi2_path,
        )

    def stochastic_descent(
        self,
        title,
        starting_xyz,
        displacements,
        wavenumbers,
        target_pcd_array,
        qvector,
        nsteps=10000,
        nruns=10,
        cutoff_value=1e-6,
        step_size=0.1,
        print_values=False,
        target_xyz_array=[],
        max_restarts=10,
    ):
        """simulated annealing minimisation to experiment,
        displace along each mode according to 'temperature' at each step"""
        # q vector
        qmax = qvector[-1]
        qlen = len(qvector)
        ntimesteps = target_pcd_array.shape[1]
        # create a string to define the run
        run_name_string = "%s_qmax_%2.1f_ds_%3.2f_N_%i_Nruns_%i_nt_%i" % (
            title,
            qmax,
            step_size,
            nsteps,
            nruns,
            ntimesteps,
        )
        natoms = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        # testing: remove wavenumber damping
        w_damping = True
        h_mode_clamping = True
        if not w_damping:
            wavenumbers = np.ones(len(wavenumbers))
        if h_mode_clamping:
            wavenumbers[29:] = 0
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        # reference IAM curve
        reference_xyz_file = "xyz/%s.xyz" % title
        _, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)
        # reference IAM up to q = 8, for comparison
        # qvector8 = np.linspace(0, 8, 79, endpoint=True)
        # reference_iam8 = x.iam_calc(atomic_numbers, reference_xyz8, qvector8)

        def displace_xyz(xyz, step_size, uniform_random=True):
            a = step_size
            if uniform_random:
                factors = self.uniform_factors(
                    nmodes, displacement_factors
                )  # random factors
            else:
                factors = np.random.choice([-1, 1], size=nmodes)
            xyz = nm.nm_displacer(xyz, displacements, modes, a * factors)
            return xyz, factors

        # start loop
        final_xyz_array = np.zeros((natoms, 3, ntimesteps))
        final_sum_distances_array = np.zeros(ntimesteps)
        final_chi2_array = np.zeros(ntimesteps)
        final_pcd_array = np.zeros((qlen, ntimesteps))
        factor_array = np.zeros(nmodes)
        for t in range(ntimesteps):
            print("time-step: %i/%i" % (t + 1, ntimesteps))
            final_chi2 = 1e9  # arbitrarily large start value
            # now we start at the final_xyz of the previous step
            target_pcd = target_pcd_array[:, t]
            chi2_end = 0
            run = 0
            while run < nruns:
                run += 1
                print("run: %i/%i" % (run, nruns))
                xyz = starting_xyz
                factor_array_ = np.zeros(nmodes)
                chi2, chi2_best = 1e9, 1e9  # arbitrarily large start value
                i, c = 0, 0  # initiate counters
                step_size_ = step_size
                nsteps_ = nsteps
                while i < nsteps_:
                    i += 1  # count steps
                    xyz_, factors_ = displace_xyz(xyz, step_size_)
                    pcd_ = self.pcd_iam(xyz_, atomic_numbers, qvector, reference_iam)
                    chi2_ = self.chi2_value(pcd_, target_pcd)
                    # acceptance criteria
                    if chi2_ < chi2:
                        c += 1  # count acceptances
                        chi2, pcd, xyz = chi2_, pcd_, xyz_  # update values
                        factor_array_ += factors_  # record total motion along each mode
                        if chi2_ < chi2_best:
                            chi2_best, pcd_best, xyz_best = (
                                chi2_,
                                pcd_,
                                xyz_,
                            )
                        if chi2_ < cutoff_value:
                            print("reached convergence value! break")
                            break
                    # end iterations loop
                # if the run improves on the previous run,
                if chi2 < final_chi2:
                    final_chi2, final_xyz, final_pcd = chi2_best, xyz_best, pcd_best
                    factor_array = factor_array_
                    print("chi2 final: %8.7f" % final_chi2)
                # end run loop
            # time-step loop updates:
            starting_xyz = final_xyz  # time-step t starts at geometry t-1
            # final_xyz_array[:, :, t] = final_xyz
            non_h_indices = [0, 1, 2, 3, 4, 5]
            _, r = self.rmsd_kabsch(final_xyz, target_xyz_array[:, :, t], non_h_indices)
            final_xyz_array[:, :, t] = np.dot(final_xyz, r.as_matrix())
            final_chi2_array[t] = final_chi2
            final_pcd_array[:, t] = final_pcd
            final_sum_distances_array[t] = np.sum(
                m.distances_array(final_xyz[non_h_indices]) ** 1.0
            )
            # print('sum(sqrt(distances)) = %4.3f' % final_sum_sqrt_distances_array[t])
            # final_pcd8_array[:, t] = self.pcd_iam(final_xyz, atomic_numbers, qvector8, reference_iam8)
            # end time loop
        write_final_xyz = True
        if write_final_xyz:
            m.write_xyz_traj(
                "final_%s.xyz" % run_name_string, atomlist, final_xyz_array
            )
        return (
            run_name_string,
            final_xyz_array,
            final_pcd_array,
            final_chi2_array,
            factor_array,
            final_sum_distances_array,
        )

    def step_size_finder(self, xyz_start, displacements):
        natoms = xyz_start.shape[0]

        displacements = self.read_nm_displacements(nmfile, natoms)
        a = factor
        factor = np.linspace(-a, a, 20, endpoint=True)
        factor = np.append(factor, np.linspace(a, -a, 20, endpoint=True))
        _, _, atoms, xyz_start = m.read_xyz(xyz_start_file)
        for k in range(len(factor)):
            xyz = self.nm_displacer(xyz_start, displacements, mode, factor[k])

        return ds_best, chi2_path

    def simulated_annealing_v3(
        self,
        title,
        starting_xyz,
        displacements,
        wavenumbers,
        target_pcd_array,
        qvector,
        nsteps=10000,
        max_restarts=10,
        cutoff_value=1e-6,
        step_size=0.1,
        print_values=False,
        target_xyz_array=[],
        max_reverts=2,
        save_chi2_path=False,
        restart_xyz_bool=True,
        stochastic=True,
        starting_temp=0.2,
        ballistic=False,
        restart_bool=True,
    ):
        """simulated annealing minimisation to experiment,
        displace along each mode at each step"""
        # q vector
        qmax = qvector[-1]
        qlen = len(qvector)
        ntimesteps = target_pcd_array.shape[1]
        # create a string to define the run
        run_name_string = "%s_qmax_%2.1f_ds_%3.2f_N_%i_nt_%i" % (
            title,
            qmax,
            step_size,
            nsteps,
            ntimesteps,
        )
        natoms = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        # testing: remove wavenumber damping
        w_damping = False
        h_mode_clamping = False
        if not w_damping:
            wavenumbers = np.ones(len(wavenumbers))
        if h_mode_clamping:
            wavenumbers[29:] = 0
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size, True
        )
        print(displacement_factors)
        # reference IAM curve
        reference_xyz_file = "xyz/%s.xyz" % title
        _, _, atomlist, reference_xyz = m.read_xyz(reference_xyz_file)
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)

        # hard coded IAM coeffs!
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
        atomic_total = np.zeros(qlen)  # total atomic factor
        atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
        for k in range(natoms):
            atomfactor = np.zeros(qlen)
            for j in range(qlen):
                for i in range(4):
                    atomfactor[j] += aa[atomic_numbers[k] - 1, i] * np.exp(
                        -bb[atomic_numbers[k] - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                    )
            atomfactor += cc[atomic_numbers[k] - 1]
            atomic_factor_array[k, :] = atomfactor
            atomic_total += atomfactor**2
        pre_molecular = np.zeros((natoms, natoms, qlen))
        for i in range(natoms):
            for j in range(natoms):
                pre_molecular[i, j, :] = np.multiply(
                    atomic_factor_array[i, :], atomic_factor_array[j, :]
                )

        if restart_bool:
            # f = np.load('restart.npz')
            # f = np.load('r_050_stride4_data/data_chd_qmax_12.0_ds_0.05_N_6050_nt_19.npz')
            f = np.load(
                "r_055_stride4_data/data_chd_qmax_12.0_ds_0.03_N_10055_nt_18.npz"
            )

            # f = np.load('data_chd_qmax_12.0_ds_0.05_N_5909_nt_19.npz')
            ### ['step_size', 'nsteps', 'qvector', 'target_pcd_array', 'final_pcd_traj',
            ### 'final_xyz_traj', 'final_chi2_traj', 'factor_distribution', 'final_distances', 'chi2_path']
            # previous_step_size = f["step_size"]
            previous_xyz_array = f["final_xyz_traj"]
            previous_pcd_array = f["final_pcd_traj"]
            previous_chi2_array = f["final_chi2_traj"]
            # known bool array
            known = np.zeros(ntimesteps, dtype=bool)
            for k in range(len(previous_chi2_array)):
                if previous_chi2_array[k] < 1e-5:
                    known[k] = True
            print(known)
        else:
            known = np.zeros(ntimesteps, dtype=bool)
        # start loop
        final_xyz_array = np.zeros((natoms, 3, ntimesteps))
        final_distances_array = np.zeros((natoms, natoms, ntimesteps))
        final_sum_distances_array = np.zeros(ntimesteps)
        final_chi2_array = np.zeros(ntimesteps)
        final_pcd_array = np.zeros((qlen, ntimesteps))
        factor_array = np.zeros(nmodes)
        final_factor_array = np.zeros((nmodes, ntimesteps))
        chi2_path = np.zeros(
            (nsteps * (max_restarts + 1) * (max_reverts + 1), ntimesteps)
        )
        qpi = qvector / np.pi
        abs_mean_array = np.sum(np.abs(target_pcd_array), axis=0) / qlen
        tmp_arr = np.abs(np.append(abs_mean_array[1:], [0]) - abs_mean_array)
        tmp_arr = np.concatenate(([1], tmp_arr[:-1]))
        step_size_t = step_size * tmp_arr
        step_size_t = step_size * np.ones(ntimesteps)
        for t in range(0, 17):
            step_size_t[t] = 1 * step_size
        print(step_size_t)
        # known[0:11] = True
        thresh_array = np.ones(ntimesteps)
        thresh_array[0] = 2e-4
        thresh_array[1:16] = 1e-4
        thresh_array[16:20] = 5e-4
        thresh_array[20:24] = 5e-4
        thresh_array[24:28] = 1e-3
        thresh_array[28:34] = 1e-3
        thresh_array[34:42] = 5e-3
        thresh_array[42:49] = 5e-3
        thresh_array[1:30] = 1e-3
        print(thresh_array)
        for t in range(ntimesteps):
            print("time-step: %i/%i" % (t + 1, ntimesteps))
            final_chi2 = 1e9  # arbitrarily large start value
            target_pcd = target_pcd_array[:, t]
            xyz = starting_xyz
            factor_array_ = np.zeros(nmodes)
            chi2, chi2_best, chi2_best_, chi2_tmp = (
                1e9,
                1e9,
                1e9,
                1e10,
            )  # arbitrarily large start value
            nreverts, nrestarts, c, i = 0, 0, 0, 0  # initiate counters
            starting_temp_ = starting_temp
            if ballistic:
                int_len = 2
                if t <= int_len:
                    step_size_ = step_size * np.ones(nmodes)
                else:
                    t_int_factors = (
                        np.sum(final_factor_array[:, t - int_len : t], axis=1) / int_len
                    )
                    abs_factors = np.abs(t_int_factors)
                    abs_factors += step_size
                    sum_abs_factors = np.sum(abs_factors)
                    step_size_ = 1 * nmodes * step_size * abs_factors / sum_abs_factors
            else:
                step_size_ = step_size_t[t] * np.ones(nmodes)
            print(step_size_)
            nsteps_ = nsteps
            restart_xyz_bool_ = restart_xyz_bool
            max_restarts_ = max_restarts
            abs_mean = np.sum(np.abs(target_pcd)) / qlen
            abs_max = np.max(np.abs(target_pcd))
            print("ABS MAX: %5.4f" % abs_max)
            # step_size_best = step_size
            if known[t]:  # skip the known steps
                i = nsteps_
                previous = restart_bool
                if previous:
                    final_xyz = previous_xyz_array[:, :, t]
                    final_pcd = previous_pcd_array[:, t]
                    final_chi2 = previous_chi2_array[t]
                else:
                    final_xyz = target_xyz_array[:, :, t]
                    final_pcd = target_pcd_array[:, t]
                    final_chi2 = 1e-5
            while i < nsteps_:
                i += 1  # count steps

                tmp = 1 - i / nsteps_  # this is prop. to how far the molecule moves
                temp = starting_temp_ * tmp  # this is the probability of going uphill

                # calculate random displacement factors
                factors_ = np.zeros(nmodes)
                for n in range(nmodes):
                    a = displacement_factors[n]
                    factors_[n] = (
                        tmp * step_size_[n] * (2 * a * random.random_sample() - a)
                    )

                # displace the xyz coordinates
                summed_displacement = np.zeros(displacements[0, :, :].shape)
                for n in range(nmodes):
                    summed_displacement += displacements[modes[n], :, :] * factors_[n]
                xyz_ = xyz + summed_displacement

                # IAM calculation
                molecular = np.zeros(qlen)  # total molecular factor
                for ii in range(natoms):
                    for jj in range(ii + 1, natoms):  # j > i
                        molecular += pre_molecular[ii, jj, :] * np.sinc(
                            qpi * np.linalg.norm(xyz_[ii, :] - xyz_[jj, :])
                        )
                iam_ = atomic_total + 2 * molecular

                # PCD, chi2 calculations
                pcd_ = 100 * (iam_ / reference_iam - 1)
                pcd_ /= np.max(np.abs(pcd_))  # normalise to abs max
                # chi2_ = np.sum((pcd_ - target_pcd) ** 2) / (qlen * abs_mean)
                chi2_ = np.sum((pcd_ - target_pcd) ** 2) / qlen
                if chi2_ < chi2 or temp > random.rand():
                    # accept the step
                    c += 1  # count acceptances
                    chi2, pcd, xyz = chi2_, pcd_, xyz_  # update values
                    factor_array_ += factors_  # check how far along each mode
                    if chi2_ < chi2_best:
                        # store lowest chi2 values
                        chi2_best, pcd_best, xyz_best = chi2_, pcd_, xyz_
                    if chi2_ < chi2_best_:
                        # store lowest chi2 values
                        chi2_best_, pcd_best_, xyz_best_ = chi2_, pcd_, xyz_
                    if print_values:
                        print("temp = %f" % temp)
                        print("chi2 = %f" % chi2)
                    if save_chi2_path:
                        chi2_path[c - 1, t] = chi2_
                # criteria for restarting
                if i == nsteps_:
                    if nsteps_ > 11000:
                        print("too many iterations. Accepting best, breaking.")
                        final_chi2, final_xyz, final_pcd = (
                            chi2_best_,
                            xyz_best_,
                            pcd_best_,
                        )
                        final_temp = temp
                        factor_array = factor_array_
                        print("chi2 best: %8.7f" % final_chi2)
                        break
                    # step_size_ = step_size_best
                    # don't restart criteria
                    if nreverts > max_reverts:
                        nreverts = 0
                        if chi2_best < thresh_array[t]:
                            final_chi2, final_xyz, final_pcd = (
                                chi2_best,
                                xyz_best,
                                pcd_best,
                            )
                            if restart_bool:
                                if chi2_best > previous_chi2_array[t]:
                                    final_xyz = previous_xyz_array[:, :, t]
                                    final_pcd = previous_pcd_array[:, t]
                                    final_chi2 = previous_chi2_array[t]
                            final_temp = temp
                            factor_array = factor_array_
                            print("chi2 best: %8.7f" % final_chi2)
                            print("Reached max_reverts. End of t-step...")
                            break
                        else:
                            i, c = 0, 0
                            factor_array_ = np.zeros(nmodes)
                            chi2 = 1e9
                            xyz = starting_xyz
                            chi2_best = 1e9
                            # nsteps_ += int(nsteps_ * random.rand())
                            nsteps_ += int(nsteps_ * 1)
                            # step_size_ = step_size * (0.9 * random.rand() + 0.1)
                            print("RESTART becasue chi2 is >= %5.4f" % thresh_array[t])
                            # print('nsteps = %i, steps_size = %4.3f' % (nsteps_, step_size_))
                            print("nsteps = %i" % nsteps_)
                    else:
                        print(
                            "restarting (%i) run (%8.7f > %8.7f)"
                            % (nrestarts, chi2, cutoff_value)
                        )
                        i, c = 0, 0
                        factor_array_ = np.zeros(nmodes)
                        chi2 = 1e9  # arbitrarily large start value
                        if nrestarts < max_restarts_:
                            nrestarts += 1
                        else:
                            # else "revert": start from starting_xyz
                            nrestarts = 0
                            nreverts += 1
                            print("starting from starting_xyz...")
                            xyz = starting_xyz
                            restart_xyz_bool_ = False
                            if nreverts == max_reverts:
                                starting_temp_ = 0.0
                                # nsteps_ *= 2
                                # step_size_ = step_size_ / 2
                                print("nsteps = %i" % nsteps_)
                                # print(
                                #    "step-size: %3.2f, nsteps: %i"
                                #    % (step_size_, nsteps_)
                                # )
                                xyz = xyz_best
                                print("starting from xyz_best!")
                                # end run loop
            # time-step loop updates:
            starting_xyz = final_xyz  # time-step t starts at geometry t-1
            non_h_indices = [0, 1, 2, 3, 4, 5]
            # Kabsch rotation to target
            _, r = self.rmsd_kabsch(final_xyz, target_xyz_array[:, :, t], non_h_indices)
            final_xyz_array[:, :, t] = np.dot(final_xyz, r.as_matrix())
            final_chi2_array[t] = final_chi2
            final_pcd_array[:, t] = final_pcd
            final_factor_array[:, t] = factor_array
            # write xyz file at each t
            m.write_xyz(
                "out_%i.xyz" % t, "t %i" % t, atomlist, final_xyz_array[:, :, t]
            )
            # end time loop
        write_final_xyz = True
        if write_final_xyz:
            m.write_xyz_traj(
                "final_%s.xyz" % run_name_string, atomlist, final_xyz_array
            )
        if np.sum(np.sum(chi2_path)) == 0:
            chi2_path = 0
        return (
            run_name_string,
            final_xyz_array,
            final_pcd_array,
            final_chi2_array,
            final_factor_array,
            chi2_path,
        )

    def read_iam_coeffs(self):
        """returns the IAM coefficient arrays"""
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
        return aa, bb, cc

    def atomic_pre_molecular(self, atomic_numbers, qvector, aa, bb, cc):
        """both parts of IAM equation that don't depend on atom-atom distances"""
        natoms = len(atomic_numbers)
        qlen = len(qvector)
        atomic_total = np.zeros(qlen)  # total atomic factor
        atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
        for k in range(natoms):
            atomfactor = np.zeros(qlen)
            for j in range(qlen):
                for i in range(4):
                    atomfactor[j] += aa[atomic_numbers[k] - 1, i] * np.exp(
                        -bb[atomic_numbers[k] - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                    )
            atomfactor += cc[atomic_numbers[k] - 1]
            atomic_factor_array[k, :] = atomfactor
            atomic_total += atomfactor**2
        pre_molecular = np.zeros((natoms, natoms, qlen))
        for i in range(natoms):
            for j in range(natoms):
                pre_molecular[i, j, :] = np.multiply(
                    atomic_factor_array[i, :], atomic_factor_array[j, :]
                )
        return atomic_total, pre_molecular

    def simulated_annealing_v4(
        self,
        title,
        displacements,
        target_pcd,
        qvector,
        starting_temp=0.2,
        nsteps=10000,
        step_size=0.1,
    ):
        """simulated annealing minimisation to target_pcd_array"""
        ##=#=#=# DEFINITIONS #=#=#=#
        _, _, atomlist, starting_xyz = m.read_xyz('xyz/start.xyz')
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        natoms = starting_xyz.shape[0]  # number of atoms
        nmodes = displacements.shape[0]  # number of displacement vectors
        modes = list(range(nmodes))  # all modes
        qlen = len(qvector)  # length of q-vector
        aa, bb, cc = self.read_iam_coeffs()
        atomic_total, pre_molecular = self.atomic_pre_molecular(atomic_numbers, qvector, aa, bb, cc)
        qpi = qvector / np.pi  # used with np.sinc function inside loop
        _, _, atomlist, reference_xyz = m.read_xyz('xyz/reference.xyz')
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)
        ##=#=#=# END DEFINITIONS #=#=#=#

        ##=#=#=# INITIATE LOOP VARIABLES #=#=#=#=#
        xyz = starting_xyz
        i, c = 0, 0
        chi2, chi2_best = 1e9, 1e10
        ##=#=#=# END INITIATE LOOP VARIABLES #=#=#
        while i < nsteps:
            i += 1  # count steps

            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
            ##=#=#=#=# TEMPERATURE #=#=#=#=#=#=#=#=##
            tmp = 1 - i / nsteps  # this is prop. to how far the molecule moves
            temp = starting_temp * tmp  # this is the probability of going uphill
            ##=#=#=# END TEMPERATURE #=#=#=#=#=#=#=##
            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
            ##=#=#=# DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##
            # inefficient? Can probably vectorise / move some out of loop
            summed_displacement = np.zeros(displacements[0, :, :].shape)
            for n in range(nmodes):
                summed_displacement += (
                    displacements[modes[n], :, :]
                    * tmp
                    * step_size
                    * (2 * random.random_sample() - 1)
                )
            xyz_ = xyz + summed_displacement  # save a temporary displaced xyz: xyz_
            ##=#=#=# END DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##
            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
            ##=#=#=# IAM CALCULATION #=#=#=##
            molecular = np.zeros(qlen)  # total molecular factor
            for ii in range(natoms):
                for jj in range(ii + 1, natoms):  # j > i
                    molecular += pre_molecular[ii, jj, :] * np.sinc(
                        qpi * np.linalg.norm(xyz_[ii, :] - xyz_[jj, :])
                    )
            iam_ = atomic_total + 2 * molecular
            ##=#=#=# END IAM CALCULATION #=#=#=##
            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##


            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
            ##=#=#=# PCD & CHI2 CALCULATIONS #=#=#=##
            pcd_ = 100 * (iam_ / reference_iam - 1)
            pcd_ /= np.max(np.abs(pcd_))  # normalise to abs max
            chi2_ = np.sum((pcd_ - target_pcd) ** 2) / qlen
            ##=#=#=# END PCD & CHI2 CALCULATIONS #=#=#=##
            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
            ##=#=#=# ACCEPTANCE CRITERIA #=#=#=##
            if chi2_ < chi2 or temp > random.rand():
                c += 1  # count acceptances
                chi2, pcd, xyz = chi2_, pcd_, xyz_  # update values
                if chi2 < chi2_best:
                    chi2_best, pcd_best, xyz_best = chi2, pcd, xyz  # store best chi2 values
            ##=#=#=# END ACCEPTANCE CRITERIA #=#=#=##
            ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        return chi2_best, pcd_best, xyz_best

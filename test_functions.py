import numpy as np

# my own modules
import molecule

# create class objects
m = molecule.Molecule()
nm = molecule.Normal_modes()
sp = molecule.Structure_pool_method()
# define stuff
natom = 3
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
dim = 3
tcm, fcm = m.triangle_cm(atomic_numbers, xyz, dim)

# normal mode definitions
nmfile = "nm/test_normalmodes.txt"
displacements = nm.read_nm_displacements(nmfile, natom)
displacement = displacements[0, :, :]  # 1st mode displacements
factor = 1

# xray testing
x = molecule.Xray()
qlen = 100
qvector = np.linspace(0, 10, qlen, endpoint=True)  # q probably in a.u.


def test_read_xyz():
    assert xyzheader == 3, "xyzheader should be 3"
    assert comment.__contains__("test"), "comment should be 'test'"
    assert atomlist[0] == "O", "1st atom should be O"
    assert atomic_numbers[0] == 8, "1st atomic charge should be 8"
    assert xyz[0, 0] == 0.0, "Upper left coordinate should be 0.0"


def test_write_xyz():
    fname = "xyz/out.xyz"
    comment = "test"
    m.write_xyz(fname, comment, atomlist, xyz)
    with open(fname) as out:
        assert out.readline() == "3\n", "1st line of out.xyz != 3"
        assert out.readline() == "test\n", "2nd line of out.xyz != 'test'"

def test_read_xyz_traj():
    natoms, comment, atomlist, xyz_traj = m.read_xyz_traj('xyz/chd_target_traj.xyz', 12)
    fname = 'out.xyz'
    m.write_xyz_traj(fname, atomlist, xyz_traj)

test_read_xyz_traj()

def test_sort_array():
    print(atomic_numbers)
    print(xyz)
    xyz_sorted = m.sort_array(xyz, atomic_numbers)
    print(xyz_sorted)
    print(atomlist)
    atoms = m.sort_array(atomlist, atomic_numbers)
    print(atoms)
    # add assertion ...


def test_periodic_table():
    h = m.periodic_table("H")
    he = m.periodic_table("He")
    c = m.periodic_table("C")
    assert h == 1, "H should have atom number 1"
    assert he == 2, "He should have atom number 2"
    assert c == 6, "C should have atom number 2"


def test_triangle_cm():
    print("tcm")
    print(tcm)
    assert round(tcm[0, 0]) == 74, "rounded [0, 0] element != 74"
    assert tcm[0, 1] == 8, "[0, 1] element not != 8"
    assert tcm[-1, -1] == 0.5, "bottom right element != 0.5"
    assert tcm[1, 0] == 0, "bottom left diagonal != 0"


def test_full_cm():
    print("fcm")
    print(fcm)
    assert fcm[1, 0] == fcm[0, 1], "upper diagonal != lower diagonal"
    assert fcm[2, 0] == fcm[0, 2], "upper diagonal != lower diagonal"
    assert fcm[2, 1] == fcm[1, 2], "upper diagonal != lower diagonal"


def test_read_nm_displacements():
    assert displacements[0, 0, 1] == 0.07049, "displacements[0, 0, 1] != 0.07049"
    assert displacements[1, 1, 0] == 0.58365, "displacements[1, 1, 0] != 0.58365"


def test_displace_xyz():
    displaced_xyz = nm.displace_xyz(xyz, displacement, factor)
    assert displaced_xyz[1, 0] == 0.57028, (
        "displaced_xyz[1, 0] !== 0.57028, for factor %d" % factor
    )


def test_displace_write_xyz():
    displacement = displacements[0, :, :]  # 1st mode displacements
    factor = 1
    displaced_xyz = nm.displace_xyz(xyz, displacement, factor)
    fname = "xyz/displaced.xyz"
    comment = "displaced"
    m.write_xyz(fname, comment, atomlist, displaced_xyz)
    with open(fname) as out:
        assert out.readline() == "3\n", "1st line of %s != 3" % fname
        assert out.readline() == "displaced\n", "2nd line of %s != %s" % (
            fname,
            comment,
        )


def test_nm_displacer():
    factors = [1, 1, 1]
    modes = [0, 1, 2]
    displaced_xyz = nm.nm_displacer(xyz, displacements, modes, factors)
    assert round(displaced_xyz[0, 1], 5) == round(
        xyz[0, 1] + 0.07049 + 0.05016 + 0.00003, 5
    ), "displaced xyz error"
    assert round(displaced_xyz[1, 0], 5) == round(
        xyz[1, 0] - 0.42972 + 0.58365 - 0.55484, 5
    ), "displaced xyz error"


def test_atomic_factor():
    atom_number = 1  # atom_number = 1 is hydrogen, etc.
    atom_factor = x.atomic_factor(atom_number, qvector)
    assert round(atom_factor[0], 3) == 1.0, "H  atomic factor (q = 0) != 1"
    assert (
        round(x.atomic_factor(2, qvector)[0], 3) == 2.0
    ), "He atomic factor (q = 0) != 2"


def test_iam_calc():
    compton_array = x.compton_spline(
        atomic_numbers, qvector
    )  # atomic compton factors
    iam, compton = x.iam_calc(atomic_numbers, xyz, qvector, compton_array)
    assert round(iam[0], 1) == 100.0, "H2O molecular factor (q = 0) != 100"
    assert round(iam[-1], 5) == 2.4691, "H2O molecular factor (q = 10)"

def test_distances_array():
    dist_array = m.distances_array(xyz)
    assert dist_array[1, 2] == 2, "distance between hydrogens != 2"

def test_simulate_trajectory():
    xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/nmm.xyz")
    starting_xyz = xyz
    natom = xyz.shape[0]
    nsteps = 100
    step_size = 0.5
    wavenumbers = np.loadtxt('quantum/nmm_wavenumbers.dat')[:, 1]
    nmfile = "nm/nmm_normalmodes.txt"
    displacements = nm.read_nm_displacements(nmfile, natom)
    xyz_traj = sp.simulate_trajectory(starting_xyz, displacements, wavenumbers, nsteps, step_size)
    sp.xyz_traj_to_file(atomlist, xyz_traj)

#test_simulate_trajectory()

def test_simulated_annealing():
    _, _, atomlist, xyz = m.read_xyz("xyz/nmm.xyz")
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    starting_iam = x.iam_calc(atomic_numbers, xyz, qvector)
    starting_xyz = xyz
    wavenumbers = np.loadtxt('quantum/nmm_wavenumbers.dat')[:, 1]
    nmfile = "nm/nmm_normalmodes.txt"
    natom = 18
    displacements = nm.read_nm_displacements(nmfile, natom)
    # experiment percent diff
    _, _, _, xyz_displaced = m.read_xyz("xyz/nmm_displaced.xyz")
    displaced_iam = x.iam_calc(atomic_numbers, xyz_displaced, qvector)
    experiment_pcd = 100 * (displaced_iam/starting_iam - 1)
    # run sim annealing
    nsteps = 10000
    convergence_value = 0.001
    cooling_rate=4.0
    step_size=0.1
    save_xyz_path=True
    xyz_min_traj, chi2_path = sp.simulated_annealing(
        starting_xyz,
        displacements,
        wavenumbers,
        experiment_pcd,
        qvector,
        nsteps,
        convergence_value,
        cooling_rate,
        step_size,
        save_xyz_path,
    )
    save_xyz_traj_file = True
    if save_xyz_traj_file:
        fname = 'data/min_traj.xyz'
        sp.xyz_traj_to_file(atomlist, xyz_min_traj, fname)

#test_simulated_annealing()

import numpy as np

# my own modules
import molecule

# create class objects
m = molecule.Molecule()
nm = molecule.Normal_modes()
# define stuff
natom = 3
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
chargelist = [m.periodic_table(symbol) for symbol in atomlist]
dim = 3
tcm, fcm = m.triangle_cm(chargelist, xyz, dim)

# normal mode definitions
nmfile = "nm/test_normalmodes.txt"
displacements = nm.read_nm_displacements(nmfile, natom)
displacement = displacements[0, :, :]  # 1st mode displacements
factor = 1

# xray testing
x = molecule.Xray()
qlen = 80
qvector = np.linspace(0, 10, qlen, endpoint=True)  # q probably in a.u.


def test_read_xyz():
    assert xyzheader == 3, "xyzheader should be 3"
    assert comment.__contains__("test"), "comment should be 'test'"
    assert atomlist[0] == "O", "1st atom should be O"
    assert chargelist[0] == 8, "1st atomic charge should be 8"
    assert xyz[0, 0] == 0.0, "Upper left coordinate should be 0.0"


def test_write_xyz():
    fname = "xyz/out.xyz"
    comment = "test"
    m.write_xyz(fname, comment, atomlist, xyz)
    with open(fname) as out:
        assert out.readline() == "3\n", "1st line of out.xyz != 3"
        assert out.readline() == "test\n", "2nd line of out.xyz != 'test'"


def test_sort_array():
    print(chargelist)
    print(xyz)
    xyz_sorted = m.sort_array(xyz, chargelist)
    print(xyz_sorted)
    print(atomlist)
    atoms = m.sort_array(atomlist, chargelist)
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
    iam = x.iam_calc(chargelist, xyz, qvector)
    assert round(iam[0], 1) == 100.0, "H2O molecular factor (q = 0) != 100"

def test_distances_array():
    dist_array = m.distances_array(xyz)
    assert dist_array[1, 2] == 2, "distance between hydrogens != 2"


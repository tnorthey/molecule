import numpy as np
import molecule
import sys

# command line arguments
qmax = float(sys.argv[1])
qlen = int(sys.argv[2])
noise_factor = float(sys.argv[3])
step_size = float(sys.argv[4])
nsteps = int(sys.argv[5])
nruns = int(sys.argv[6])

m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sp = molecule.Structure_pool_method()
# qmax, qlen = 12.0, 119
# qmax, qlen = 2.0, 19
# qmax, qlen = 4.0, 39
qvector = np.linspace(0, qmax, qlen, endpoint=True)

title = 'chd'
_, _, atomlist, xyz = m.read_xyz("xyz/%s.xyz" % title)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
starting_iam = x.iam_calc(atomic_numbers, xyz, qvector)
starting_xyz = xyz
wavenumbers = np.loadtxt("nm/%s_wavenumbers.dat" % title)[:, 1]
nmfile = "nm/%s_normalmodes.txt" % title
natom = 14
displacements = nm.read_nm_displacements(nmfile, natom)

# "experiment" target percent diff
_, _, _, xyz_displaced = m.read_xyz("xyz/chd_target.xyz")
displaced_iam = x.iam_calc(atomic_numbers, xyz_displaced, qvector)
target_pcd = 100 * (displaced_iam / starting_iam - 1)
# add noise
# noise_factor = 0.9
delta = np.max(target_pcd) - np.min(target_pcd)
noise = noise_factor * delta
for i in range(len(target_pcd)):
    target_pcd[i] *= 1 - noise_factor * (2 * np.random.rand() - 1)

# run sim annealing
convergence_value = 0.00001
starting_temp = 0.5
save_xyz_path = True
print_values = False

# run sim annealing function
(
    chi2_path,
    rmsd_path,
    xyz_min_traj,
    final_chi2,
    final_temp,
    final_pcd,
    final_xyz,
) = sp.simulated_annealing(
    starting_xyz,
    displacements,
    wavenumbers,
    target_pcd,
    qvector,
    nsteps,
    nruns,
    convergence_value,
    step_size,
    starting_temp,
    save_xyz_path,
    print_values,
)
print(chi2_path)
c = len(chi2_path)
save_xyz_traj_file = True
if save_xyz_traj_file:
    fname = "data/min_traj.xyz"
    sp.xyz_traj_to_file(atomlist, xyz_min_traj, fname)
print(chi2_path)
print("Final chi^2 value: %f" % final_chi2)
print("Final T value: %f" % final_temp)

# save to file
data_file = "%s_data_stepsize_%3.2f_qmax_%2.1f_noise_%3.2f.npz" % (
    title,
    step_size,
    qmax,
    noise_factor,
)
np.savez(
    data_file,
    step_size=step_size,
    nruns=nruns,
    noise_factor=noise_factor,
    qvector=qvector,
    target_pcd=target_pcd,
    final_pcd=final_pcd,
    final_xyz=final_xyz,
    final_temp=final_temp,
    rmsd_path=rmsd_path,
    chi2_path=chi2_path,
    counts=c,
)

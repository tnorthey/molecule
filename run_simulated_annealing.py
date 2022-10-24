import numpy as np
import molecule
import sys

# command line arguments
qmax = float(sys.argv[1])
qlen = int(sys.argv[2])
noise_factor = float(sys.argv[3])
cooling_rate = float(sys.argv[4])
step_size = float(sys.argv[5])
nsteps = int(sys.argv[6])
nruns = int(sys.argv[7])

m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sp = molecule.Structure_pool_method()
# qmax, qlen = 12.0, 119
# qmax, qlen = 2.0, 19
# qmax, qlen = 4.0, 39
qvector = np.linspace(0, qmax, qlen, endpoint=True)

_, _, atomlist, xyz = m.read_xyz("xyz/nmm.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
starting_iam = x.iam_calc(atomic_numbers, xyz, qvector)
starting_xyz = xyz
wavenumbers = np.loadtxt("quantum/nmm_wavenumbers.dat")[:, 1]
nmfile = "nm/nmm_normalmodes.txt"
natom = 18
displacements = nm.read_nm_displacements(nmfile, natom)

# "experiment" target percent diff
_, _, _, xyz_displaced = m.read_xyz("xyz/nmm_displaced.xyz")
displaced_iam = x.iam_calc(atomic_numbers, xyz_displaced, qvector)
target_pcd = 100 * (displaced_iam / starting_iam - 1)
# add noise
# noise_factor = 0.9
delta = np.max(target_pcd) - np.min(target_pcd)
noise = noise_factor * delta
for i in range(len(target_pcd)):
    target_pcd[i] *= 1 - noise_factor * (2 * np.random.rand() - 1)

# run sim annealing
#nsteps = 10000
convergence_value = 0.00001
#cooling_rate = 4.0
#step_size = 0.1
save_xyz_path = True
print_values = False

# run multiple times
#nruns = 1
chi2_path_array = np.zeros((nsteps, nruns))
rmsd_path_array = np.zeros((nsteps, nruns))
final_pcd_array = np.zeros((qlen, nruns))
final_xyz_array = np.zeros((natom, 3, nruns))
final_temp_array = np.zeros(nruns)
counts_array = np.zeros(nruns)
for i in range(nruns):
    print("run number: %i" % i)
    (
        xyz_min_traj,
        chi2_path,
        rmsd_path,
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
        convergence_value,
        cooling_rate,
        step_size,
        save_xyz_path,
        print_values,
    )
    c = len(chi2_path)
    counts_array[i] = c
    chi2_path_array[:c, i] = chi2_path
    rmsd_path_array[:c, i] = rmsd_path
    final_pcd_array[:, i] = final_pcd
    final_xyz_array[:, :, i] = final_xyz
    final_temp_array[i] = final_temp
    save_xyz_traj_file = True
    if save_xyz_traj_file:
        fname = "data/min_traj_run%i.xyz" % i
        sp.xyz_traj_to_file(atomlist, xyz_min_traj, fname)
    print("Final chi^2 value: %f" % chi2_path[-1])

# save to file
data_file = "data_stepsize_%3.2f_gamma_%2.1f_qmax_%2.1f_noise_%3.2f.npz" % (
    step_size,
    cooling_rate,
    qmax,
    noise_factor,
)
np.savez(
    data_file,
    cooling_rate=cooling_rate,
    step_size=step_size,
    noise_factor=noise_factor,
    qvector=qvector,
    target_pcd=target_pcd,
    final_pcd_array=final_pcd_array,
    final_xyz_array=final_xyz_array,
    final_temp_array=final_temp_array,
    rmsd_path_array=rmsd_path_array,
    chi2_path_array=chi2_path_array,
    counts_array=counts_array,
)

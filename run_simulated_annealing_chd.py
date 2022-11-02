import numpy as np
import molecule
import sys

# command line arguments
qmax            = float( sys.argv[1] )
qlen            = int(   sys.argv[2] )
step_size       = float( sys.argv[3] )
starting_temp   = float( sys.argv[4] )
nsteps          = int(   sys.argv[5] )
nruns           = int(   sys.argv[6] )
ntsteps      = int(   sys.argv[7] )

m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sp = molecule.Structure_pool_method()
qvector = np.linspace(0, qmax, qlen, endpoint=True)

title = 'chd'
_, _, atomlist, xyz = m.read_xyz("xyz/%s.xyz" % title)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
starting_iam = x.iam_calc(atomic_numbers, xyz, qvector)
starting_xyz = xyz
wavenumbers = np.loadtxt("nm/%s_wavenumbers.dat" % title)[:, 1]
nmfile = "nm/%s_normalmodes.txt" % title
natom = starting_xyz.shape[0]
displacements = nm.read_nm_displacements(nmfile, natom)

# "experiment" target percent diff
target_pcd_array = np.zeros((qlen, ntsteps))
target_xyz_array = np.zeros((natom, 3, ntsteps))
target_sum_sqrt_distances = np.zeros(ntsteps)
_, _, _, target_xyz_array = m.read_xyz_traj("xyz/chd_target_traj.xyz", ntsteps)
for t in range(ntsteps):
    non_h_indices = [0, 1, 2, 3, 4, 5]
    distances = m.distances_array(target_xyz_array[non_h_indices, : , t])
    target_sum_sqrt_distances[t] = np.sum(distances**1.0)
    target_iam = x.iam_calc(atomic_numbers, target_xyz_array[:, : , t], qvector)
    target_pcd_array[:, t] = 100 * (target_iam / starting_iam - 1)

# run sim annealing
convergence_value = 1e-6
nrestarts = 50
nreverts = 1
print_values = False

# run sim annealing function
(
    run_name_string,
    final_xyz_traj,
    final_pcd_traj,
    final_chi2_traj,
    factor_distribution,
    final_sum_sqrt_distances_traj,
) = sp.simulated_annealing(
    title,
    starting_xyz,
    displacements,
    wavenumbers,
    target_pcd_array,
    qvector,
    nsteps,
    nruns,
    convergence_value,
    step_size,
    starting_temp,
    print_values,
    target_xyz_array,
    nrestarts,
    nreverts,
)

print('Row 1: target sum sqrt distances:')
print('Row 2: found')
print('Row 2: |target - found|')
print(target_sum_sqrt_distances)
print(final_sum_sqrt_distances_traj)
print(np.abs(target_sum_sqrt_distances - final_sum_sqrt_distances_traj))

# save to file
data_file = "data_%s.npz" % run_name_string
np.savez(
    data_file,
    step_size=step_size,
    nruns=nruns,
    nsteps=nsteps,
    starting_temp=starting_temp,
    qvector=qvector,
    target_pcd_array=target_pcd_array,
    final_pcd_traj=final_pcd_traj,
    final_xyz_traj=final_xyz_traj,
    final_chi2_traj=final_chi2_traj,
    factor_distribution=factor_distribution,
)

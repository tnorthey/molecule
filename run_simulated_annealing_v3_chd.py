import numpy as np
import molecule
import sys
from timeit import default_timer

start = default_timer()

# command line arguments
qmax            = float( sys.argv[1] )
qlen            = int(   sys.argv[2] )
step_size       = float( sys.argv[3] )
nsteps          = int(   sys.argv[4] )
ntsteps         = int(   sys.argv[5] )

m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sp = molecule.Structure_pool_method()
qvector = np.linspace(0, qmax, qlen, endpoint=True)

title = 'chd_t48'
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
target_sum_distances = np.zeros(ntsteps)
_, _, _, target_xyz_array = m.read_xyz_traj("xyz/chd_target_traj.xyz", ntsteps)
#reverse_time = True
reverse_time = False
if reverse_time:
    target_xyz_array = np.flip(target_xyz_array, axis=2)
    print(target_xyz_array[:, :, 0])
for t in range(ntsteps):
    non_h_indices = [0, 1, 2, 3, 4, 5]
    target_distances = m.distances_array(target_xyz_array[non_h_indices, : , t])
    target_sum_distances[t] = np.sum(target_distances)
    target_iam = x.iam_calc(atomic_numbers, target_xyz_array[:, : , t], qvector)
    target_pcd_array[:, t] = 100 * (target_iam / starting_iam - 1)
    target_pcd_array[:, t] /= np.max(np.abs(target_pcd_array[:, t]))  # normalise abs. max value to 1

# run sim annealing
convergence_value = 1e-6
nrestarts = 1
nreverts = 5
print_values = False
save_chi2_path = False

# run sim annealing function
(
    run_name_string,
    final_xyz_traj,
    final_pcd_traj,
    final_chi2_traj,
    factor_distribution,
    chi2_path,
) = sp.simulated_annealing_v3(
    title,
    starting_xyz,
    displacements,
    wavenumbers,
    target_pcd_array,
    qvector,
    nsteps,
    nrestarts,
    convergence_value,
    step_size,
    print_values,
    target_xyz_array,
    nreverts,
    save_chi2_path,
)

# found distances etc.
found_sum_distances = np.zeros(ntsteps)
found_distances_traj = np.zeros((natom, natom, ntsteps))
for t in range(ntsteps):
    found_distances = m.distances_array(final_xyz_traj[:, : , t])
    found_distances_traj[:, :, t] = found_distances
    found_sum_distances[t] = np.sum(found_distances)

print('Row 1: target sum sqrt distances:')
print('Row 2: found')
print('Row 2: |target - found|')
print(target_sum_distances)
print(found_sum_distances)
print(np.abs(target_sum_distances - found_sum_distances))

# save to file
data_file = "data_%s.npz" % run_name_string
np.savez(
    data_file,
    step_size=step_size,
    nsteps=nsteps,
    qvector=qvector,
    target_pcd_array=target_pcd_array,
    final_pcd_traj=final_pcd_traj,
    final_xyz_traj=final_xyz_traj,
    final_chi2_traj=final_chi2_traj,
    factor_distribution=factor_distribution,
    final_distances=found_distances,
    chi2_path=chi2_path,
)

print('Total time: %3.2f s' % float(default_timer() - start))

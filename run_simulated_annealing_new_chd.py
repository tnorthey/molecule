import numpy as np
import molecule
import sys

# command line arguments
qmax            = float( sys.argv[1] )
qlen            = int(   sys.argv[2] )
step_size       = float( sys.argv[3] )
nsteps          = int(   sys.argv[4] )
nruns           = int(   sys.argv[5] )
ntimesteps      = int(   sys.argv[6] )

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
target_pcd_array = np.zeros((qlen, ntimesteps))
for t in range(ntimesteps):
    _, _, _, xyz_displaced = m.read_xyz("xyz/chd_target_t%i.xyz" % t)
    displaced_iam = x.iam_calc(atomic_numbers, xyz_displaced, qvector)
    target_pcd_array[:, t] = 100 * (displaced_iam / starting_iam - 1)

# run sim annealing
convergence_value = 1e-6
starting_temp = 0.5
print_values = False

# run sim annealing function
(
    run_name_string,
    final_xyz_traj,
    final_pcd_traj,
    final_chi2_traj,
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
)

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
)

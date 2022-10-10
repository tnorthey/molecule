import numpy as np
import molecule

x = molecule.Xray()

xyz = np.array([0.0 ,0.0, 0.0])
print(xyz)
qend = 10 # au
qau = np.linspace(0, qend, 100, endpoint=True)
qend /= 0.529177249
qvector = np.linspace(0, qend, 100, endpoint=True)
print(qvector)

atomic_numbers = np.array([10])

iam = x.iam_calc(atomic_numbers, xyz, qvector)
print(iam)


q_iam = np.column_stack((qau, iam))

print(q_iam)

np.savetxt('ne_iam.dat', q_iam)

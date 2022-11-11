import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from gofee.surrogate.descriptor.fingerprint import Fingerprint

from ase.io import read

import sys

traj_name = sys.argv[1]
plot_name = sys.argv[2]
N = int(sys.argv[3])
<<<<<<< HEAD
try:
    eta = float(sys.argv[4])
except:
    eta = None

traj = read(traj_name, f':{N}')

if eta is not None:
    descriptor = Fingerprint(eta=eta)
else:
    descriptor = Fingerprint()
=======

traj = read(traj_name, f':{N}')

descriptor = Fingerprint()
>>>>>>> scale_reg
f_all = descriptor.get_featureMat(traj)

Nf = f_all.shape[1]
x = np.arange(Nf)

fig, ax = plt.subplots()
<<<<<<< HEAD
ax.set_xlabel('Descriptor coordinate')
ax.set_ylabel('Descriptor value')
=======
>>>>>>> scale_reg
ax.plot(x, f_all.T)
plt.savefig(plot_name)
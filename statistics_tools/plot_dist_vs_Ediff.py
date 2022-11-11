import numpy as np
from scipy.spatial.distance import cdist
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
=======
>>>>>>> scale_reg

traj = read(traj_name, f':{N}')
E = np.array([a.get_potential_energy() for a in traj])
print(E)
E = E.reshape(-1,1)
<<<<<<< HEAD

if eta is not None:
    descriptor = Fingerprint(eta=eta)
else:
    descriptor = Fingerprint()
=======
descriptor = Fingerprint()
>>>>>>> scale_reg
f_all = descriptor.get_featureMat(traj)

N = len(traj)

idx_triu = np.triu_indices(N,k=1)
d = cdist(f_all, f_all, metric='euclidean')[idx_triu]
Ediff = cdist(E,E,metric='cityblock')[idx_triu]

#d_p80 = np.percentile(d,80)
d_p90 = np.percentile(d,90)
d_p95 = np.percentile(d,95)
d_p98 = np.percentile(d,98)

Nf = f_all.shape[1]
x = np.arange(Nf)

fig, ax = plt.subplots()
ax.set_xlabel('Feature space distance')
ax.set_ylabel('Energy difference [eV]')
ax.scatter(d, Ediff, alpha=0.3)
ylim = ax.get_ylim()
#ax.plot([d_p80, d_p80], ylim, 'k')
ax.plot([d_p90, d_p90], ylim, 'k')
ax.plot([d_p95, d_p95], ylim, 'k')
ax.plot([d_p98, d_p98], ylim, 'k')
ax.set_ylim(ylim)
plt.savefig(plot_name)
import numpy as np

#Needed to include local python package ci_lib
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib import DecompData

path = Path(__file__).parent.parent.parent / Path('results/GN06/SVD/data.h5')
svd = DecompData.load(str(path))

temps = svd.temporals_flat #frames x comps
print(svd.spatials.shape)
spats = np.reshape(svd.spatials,(svd.spatials.shape[0],-1)) # comps x flat_img
spats = np.nan_to_num(spats) #we have Nan outside our frame from aligning the brain

temps_norm = np.linalg.norm(temps, axis=0)
spats_norm = np.linalg.norm(spats, axis=1)
np.set_printoptions(threshold=10)

print("temps norm",temps_norm,"mean",np.mean(temps_norm))
print("spats norm",spats_norm,"mean",np.mean(spats_norm))
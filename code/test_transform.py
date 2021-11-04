from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

import numpy as np
import scipy.io, scipy.ndimage


from data import DecompData
from decomposition import anatomical_parcellation

import pickle as pkl
import h5py
import sys
sys.path.append(Path(__file__).parent)





######################
# add missing h5 files here
missing_task_data = []


### New data extraction
data_path = Path(__file__).parent.parent / Path('data')
plot_path = Path(__file__).parent.parent / Path('plots')
if not (data_path/'extracted_data.pkl').exists() :
    # load behavior data
    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(root_paths=[data_path], mouse_ids=["GN06"], reextract=False)
    with open( data_path / 'extracted_data.pkl', 'wb') as handle:
        pkl.dump(sessions, handle)
else:
    # load saved data
    with open( data_path / 'extracted_data.pkl', 'rb') as handle:
        sessions = pkl.load(handle)
    print("Loaded pickled data.")

file_path = data_path / "GN06" / Path('2021-01-20_10-15-16/SVD_data/Vc.mat')
f = h5py.File(file_path, 'r')

frameCnt = np.array(f['frameCnt'])
trial_starts = np.cumsum(frameCnt[:, 1])[:-1]

mask = np.ones( len(trial_starts), dtype=bool )
mask[missing_task_data] = False
trial_starts = trial_starts[mask]


###########################

#include in loading df?
opts_path = data_path / "GN06" / Path('2021-01-20_10-15-16/SVD_data/opts.mat')
dorsal_path = data_path/"anatomical"/"allenDorsalMap.mat"
mask_path = data_path/"anatomical"/"areaMasks.mat"

trans_params = scipy.io.loadmat(opts_path,simplify_cells=True)['opts']['transParams']
dorsal_maps = scipy.io.loadmat(dorsal_path ,simplify_cells=True)['dorsalMaps']
dorsal_masks = scipy.io.loadmat(mask_path ,simplify_cells=True)['areaMasks']

svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts))

align_svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts), trans_params=trans_params)

temps, spats = anatomical_parcellation(align_svd)

anatomical = DecompData( sessions, temps, spats, np.array(trial_starts))

'''
###On reference image
ref_path = data_path/ "GN06" / Path('2021-01-20_10-15-16/reference_image.tif')
spatials = np.asarray(Image.open(ref_path),dtype='float')
org_img = spatials.copy()

h , w = spatials.shape #org shape

#Offset instead of Nans as interpolation is used
min = np.nanmin(spatials)
print(min)
eps = 2 * np.finfo(np.float32).eps
#offset = 2*eps # 10
spatials = spatials - min #+ offset
print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))
#Rotation
print("Rotation")
spatials = scipy.ndimage.rotate(spatials,trans_params['angleD'], reshape=True, cval= -eps)
print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

### introduces weird aliasing along edges due to interpolation
#Scale
print("Scale/Zoom")
spatials = scipy.ndimage.zoom(spatials, trans_params['scaleConst'],order=1,cval= -eps) #slow
print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

#Translate
print("Translate/Shift")
spatials = scipy.ndimage.shift(spatials, np.flip(trans_params['tC']),cval= -eps, order=1, mode='constant') #slow
### ---
print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

#Remove offset
spatials[spatials<0]= np.NAN

spatials = spatials + min #- offset

#Crop
print("crop")
h_new , w_new = spatials.shape
trim_h = int(np.floor((h_new - h) / 2 ))
trim_w = int(np.floor((w_new - w) / 2 ))

#Eleganter lösen, hier nur 1 zu 1 matlab nachgestellt
if trans_params['scaleConst'] < 1:
    if trim_h < 0:
        temp_spats = np.full((h, w_new),np.NAN)
        temp_spats[abs(trim_h):abs(trim_h)+h_new, :] = spatials
        spatials = temp_spats
    else:
        spatials = spatials[trim_h:trim_h + h, :]

    h_new , w_new = spatials.shape
    if trim_w < 0:
        temp_spats = np.full((h_new, w),np.NAN)
        temp_spats[:,abs(trim_w):abs(trim_w) + w_new] = spatials
        spatials = temp_spats
    else:
        spatials = spatials[:,trim_w:trim_w+w]

else:
    spatials = spatials[trim_h:trim_h + h, trim_w:trim_w+w]


_ , dorsal_w = dorsal_maps['edgeMapScaled'].shape
spatials_h , _ = spatials[:,:].shape

f, axs = plt.subplots(2)
axs[0].imshow(spatials[:,:dorsal_w], interpolation='none')

edges = dorsal_maps['edgeMapScaled']
masked_data = np.ma.masked_where(edges < 1, edges)
print(masked_data)
#cmap = ListedColormap(['white', 'black'])
#my_cmap = cmap(np.arange(cmap.N))
#my_cmap[0,-1]=0
axs[0].imshow(masked_data[:spatials_h,:], interpolation='none')
#mask = edges>0

#axs[0].imshow(np.full(edges.shape,1),alpha=edges)
axs[1].imshow(org_img[:,:])
plt.show()


'''



_ , dorsal_w = dorsal_maps['edgeMapScaled'].shape
spatials_h , _ = align_svd.spatials[0,:,:].shape

print((spatials_h, dorsal_w))

f, axs = plt.subplots(2)
axs[0].imshow(anatomical.pixel[0,:,:dorsal_w], interpolation='none')

edges = dorsal_maps['edgeMapScaled']


masked_data = np.ma.masked_where(edges < 1, edges)


axs[0].imshow(masked_data[:spatials_h,:], interpolation='none')

imgs = axs[0].get_images()
if len(imgs) > 0:
    print(imgs[0].get_clim())
#######


axs[1].imshow(align_svd.pixel[0,:,:dorsal_w],  interpolation='none')

axs[1].imshow(masked_data[:spatials_h,:], interpolation='none')

imgs = axs[1].get_images()
if len(imgs) > 0:
    print(imgs[0].get_clim())
######
#axs[1].imshow(svd.pixel[0,:,:], vmin=-0.003, vmax=0.003)


#Animation
fig, ax = plt.subplots()

ani = animation.ArtistAnimation(fig, [[ax.imshow(i, animated=True, vmin=-0.05, vmax=0.05)] for i in anatomical.pixel[:100,:,:dorsal_w]], interval=int(1000/5), blit=True,
                                repeat_delay=1000)

plt.show()

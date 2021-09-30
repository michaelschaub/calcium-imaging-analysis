from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import scipy.io, scipy.ndimage


from data import DecompData
import pickle as pkl
import h5py
import sys
sys.path.append(Path(__file__).parent)

#include in loading df?
data_path = Path(__file__).parent.parent/'data'/'GN06'/'2021-01-20_10-15-16'
opts_path = data_path/'SVD_data'/'opts.mat'

trans_params_obj = scipy.io.loadmat(opts_path,simplify_cells=True)
trans_params = trans_params_obj['opts']['transParams']


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

svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts))
align_svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts), trans_params=trans_params)

'''
ref_path = data_path/'reference_image.tif'
img = np.asarray(Image.open(ref_path),dtype='float')
org_img = img.copy()

#org shape
h , w = img.shape

#Rotation
min = np.nanmin(img)
offset = 10
img = img - min + offset
img = scipy.ndimage.rotate(img,trans_params['angleD'], reshape=True, cval= 0)

#Scale
img = scipy.ndimage.zoom(img, trans_params['scaleConst'])

#Translate
img = scipy.ndimage.shift(img, np.flip(trans_params['tC']))
img[img<0.9999*offset]= np.NAN
img = img - offset + min

#Crop
h_new , w_new = img.shape
trim_h = int(np.floor((h_new - h) / 2 ))
trim_w = int(np.floor((w_new - w) / 2 ))

if trans_params['scaleConst'] < 1:
    if trim_h < 0:
        temp_img = np.full((h, w_new),np.NAN)
        temp_img[abs(trim_h):abs(trim_h)+h_new, :] = img
        img = temp_img
    else:
        img = img[trim_h:trim_h + h, :]

    h_new , w_new = img.shape
    if trim_w < 0:
        temp_img = np.full((h_new, w),np.NAN)
        temp_img[:,abs(trim_w):abs(trim_w) + w_new] = img
        img= temp_img
    else:
        img = img[:,trim_w:trim_w+w]

else:
    img = img[trim_h:trim_h + h, trim_w:trim_w+w]
    
    
f, axs = plt.subplots(2)
axs[0].imshow(img)
axs[1].imshow(org_img)
plt.show()


#Leftover from Matlab Code
#try
#    im = im(1:540, 1:size(dorsalMaps.edgeMapScaled, 2), :)
#catch
#end
'''

print(align_svd.spatials[0,:,:])
print(svd.spatials[0,:,:])

f, axs = plt.subplots(2)
axs[0].imshow(align_svd.spatials[0,:,:]) #, vmin=0, vmax=0.002)
axs[1].imshow(svd.spatials[0,:,:], vmin=0, vmax=0.002)
plt.show()


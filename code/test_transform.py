from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy
import scipy.io, scipy.ndimage

data_path = Path(__file__).parent.parent/'data'/'GN06'/'2021-01-20_10-15-16'
ref_path = data_path/'reference_image.tif'
opts_path = data_path/'SVD_data'/'opts.mat'

img = numpy.asarray(Image.open(ref_path),dtype='float')
org_img = img.copy()
trans_params_obj = scipy.io.loadmat(opts_path,simplify_cells=True)
trans_params = trans_params_obj['opts']['transParams']

#org shape
h , w = img.shape

#Rotation
min = numpy.nanmin(img)
offset = 10
img = img - min + offset
img = scipy.ndimage.rotate(img,trans_params['angleD'], reshape=True, cval= 0)

#Scale
img = scipy.ndimage.zoom(img, trans_params['scaleConst'])

#Translate
img = scipy.ndimage.shift(img, numpy.flip(trans_params['tC']))
img[img<0.9999*offset]= numpy.NAN
img = img - offset + min

#Crop
h_new , w_new = img.shape
trim_h = int(numpy.floor((h_new - h) / 2 ))
trim_w = int(numpy.floor((w_new - w) / 2 ))

if trans_params['scaleConst'] < 1:
    if trim_h < 0:
        temp_img = numpy.full((h, w_new),numpy.NAN)
        temp_img[abs(trim_h):abs(trim_h)+h_new, :] = img
        img = temp_img
    else:
        img = img[trim_h:trim_h + h, :]

    h_new , w_new = img.shape
    if trim_w < 0:
        temp_img = numpy.full((h_new, w),numpy.NAN)
        temp_img[:,abs(trim_w):abs(trim_w) + w_new] = img
        img= temp_img
    else:
        img = img[:,trim_w:trim_w+w]

else:
    img = img[trim_h:trim_h + h, trim_w:trim_w+w]

'''
#Leftover from Matlab Code
try
    im = im(1:540, 1:size(dorsalMaps.edgeMapScaled, 2), :)
catch
end
'''

f, axs = plt.subplots(2)
axs[0].imshow(img)
axs[1].imshow(org_img)
plt.show()
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy
import scipy.io, scipy.ndimage
import sys
sys.path.append(Path(__file__).parent)

#data_path = Path(__file__).parent.parent/'data'
#file_path = data_path/'GN06'/'2021-01-20_10-15-16'/'reference_image.tif'
test_img = numpy.asarray(Image.open("C:/Master/Calcium_Analysis/data/GN06/2021-01-20_10-15-16/reference_image.tif"),dtype='float')

trans_params_obj = scipy.io.loadmat("C:/Master/Calcium_Analysis/data/GN06/2021-01-20_10-15-16/SVD_data/opts.mat",simplify_cells=True)
trans_params = trans_params_obj['opts']['transParams']

#tp_dict = dict(zip(trans_params_obj['opts']['transParams'][0,0][0][0].dtype.names, trans_params_obj['opts']['transParams'][0,0][0][0]))

#org shape
h , w = test_img.shape

#Rotation
#test_img = scipy.ndimage.rotate(test_img,numpy.asarray(trans_params['angleD'],dtype='int'), reshape=True, cval= numpy.NAN)
min = numpy.nanmin(test_img)
offset = 10
test_img = test_img - min + offset
test_img = scipy.ndimage.rotate(test_img,trans_params['angleD'], reshape=True, cval= 0)

#Scale
print(trans_params)
test_img = scipy.ndimage.zoom(test_img, trans_params['scaleConst'])

#Translate
test_img = scipy.ndimage.shift(test_img, numpy.flip(trans_params['tC']))

test_img[test_img<0.9999*offset]= numpy.NAN
test_img = test_img - offset + min

#Crop
hn , wn = test_img.shape
trim_h = numpy.floor((hn - h) / 2 )
trim_w = numpy.floor((wn - w) / 2 )

new_img = numpy.empty((h,w))

if trans_params['scaleConst'] < 1:
    if trim_h < 0:

    #pad
        test_img = numpy.pad(test_img)

else:
    #pad
    pass

plt.figure()
img = plt.imshow(test_img)
plt.show()
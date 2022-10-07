import numpy as np
import scipy.io , scipy.ndimage
import logging
LOGGER = logging.getLogger(__name__)

from ci_lib.plotting import draw_neural_activity

def align_spatials_path(spatials,trans_path,plot_alignment_path=None):
    trans_params = scipy.io.loadmat(trans_path,simplify_cells=True)['opts']['transParams']
    return align_spatials(spatials,trans_params,plot_alignment_path=plot_alignment_path)


def align_spatials(spatials,trans_params, logger=None, plot_alignment_path=None):
    logger = LOGGER if logger is None else logger.getChild(__name__)

    f , h , w = spatials.shape #org shape
    raw_spatials = np.mean(spatials[:,:,:],axis=0)

    #Append bitmap as last frame
    spatials = np.append(spatials,np.ones((1,h,w)),axis=0)
    #Set NANs from spatials to zero in bitmap
    spatials[-1, np.isnan(spatials[:-1]).all(axis=0)] = 0

    #Replace NANs with zeros
    spatials[np.isnan(spatials)] = 0

    #Rotation
    logger.info("Rotation")
    logger.debug(f"{trans_params['angleD']}")
    spatials = scipy.ndimage.rotate(spatials,trans_params['angleD'], axes=(2,1), reshape=True, cval= 0)

    #Scale
    logger.info("Scale/Zoom")
    logger.debug(f"{trans_params['scaleConst']}")
    spatials = scipy.ndimage.zoom(spatials, (1,trans_params['scaleConst'],trans_params['scaleConst']),order=1,cval= 0) #slow

    #Translate
    logger.info("Translate/Shift")
    logger.debug(f"{trans_params['tC']}")
    spatials = scipy.ndimage.shift(spatials, np.insert(np.flip(trans_params['tC']),0,0),cval= 0, order=1, mode='constant') #slow

    #Remove offset
    bitmask = spatials[-1,:,:]<0.5 #set bitmap as all elements that were interpolated under 0.5
    logger.debug(f"{bitmask.nonzero()}")
    spatials = np.delete(spatials,-1,axis=0) #delete Bitmap from spatials

    bitmask = np.broadcast_to(bitmask,spatials.shape) #for easier broadcasting, is not in memory
    np.putmask(spatials,bitmask,np.NAN) #set all elements of bitmap to NAN

    #Crop
    logger.info("Crop")
    n_spatials , h_new , w_new = spatials.shape
    trim_h = int(np.floor((h_new - h) / 2 ))
    trim_w = int(np.floor((w_new - w) / 2 ))
    logger.debug(f"trims {trim_h}, {trim_w}")

    #Eleganter lösen, hier nur 1 zu 1 matlab nachgestellt
    if trans_params['scaleConst'] < 1:
        if trim_h < 0:
            temp_spats = np.full((n_spatials, h, w_new),np.NAN)
            temp_spats[:,abs(trim_h):abs(trim_h)+h_new, :] = spatials
            spatials = temp_spats
        else:
            spatials = spatials[:,trim_h:trim_h + h, :]

        n_spatials , h_new , w_new = spatials.shape
        if trim_w < 0:
            temp_spats = np.full((n_spatials, h_new, w),np.NAN)
            temp_spats[:,:,abs(trim_w):abs(trim_w) + w_new] = spatials
            spatials = temp_spats
        else:
            spatials = spatials[:,:,trim_w:trim_w+w]

    else:
        spatials = spatials[:,trim_h:trim_h + h, trim_w:trim_w+w]

    if plot_alignment_path is not None:
        alignment_plot = draw_neural_activity(frames=np.array([raw_spatials,np.mean(spatials[:,:,:],axis=0)]),
                                            path=plot_alignment_path,
                                            plt_title="Brain Alignment", subfig_titles=["Raw","Aligned"], overlay=True, logger=logger,vmin=-0.0004,vmax=0.0004) #TODO find vmin and vmax dynamically
        return spatials, alignment_plot

    return spatials
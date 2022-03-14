import numpy as np
import scipy.ndimage
import scipy.io


from pathlib import Path
import sys
sys.path.append(Path(__file__).parent)

######################
# add missing h5 files here
missing_task_data = []

data_path = Path(__file__).parent.parent / Path('resources')
dorsal_path = data_path/"meta"/"legacy"/"allenDorsalMap.mat"
mask_path = data_path/"meta"/"legacy"/"areaMasks.mat"

dorsal_maps = scipy.io.loadmat(dorsal_path ,simplify_cells=True) ['dorsalMaps']
dorsal_labels = dorsal_maps['labelsSplit']
dorsal_masks = np.asarray(scipy.io.loadmat(mask_path ,simplify_cells=True)['areaMasks'],dtype='bool')
dorsal_side =  np.asarray(dorsal_maps['sidesSplit'])[:-1]

dorsal_edgeMap = dorsal_maps['edgeMapScaled']


def get_super(x):
    if(isinstance(x,str)):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        res = x.maketrans(''.join(normal), ''.join(super_s))
        return x.translate(res)
    else:
        return ''

dorsal_masks = np.moveaxis(dorsal_masks,-1,0)

left = [dorsal_side == 'L']
left_mask = np.flip(np.nonzero([dorsal_side == 'L'])[1])
right_mask = np.nonzero([dorsal_side == 'R'])[1]

reordered_mask = np.concatenate((right_mask,left_mask))

dorsal_dict = {
    'cortexMask': dorsal_masks[-1],
    'areaMasks': dorsal_masks[reordered_mask],
    'areaLabels': dorsal_labels[reordered_mask],
    'areaSide': dorsal_side[reordered_mask],
    'areaLabels_wSide': np.asarray([f"{s}{get_super(m)}" for s, m in zip(dorsal_labels[reordered_mask],dorsal_side[reordered_mask])]),
    'edgeMap':dorsal_edgeMap
}

dict_path = data_path/"meta"/"anatomical.mat"
scipy.io.savemat(dict_path,dorsal_dict,do_compression=True)

dorsal_labels = np.asarray(scipy.io.loadmat(dict_path ,simplify_cells=True) ['areaLabels'], dtype ='str')
dorsal_masks = np.asarray(scipy.io.loadmat(dict_path ,simplify_cells=True)['areaMasks'], dtype='bool')
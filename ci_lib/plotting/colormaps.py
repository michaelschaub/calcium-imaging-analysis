import scipy.io
import matplotlib as mpl
from pathlib import Path

def cmap_blueblackred():
    colors = scipy.io.loadmat(Path(__file__).parent/"colormap_blueblackred.mat")['map']
    return mpl.colors.LinearSegmentedColormap.from_list("ArbitraryName",colors) #,N = 1024)


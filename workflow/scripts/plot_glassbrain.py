#from nilearn import plotting
from pathlib import Path
import numpy as np
import scipy.io
import cv2

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label, LabelSet, ColumnDataSource

from shapely.geometry import Polygon


def plot_glassbrain(frame,title='',dict_path=None):
    if dict_path is None:
        data_path = Path(__file__).parent.parent.parent/"resources"
        dict_path = data_path/"meta"/"anatomical.mat"
    areaDict = scipy.io.loadmat(dict_path ,simplify_cells=True)
    bitMasks = areaDict['areaMasks']
    labels = areaDict['areaLabels_wSide']

    polygons = []
    centers = np.empty((len(bitMasks),2))

    for c,mask in enumerate(bitMasks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Saves all contour points
        polygons.append(contours)

        centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords

        '''
        for i in contours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                if(cx[c]>0): print(labels[c])
                cx[c] = (int(M['m10']/M['m00']))
                cy[c] = (int(M['m01']/M['m00']))
        '''

    #Weird nested list thats required for bokeh
    xs = [[[poly[0][:,:,0].flatten()]] for poly in polygons]
    ys = [[[poly[0][:,:,1].flatten()]] for poly in polygons]

    #Labels
    source = ColumnDataSource(data=dict(x=centers[:,0],y=centers[:,1],names=labels))
    labels = LabelSet(x='x',y='y', text='names', source=source,text_align='center')

    output_file("gfg.html")

    #Plotting
    h, w = bitMasks[0].shape
    graph = figure(title = title,plot_width=w*2, plot_height=h*2 )

    # color values of the poloygons
    #color = ["red", "purple", "yellow"]

    # fill alpha values of the polygons
    fill_alpha = 0.1

    # plotting the graph and add labels
    graph.multi_polygons(xs, ys, fill_alpha = fill_alpha)
    graph.add_layout(labels)
    graph.image(image=[frame],x=0,y=0,dw=0,dh=0)

    # displaying the model
    show(graph)




plot_glassbrain(None)
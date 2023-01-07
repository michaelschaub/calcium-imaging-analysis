import cv2
from shapely.geometry import Polygon

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import numpy as np

import sys;

def polygons_from_mask(masks,type="polygons"):
    if masks.ndim == 2:
        masks = [masks]
    polygons = np.empty((len(masks)),dtype=object)
    contours = np.empty((len(masks)),dtype=object)
    centers = np.empty((len(masks),2))

    for c,mask in enumerate(masks):
        #Partitial credit https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib/56140178#56140178

        contours[c], _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Finds all contour points

        #Convert to 2D contours
        #check if ndim <= 2

        try:
            contours[c] = np.squeeze(contours[c])
            #print(contours[c].shape)
            polygons[c] = Polygon((contours[c]))
        except ValueError as err:
            #cv2.findContours is incosistent in the dim of it's return values
            #sometimes additional dim?
            #print(contours[c][0].shape)
            #print(contours[c][1].shape)
            contours[c] = np.squeeze(np.concatenate(contours[c]))
            #print(contours[c].shape)
            #for contour in contours[c]]
            polygons[c] = Polygon((contours[c]))

        #print(contours[c].shape)
        #polygons[c] = Polygon((contours[c]))
        centers[c,:] = polygons[c].centroid.coords

    if "polygon" in type:
        return polygons
    elif "contour" in type:
        return contours
    else:
        return polygons, contours, centers

# Plots a Polygon to pyplot `ax`
def plt_polygons(ax, polys, fill=False, **kwargs):
    #if not isinstance(polys,list):
    #    polys = [polys]
    #    print("not a list")

    for poly in polys:
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

        
        if not fill:
            kwargs["facecolor"] = (0,0,0,0) 
        patch = PathPatch(path, fill=fill, **kwargs)
        collection = PatchCollection([patch], **kwargs)
        
        ax.add_collection(collection, autolim=True)
        ax.autoscale_view()

    return collection

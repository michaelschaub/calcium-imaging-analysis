import networkx as nx
import numpy as np
import math

import scipy.io
import cv2

from bokeh.plotting import figure, output_file, save, show, from_networkx
from bokeh.models import (Label, LabelSet, HoverTool, BoxSelectTool,  TapTool, ColumnDataSource, LinearColorMapper,
                          Circle, EdgesAndLinkedNodes, NodesAndLinkedEdges, MultiLine, Patch,MultiPolygons,Patches , CustomJS)
from bokeh.palettes import Viridis, Viridis256, Spectral4

from shapely.geometry import Polygon
from scipy.ndimage import center_of_mass

from pathlib import Path

#Needs to be restructured badly
def plot_glassbrain_bokeh(graph=None, img=None,meta_file=None,
                        components_spatials=None,components_labels=None,components_pos=None,
                        title='',save_path=None,small_file=False):



    #center for components
    if components_pos is None:
        components_pos = calc_center(components_spatials)

    #label for components
    if components_labels is None:
        digits = math.floor(math.log(len(components_pos), 10))+1
        components_labels = np.array([str(i).zfill(digits) for i in range(len(components_pos))])

    #Load fallback
    if meta_file is None:
        meta_file = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
    metaDict = scipy.io.loadmat(meta_file ,simplify_cells=True)
    anatomicalMasks, anatomicalLabels = metaDict['areaMasks'],metaDict['areaLabels_wSide']

    #Calc poly, center, labels,
    anatomicalPolygons, anatomicalCenters = calc_polygons(anatomicalMasks)
    labelset = create_labelset(anatomicalCenters,anatomicalLabels)

    #Bounding Box
    poly_bbox = list(anatomicalMasks[0].shape)
    img_bbox =list(img.shape) if img is not None else np.zeros((2),dtype=int)
    y_range, x_range= tuple(np.maximum(poly_bbox, img_bbox,dtype=int))

    #Plot Glassbrain
    draw_glassbrain(anatomicalPolygons,labelset, graph, img, components_pos=components_pos, components_labels=components_labels,components_spatials=components_spatials, x_range=x_range, y_range=y_range, scale=2,title=title, save_path=save_path, small_file=small_file)

# Helper Functions
def calc_center(spatials):
    spatials = np.nan_to_num(spatials)

    center = np.empty((len(spatials),2))
    for i,component in enumerate(spatials):
        component[component<0]=0 #only use excitatory activity
        center[i,:] = center_of_mass(component)
        #center[i,:] = center_of_mass(np.absolute(component))
    return np.flip(center,axis=1)

def calc_polygons(masks):
    polygons = []
    centers = np.empty((len(masks),2))

    for c,mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Finds all contour points
        polygons.append(contours)
        centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords

    return polygons, centers

def format_polygons(polys,labels):
    xs = [[[poly[0][:,:,0].flatten()]] for poly in polys]
    ys = [[[poly[0][:,:,1].flatten()]] for poly in polys]
    return ColumnDataSource(data=dict(xs=xs, ys=ys,name=labels))

def create_labelset(centers, labels):
    source = ColumnDataSource(data=dict(x=centers[:,0],y=centers[:,1],names=labels))
    return LabelSet(x='x',y='y', text='names', source=source,text_align='center',text_color='black')


def draw_glassbrain(polygons=None,labelset=None,graph=None,bg_img=None,
                    components_pos=None, components_labels=None, components_spatials=None,
                    x_range=None,y_range=None,scale=1,
                    title='',save_path=None,small_file=False):

    width, height = x_range * scale,y_range * scale
    fig = figure(title = title,plot_width=width, plot_height=height ,
                 x_range=[0,x_range], y_range=[y_range,0])

    #Labels
    #spatial_labels = labelset.source.data["names"] if labelset is not None else [str(i).zfill(3) for i in range(len(components_pos))]

    #Background Image
    if bg_img is not None:
        colors= LinearColorMapper(palette=Viridis256,nan_color=(0,0,0,0))
        img_h, img_w = bg_img.shape
        fig.image(image=[np.flipud(bg_img)],x=0,y=img_h,dw=img_w,dh=img_h,color_mapper=colors)

    #Polygons with Labels
    if None not in (polygons,labelset):
        poly_glyph = MultiPolygons(xs="xs", ys="ys", line_color='black', line_width=2, line_alpha=0.5, fill_alpha=0.05)
        poly_render = fig.add_glyph(format_polygons(polygons,labelset.source.data["names"]), poly_glyph)
        #label_render = fig.add_layout(labelset)

        fig.add_tools(HoverTool(renderers=[poly_render],tooltips="@name"))

    #Graph
    if graph is not None:
        #Currently only center for anatomical
        nodes = list(graph.nodes)
        #centers = list(zip(labelset.source.data["x"][nodes] , labelset.source.data["y"][nodes]))
        centers = components_pos[nodes,:]
        layout_nodes = dict(zip(nodes,centers))

        #Add labels to edges & nodes
        edges_labels = {k: [] for k in ['connection','source','target']}
        node_labels = np.full((len(nodes),2 if nx.is_directed(graph) else 1),"",dtype='O')
        for e in graph.edges():
            src , trgt = e
            if not nx.is_directed(graph) or graph.has_edge(trgt,src):
                edge_sym,node_sym = " ⬌ ", ""
            else:
                edge_sym,node_sym = " ➔ ", "➔"
            edges_labels['connection'].append(edge_sym.join([components_labels[src],components_labels[trgt]]))
            edges_labels['source'].append(f"{components_labels[src]} {node_sym}")
            edges_labels['target'].append(f"{node_sym} {components_labels[trgt]}")

            node_labels[nodes.index(src),0] += f"{components_labels[trgt]},\n"
            node_labels[nodes.index(trgt),1 if nx.is_directed(graph) else 0] += f"{components_labels[src]},\n"

        #Create renderer based on graph
        graph_renderer = from_networkx(graph, layout_nodes, node_attrs=['color','selected'],scale=1)

        #Prepare alpha values of nodes
        selected_dict = nx.get_node_attributes(graph,'selected')
        selected_nodes = [k for k,v in selected_dict.items() if v] #np.asarray((nx.get_node_attributes(graph,'selected')).values(),dtype=bool)
        alpha_nodes = np.full(len(components_pos),0.5) #TODO Hardcoded number of comps
        alpha_nodes[selected_nodes] = 1

        #add interaction tools to edges
        fig.add_tools(HoverTool(renderers=[graph_renderer.edge_renderer],tooltips="@connection",line_policy='interp'))
        fig.add_tools(HoverTool(renderers=[graph_renderer.edge_renderer],tooltips="@source",line_policy='prev'))
        fig.add_tools(HoverTool(renderers=[graph_renderer.edge_renderer],tooltips="@target",line_policy='next'))
        fig.add_tools(HoverTool(tooltips=None), BoxSelectTool())

        #Pass attributes to node_renderer
        graph_renderer.node_renderer.data_source.add(alpha_nodes[nodes],'alpha_nodes')
        graph_renderer.node_renderer.data_source.add(components_labels[nodes],"label")
        graph_renderer.node_renderer.data_source.add(components_spatials[nodes],"component_spatials")

        #add interaction tools to nodes
        if nx.is_directed(graph) and graph.edges():
            graph_renderer.node_renderer.data_source.add(node_labels[:,1],"inc")
            graph_renderer.node_renderer.data_source.add(node_labels[:,0],"out")
            fig.add_tools(HoverTool(renderers=[graph_renderer.node_renderer],tooltips=[("Node","@label"),("🢂◯","@inc"),("◯🢂","@out")]))
            #fig.add_tools(TapTool(renderers=[graph_renderer.node_renderer],tooltips=[("Comp","@label"),("Inc","@inc"),("Out","@out")]))

        elif graph.edges():
            graph_renderer.node_renderer.data_source.add(node_labels[:,0],"connected")
            fig.add_tools(HoverTool(renderers=[graph_renderer.node_renderer],tooltips=[("Node","@label"),("🢀🢂","@connected")]))




        #Set node glyphs
        graph_renderer.node_renderer.glyph = Circle(size="node_size" , fill_color="color", fill_alpha="alpha_nodes", tags=["node"], name = "node") #Patch
        graph_renderer.node_renderer.selection_glyph = Circle(size="node_size", fill_color=Spectral4[2], tags=["node"],name = "node")
        graph_renderer.node_renderer.hover_glyph = Circle(size="node_size", fill_color=Spectral4[1], tags=["node"],name = "node")


        #Background Spatials
        colors= LinearColorMapper(palette=Viridis256,nan_color=(0,0,0,0))
        bg_source=ColumnDataSource(data=dict(image=[np.flipud(components_spatials[0])])) #might break for bool spatials

        _ , img_h, img_w = components_spatials.shape
        #bg_fig = fig.image(image='image',source=bg_source,x=0,y=img_h,dw=img_w,dh=img_h,color_mapper=colors)

        if not small_file:
            bg_imgs = []
            for i,spats in enumerate(components_spatials[nodes]):
                bg_imgs.append(fig.image(image=[np.flipud(components_spatials[nodes][i])],x=0,y=img_h,dw=img_w,dh=img_h,color_mapper=colors))
                bg_imgs[i].visible = False

            change_bg_callback = CustomJS(args={'bg_imgs':bg_imgs,'source':bg_source,'spatials':components_spatials[nodes]}, code="""
                
                //var img = source.data['image'][0];
                //img = spatials[index];
                //console.log(index,img)
                //source.change.emit();
                //bg_img.visible = false
                //console.log("this",bg_imgs)
                
                bg_imgs.forEach(img => img.visible=false)
                
                
                
                if(cb_data.source.data.alpha_nodes){ //should be cb_obj.source.name or cd_data.name == "Node" , but it doesn't work
                    var index = cb_data.source.selected.indices[0];
                    console.log(cb_obj)
                console.log(cb_data)
                    if(index < bg_imgs.length){
                        bg_imgs[index].visible=true
                    }
                }
                
            """)


            '''            
                '''

            #fig.add_tools(TapTool(renderers=[graph_renderer.node_renderer],callback=change_bg_callback))
            fig.add_tools(TapTool(callback=change_bg_callback))

        ''''''
        #def callback(attr,old,new):
        #    logger.info("hello")

        #graph_renderer.node_renderer.selected.on_change('indices', change_bg_callback)


        #taptool = fig.select(type=TapTool)
        #taptool.callback = change_bg_callback



    #Pass attributes to edge_renderer
        graph_renderer.edge_renderer.data_source.add(edges_labels["connection"],"connection")
        graph_renderer.edge_renderer.data_source.add(edges_labels["source"],"source")
        graph_renderer.edge_renderer.data_source.add(edges_labels["target"],"target")

        #set edge glyphs
        #line_cap = round , line_join  = "round"
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha="edge_alpha", line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)



        #Select Nodes and hover Edges
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    #Polygons with Labels
    #Layered between background and graph
    if None not in (polygons,labelset):
        poly_glyph = MultiPolygons(xs="xs", ys="ys", line_color='black', line_width=2, line_alpha=0.5, fill_alpha=0.05)
        poly_render = fig.add_glyph(format_polygons(polygons,labelset.source.data["names"]), poly_glyph)
        #label_render = fig.add_layout(labelset)

        fig.add_tools(HoverTool(renderers=[poly_render],tooltips="@name"))

    #Graph
    if graph is not None:
        fig.renderers.append(graph_renderer)


    # displaying & saving the model
    if save_path is not None:
        output_file(filename=save_path, title=title)
        save(fig)
    show(fig)

import matplotlib.pyplot as plt
import numpy as np


def colored_violinplot(*args, color=None, facecolor=None, edgecolor=None, **kwargs):
    violin_parts = plt.violinplot(*args, **kwargs)
    for part in violin_parts:
        # for pc in violin_parts['bodies']:
        parts = violin_parts[part] if part == 'bodies' else [violin_parts[part]]
        for pc in parts:
            if color is not None:
                pc.set_color(color)
            if facecolor is not None:
                pc.set_facecolor(facecolor)
            if edgecolor is not None:
                pc.set_edgecolor(edgecolor)
    return violin_parts


##Assumes that spatial is identical for all given temps
def plot_frame(temps, spatial, titles, plt_title):
    width = int(np.ceil(np.sqrt(len(temps))))
    height = int(np.ceil(len(temps) / width))
    #if height == 1: height = 2 #workaround for the moment as ax[h,w] wont work
    fig, ax = plt.subplots(height , width, constrained_layout=True, squeeze=False)
    fig.suptitle(plt_title)
    for h in range(height):
        for w in range(width):
            if h*width + w < len(temps):
                frame =  np.tensordot(temps[h*width + w], spatial, 1) #np.einsum( "n,nij->ij", temps[h*width + w], spatial) #np.tensordot(temps[w + h], spatial, (-1, 0)) #np.dot(spatial,temps[w*height + h]) #
                im = ax[h, w].imshow(frame)

                fig.colorbar(im, ax=ax[h, w])
                ax[h, w].set_title(titles[h*width + w])
                ax[h, w].set_xticks([])
                ax[h, w].set_yticks([])
                #plt.draw()
                #plt.pause(0.1)
    #plt.show()

    plt.savefig(plt_title, format='png')
    print("plotted")
import matplotlib.pyplot as plt

def colored_violinplot( *args, color=None, facecolor=None, edgecolor=None, **kwargs ):
    violin_parts = plt.violinplot( *args, **kwargs )
    for part in violin_parts:
        #for pc in violin_parts['bodies']:
        parts = violin_parts[part] if part == 'bodies' else [violin_parts[part]]
        for pc in parts:
            if color is not None:
                pc.set_color(color)
            if facecolor is not None:
                pc.set_facecolor(facecolor)
            if edgecolor is not None:
                pc.set_edgecolor(edgecolor)
    return violin_parts
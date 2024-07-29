import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
from mne_connectivity import viz
import plotly.graph_objects as go
import pandas as pd
import matplotlib.collections as mcoll
from datashader.bundling import hammer_bundle
import eelbrain as eel

################################################################################
# Heatmap
################################################################################

def heatmap_linkmatrix(self, lkm1, lkm2, status1='C1', status2='C2', hemi=False, 
                     overlay_nums=False, vmin=0, vmax=1, diffvmin=-1, diffvmax=1, 
                     figsize=(10, 3), hemimapping={'lh':'lh', 'rh':'rh'}, cbar1scaling=None, 
                     cbar2scaling=None, rotate_labels=False):
    if hemi == False:
        c1_group_avg = np.copy(lkm1).T
        c2_group_avg = np.copy(lkm2).T

        fig, ax = plt.subplots(1, 3, figsize=figsize)

        if cbar1scaling is not None:
            im = ax[0].imshow(c1_group_avg, norm=cbar1scaling)
        else:
            im = ax[0].imshow(c1_group_avg, vmin=vmin, vmax=vmax)

        ax[0].set_title(f"{status1}")

        if rotate_labels:
            ax[0].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
        else:
            ax[0].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
        ax[0].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
        
        if overlay_nums == True:
            for srcidx, source in enumerate(self.target_lobes):
                for dstidx, target in enumerate(self.target_lobes):
                    ax[0].text(srcidx, dstidx, 
                        round(c1_group_avg[dstidx, srcidx], 2), ha="center", va="center", color="w")

        if cbar1scaling is not None:
            ax[1].imshow(c2_group_avg, norm=cbar1scaling)
        else:
            ax[1].imshow(c2_group_avg, vmin=vmin, vmax=vmax)

        ax[1].set_title(f"{status2}")
        
        if rotate_labels:
            ax[1].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
        else:
            ax[1].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
        ax[1].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
        
        if overlay_nums == True:
            for srcidx, source in enumerate(self.target_lobes):
                for dstidx, target in enumerate(self.target_lobes):
                    ax[1].text(srcidx, dstidx, 
                        round(c2_group_avg[dstidx, srcidx], 2), ha="center", va="center", color="w")

        if cbar2scaling is not None:
            im2 = ax[2].imshow(c1_group_avg - c2_group_avg, cmap='seismic', norm=cbar2scaling)
        else:
            im2 = ax[2].imshow(c1_group_avg - c2_group_avg, vmin=diffvmin, vmax=diffvmax, cmap='seismic')

        ax[2].set_title(f"{status1}-{status2}")
        
        if rotate_labels:
            ax[2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
        else:
            ax[2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
        ax[2].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
        
        if overlay_nums == True:
            for srcidx, source in enumerate(self.target_lobes):
                for dstidx, target in enumerate(self.target_lobes):
                    ax[2].text(srcidx, dstidx, 
                        round(c1_group_avg[dstidx, srcidx]\
                                -c2_group_avg[dstidx, srcidx], 2), ha="center", va="center", color="w")
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.0075, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.0075, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.show()

    ###################################
    else:
        c1_group_avg_hemi = np.copy(lkm1)
        c2_group_avg_hemi = np.copy(lkm2)
        for i in range(4):
            c1_group_avg_hemi[i] = lkm1[i].T
            c2_group_avg_hemi[i] = lkm2[i].T
        
        fig, ax = plt.subplots(2, 6, figsize=figsize)

        for srchemi, dsthemi, i, j in [('lh', 'lh', 0, 0), ('rh', 'rh', 1, 1), 
                                        ('lh', 'rh', 0, 1), ('rh', 'lh', 1, 0)]:
            hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
            if cbar1scaling is not None:
                im = ax[i, j].imshow(c1_group_avg_hemi[hemi_idx], norm=cbar1scaling)
            else:
                im = ax[i, j].imshow(c1_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)

            s = hemimapping[srchemi]
            d = hemimapping[dsthemi]
            
            ax[i, j].set_title(f"{status1}:{s}->{d}")

            if rotate_labels:
                ax[i, j].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
            else:
                ax[i, j].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[i, j].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[i, j].text(srcidx, dstidx, 
                            round(c1_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")

            if cbar1scaling is not None:
                ax[i, j+2].imshow(c2_group_avg_hemi[hemi_idx], norm=cbar1scaling)
            else:
                ax[i, j+2].imshow(c2_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)
            
            s = hemimapping[srchemi]
            d = hemimapping[dsthemi]
            ax[i, j+2].set_title(f"{status2}:{s}->{d}")
            
            if rotate_labels:
                ax[i, j+2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
            else:
                ax[i, j+2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[i, j+2].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[i, j+2].text(srcidx, dstidx, 
                            round(c2_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")
            if cbar2scaling is not None:
                im2 = ax[i, j+4].imshow(c1_group_avg_hemi[hemi_idx] \
                                - c2_group_avg_hemi[hemi_idx],
                                cmap='seismic', norm=cbar2scaling)
            else:
                im2 = ax[i, j+4].imshow(c1_group_avg_hemi[hemi_idx] \
                                - c2_group_avg_hemi[hemi_idx],
                                vmin=diffvmin, vmax=diffvmax, cmap='seismic')
            s = hemimapping[srchemi]
            d = hemimapping[dsthemi]
            ax[i, j+4].set_title(f"{status1}-{status2}:{s}->{d}")

            if rotate_labels:
                ax[i, j+4].set_xticks(list(range(len(self.target_lobes))), self.target_lobes, rotation='vertical')
            else:
                ax[i, j+4].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[i, j+4].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[i, j+4].text(srcidx, dstidx, 
                            round(c1_group_avg_hemi[hemi_idx, dstidx, srcidx]\
                                    -c2_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.15, 0.005, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.005, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        plt.show()
    

def heatmap(self, condition1, condition2, status1='C1', status2='C2', hemi=False, norm=False,
                     overlay_nums=False, vmin=0, vmax=1, diffvmin=-1, diffvmax=1,
                     figsize=(10, 3), hemimapping={'lh':'lh', 'rh':'rh'}, cbar1scaling=None, 
                     cbar2scaling=None, rotate_labels=False):
    ###################################
    if hemi == False:
        if norm:
            c1_group_avg, c2_group_avg = self.group_whole_average(condition1, condition2, matnorm=True)
        else:
            c1_group_avg, c2_group_avg = self.group_whole_average(condition1, condition2, matnorm=False)

        self.heatmap_linkmatrix(c1_group_avg, c2_group_avg, status1, status2, hemi, 
                     overlay_nums, vmin, vmax, diffvmin, diffvmax, figsize, hemimapping, 
                     cbar1scaling, cbar2scaling, rotate_labels)
        return c1_group_avg, c2_group_avg
    else:
        if norm:
            c1_group_avg_hemi, c2_group_avg_hemi = self.group_hemi_average(condition1, condition2, matnorm=True)
        else:
            c1_group_avg_hemi, c2_group_avg_hemi = self.group_hemi_average(condition1, condition2, matnorm=False)

        self.heatmap_linkmatrix(c1_group_avg_hemi, c2_group_avg_hemi, status1, status2, hemi, 
                     overlay_nums, vmin, vmax, diffvmin, diffvmax, figsize, hemimapping, 
                     cbar1scaling, cbar2scaling, rotate_labels)

        return c1_group_avg_hemi, c2_group_avg_hemi


################################################################################
# Circle Plot
################################################################################

def circle_plot(self, condition1, condition2, status1='C1', status2='C2', hemi=False):
    ###################################
    if hemi == False:
        c1_group_avg, c2_group_avg = self.group_whole_average(condition1, condition2)
        
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        for i in range(3):
            ax[i] = plt.subplot(130+(i+1), projection='polar')

        viz.plot_connectivity_circle(c1_group_avg, self.target_lobes, ax=ax[0], show=False)
        ax[0].set_title(f"{status1}")


        viz.plot_connectivity_circle(c2_group_avg, self.target_lobes, ax=ax[1], show=False)
        ax[1].set_title(f"{status2}")

        viz.plot_connectivity_circle(c1_group_avg - c2_group_avg, self.target_lobes, ax=ax[2], show=False)
        ax[2].set_title(f"{status1}-{status2}")
        plt.show()

    ###################################
    else:
        c1_group_avg_hemi, c2_group_avg_hemi = self.group_hemi_average(condition1, condition2)
            
        
        fig, ax = plt.subplots(2, 6, figsize=(20, 7))
        for row in range(2):
            for col in range(6):
                ax[row, col] = plt.subplot(2,6, 0+(6*row+col+1), projection='polar')

        for srchemi, dsthemi, i, j in [('lh', 'lh', 0, 0), ('rh', 'rh', 1, 1), 
                                        ('lh', 'rh', 0, 1), ('rh', 'lh', 1, 0)]:
            hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
            viz.plot_connectivity_circle(c1_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j], show=False)
            ax[i, j].set_title(f"{status1}:{srchemi}->{dsthemi}")

            viz.plot_connectivity_circle(c2_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j+2], show=False)
            ax[i, j+2].set_title(f"{status2}:{srchemi}->{dsthemi}")
            
            
            viz.plot_connectivity_circle(c1_group_avg_hemi[hemi_idx] \
                                - c2_group_avg_hemi[hemi_idx], self.target_lobes, ax=ax[i, j+4], show=False)
            ax[i, j+4].set_title(f"{status1}-{status2}:{srchemi}->{dsthemi}")

        plt.show()


################################################################################
# Railroad Plot
################################################################################

def _get_vertices(n, rad, offset, centerxy=(0.5, 0.5)):
    # generate n vertices evenly spaced along a circle with radius rad
    # offset by offset degrees
    increment = 360 / n
    xs = []
    ys = []
    thetas = []
    for i in range(n):
        theta = i*increment
        thetas.append(theta)
        x = np.round(rad*np.cos(theta*np.pi/180), 2)
        y = np.round(rad*np.sin(theta*np.pi/180), 2)
        xs.append(x+centerxy[0])
        ys.append(y+centerxy[1])
    return xs, ys, [rad]*len(xs), thetas

# @staticmethod
# def hist_midpoint(ts):
    # helper function for trace_bezier_midpoint
#     halfdiff = (np.max(list(ts.keys())) - np.min(list(ts.keys())))/2
#     midpt =  np.min(list(ts.keys())) + halfdiff
#     idx = np.argmin(np.abs(np.array(list(ts.keys())) - midpt))
#     return ts[np.array(list(ts.keys()))[idx]]


def _trace_bezier_midpoint(arrow):
    path = arrow.get_path().vertices

    ## used the method below to find the optimal bezier t, but it is finnicky,
    ## so i'll use the best t it found and leave the commented code for future runs
    #####################################################
#     xs = {}
#     ys = {}
#     for t in np.linspace(0.0, 1, 300):
#         i, j = mpl.bezier.BezierSegment(path).point_at_t(t)
#         xs[i] = t
#         ys[i] = t
        
#     txfinal = hist_midpoint(xs)
#     tyfinal = hist_midpoint(ys)
#     tfinal = np.mean([txfinal, tyfinal])
#     print(txfinal, tyfinal)
    #####################################################0.7357859531772575
    return *mpl.bezier.BezierSegment(path).point_at_t(0.72), \
            *mpl.bezier.BezierSegment(path).point_at_t(0.7357859531772575)


def _push_wedge(factor, wx, wy, centerxy=(0.5, 0.5)):
    cx, cy = centerxy
    # radially push the wedge outwards from center
    
    # center the coords at origin     
    wx = wx - cx
    wy = wy - cy
    wx = factor*wx + cx
    wy = factor*wy + cy
    return wx, wy
        

def _add_connection(ax, posa, posb, bidirectional=False, inner=True, arc=0.2, 
                pushfactor={'inner':0.75, 'outer':1.15}, centerxy=(0.5, 0.5),
                color={'inner':1, 'outer':1}):
    inner = 2*inner-1
    style = "Simple, tail_width=6, head_width=4, head_length=8"
    kw = dict(arrowstyle=style)


    arrow = mpatches.FancyArrowPatch(posa, posb,
                                connectionstyle=f"arc3,rad={-int(inner)*arc}", color=color['inner'], **kw)

    ax.add_patch(arrow)
    
    
    mid_dx, mid_dy, midx, midy = _trace_bezier_midpoint(arrow)
    mid_dx, mid_dy = _push_wedge(pushfactor['inner'], mid_dx, mid_dy, centerxy)
    midx, midy = _push_wedge(pushfactor['inner'], midx, midy, centerxy)

#     ax.arrow(midx, midy, 0.001*(mid_dx-midx), 0.001*(mid_dy-midy), width=0.01, color=color['inner'])
    
    if bidirectional:
        arrow = mpatches.FancyArrowPatch(posa, posb,
                                connectionstyle=f"arc3,rad={-(-int(inner))*arc}", color=color['outer'], **kw)
        ax.add_patch(arrow)
        mid_dx, mid_dy, midx, midy = _trace_bezier_midpoint(arrow)
        mid_dx, mid_dy = _push_wedge(pushfactor['outer'], mid_dx, mid_dy, centerxy)
        midx, midy = _push_wedge(pushfactor['outer'], midx, midy, centerxy)
#         ax.arrow(midx, midy, 0.001*(midx-mid_dx), 0.001*(midy-mid_dy), width=0.01, color=color['outer'])
        

def _draw_selfloop_arrow(ax, centX, centY, radius, angle_, theta2_, color_):
    from numpy import radians as rad
    #========Line
    arc = mpatches.Arc([centX,centY],radius,radius,angle=angle_,
        theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=7, color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))
    
    head = mpatches.RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius=radius/8,                # radius
            orientation=rad(angle_+theta2_),     # orientation
            color=color_)
    

    ax.add_patch(head)
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort

@staticmethod
def railroad_plot(lkm, rad=0.3, centerxy=(0.5, 0.5), nverts=3, region=['P','F', 'T', 'O', 'S'], 
                    colors=['tab:blue', 'tab:green', 'tab:red', 'purple', 'pink'],
                    pushfactor={'inner':0.715, 'outer':1.147}, figsize=(10, 10), cmap=mpl.colormaps['binary'],
                    ax_=None):
    # create a railroad plot from a link matrix. 
    # the link matrix should have normalized values, i.e., each cell should be a percentage in [0, 1].
    
    # links from behrad go row:dst col:src but we need row:src col:dst.
    lkm = lkm.T
    
    if ax_ is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = ax_

    xs, ys, _, _ = _get_vertices(nverts, rad, 0, centerxy)
    xs_selfloop, ys_selfloop, _, thetas_selfloop = _get_vertices(nverts, rad+0.1, 0, centerxy)
    
    # create links in order of darkness (lightest first)
    is_, js_ = np.unravel_index(np.argsort(lkm, axis=None), (nverts,nverts))
    
    for i, j in zip(is_, js_):
        a = lkm[i, j]
        b = lkm[j, i]
        assert(a <= 1 and a >= 0 and b <= 1 and b >= 0)
        if a == 1:
            a = a - 0.00001
        if b == 1:
            b = b - 0.00001


        color_i = cmap(a)
        color_o = cmap(b)
        _add_connection(ax, (xs[i], ys[i]), (xs[j], ys[j]), 
                    bidirectional=True, arc=0.25, pushfactor=pushfactor, 
                    color={'inner':color_i, 'outer':color_o})


    for i in range(nverts):
        c = lkm[i, i]
        assert(c <= 1 and c >= 0)
        if c == 1:
            c = c - 0.00001
        color_self = cmap(c)
        _draw_selfloop_arrow(ax, xs_selfloop[i], ys_selfloop[i], 0.15, -180+thetas_selfloop[i], 315, 
            color_=color_self)
    
    for idx, (i, j) in enumerate(zip(xs, ys)):
        circle = mpatches.Circle((i, j), 0.05, linewidth=3, alpha=1, color=colors[idx])
        ax.add_patch(circle)
        
        if ax_ is None:
            plt.text(i, j, region[idx], size=35,
                ha="center", va='center_baseline', color='white', fontname='sans-serif', weight='heavy')
        else:
            ax.text(i, j, region[idx], size=35,
                ha="center", va='center_baseline', color='white', fontname='sans-serif', weight='heavy')

    if ax_ is None:
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
    return ax

@staticmethod
def sankey(lkms, labels, colorlist=None):
    dfs = []
    colors = []
    for idx, lkm in enumerate(lkms):
        lkm = lkm.T
        df = pd.DataFrame(lkm)
        df.index = labels
        df.columns = labels
        dfs.append(df)
        n = df.values.max()
        vs = (df.values.flatten()/n)**2
        if colorlist is None:
            colors.extend([f"rgba(0.0, 0.5, 0.7, {i:.20f})" for i in vs])
        else:
            assert(colorlist[idx][-1:] == ')')
            colors.extend([colorlist[idx][:-1] + f", {i:.20f})" for i in vs])

    lno = len(labels)
    srcblocks = np.repeat(range(lno*len(dfs)), lno)
    dstblocks = []
    for i in range(len(dfs)):
        dstblocks.extend(list(range(lno*(i+1), lno*(i+2)))*lno)

    vals = []
    for df in dfs:
        vals.extend(list(df.values.flatten()))


    ncols = len(dfs)+1
    yp = list(np.linspace(0.1, 0.9, len(labels)))*ncols
    xp = list(np.repeat(np.linspace(0.1, 0.9, ncols), lno))


    fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "red", width = 0.5),
        label = labels*ncols,
        color = ["darkblue"]*ncols*lno,
        y=yp,
        x=xp
        ),
        link = dict(
        source = srcblocks, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = dstblocks,
        value = vals,
        color=colors
    ))])

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show(renderer='browser')


def colorline(
    ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=1, alpha=0.025, color=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    if color is None:
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=1)
    else:
        lc = mcoll.LineCollection(segments, array=z, colors=color, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def jitter_points(df_edges, df_coords):
    targets = df_edges.target.tolist() 
    sources = df_edges.source.tolist() 

    jittered_coords = []
    jittered_edges = []

    c1 = 0
    c2 = 1
    for tprefix, sprefix in zip(targets, sources):
        factor = 125
        txjitter ,= np.clip(np.random.normal(0.5, 0.25, 1), 0.01, 0.99)/factor
        tyjitter ,= np.clip(np.random.normal(0.5, 0.25, 1), 0.01, 0.99)/factor
        sxjitter ,= np.clip(np.random.normal(0.5, 0.25, 1), 0.01, 0.99)/factor
        syjitter ,= np.clip(np.random.normal(0.5, 0.25, 1), 0.01, 0.99)/factor

        jittered_edges.append([c1, c2])

        c1 += 2
        c2 += 2

        tx_center ,= df_coords[df_coords.name == int(tprefix)].x.values
        ty_center ,= df_coords[df_coords.name == int(tprefix)].y.values
        sx_center ,= df_coords[df_coords.name == int(sprefix)].x.values
        sy_center ,= df_coords[df_coords.name == int(sprefix)].y.values

        jittered_coords.append([c1, tx_center+(txjitter-(0.5/factor)), ty_center+(tyjitter-(0.5/factor))])
        jittered_coords.append([c2, sx_center+(sxjitter-(0.5/factor)), sy_center+(syjitter-(0.5/factor))])

    df_coords = pd.DataFrame(jittered_coords)
    df_edges = pd.DataFrame(jittered_edges)

    df_coords.columns = ["name", "x", "y"]
    df_edges.columns = ["target", "source"]
    return df_edges, df_coords


def _hammer_plot(self, adjb, x, y, ax, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), 
                linecolors='uniform', split_source_sink=False, 
               jitter=False, adjb2=None, subject='fsaverage', 
               initial_bandwidth=0.3, decay=0.7):
    if subject == 'fsaverage':
        self.experiment.e.set(subject=subject, match=False)
    else:
        self.experiment.e.set(subject=subject)
    ss = self.experiment.e.load_src(src='ico-1')
    ax.plot(ss[0]['rr'][:, x], ss[0]['rr'][:, y], color='black')
    ax.plot(ss[1]['rr'][:, x], ss[1]['rr'][:, y], color='black')
    s = eel.SourceSpace.from_mne_source_spaces(ss, 'ico-1', self.subjects_dir, 'aparc')

    xyproj = s.coordinates[:, [x, y]]
    df_coords = pd.DataFrame(xyproj).reset_index()
    
    if adjb2 is not None:
        # combine two link matrices to build a common graph
        # now df_edges will contain edges from both link matrices
        # therefore the total derived graph will look the same 
        # for both. we can then color one or the other later
        df_edges = pd.DataFrame(np.where(adjb + adjb2)).T
    else:
        df_edges = pd.DataFrame(np.where(adjb)).T

    df_coords.columns = ["name", "x", "y"]
    df_edges.columns = ["target", "source"]
    df_edges_old = df_edges
    
    if jitter:
        df_edges, df_coords = jitter_points(df_edges, df_coords)
    
    
    if split_source_sink:
        assert(jitter == False)
        df_coords2 = df_coords.copy()
        df_coords2.name += 84
        df_coords2.x += 0.005
        df_coords = pd.concat([df_coords, df_coords2]).reset_index(drop=True)
        df_edges.target += 84

    vmax = np.abs(adjb).max()
    
    df_e = pd.DataFrame(np.where(adjb)).T
    df_e.columns = ["target", "source"]

    hb = hammer_bundle(df_coords, df_edges, initial_bandwidth=initial_bandwidth, decay=decay)
    
    cutoffs = [0]+hb[np.isnan(hb['x'])].index.values.tolist()
    

    for dst, src in zip(df_e.target.values, df_e.source.values):
        sidx = df_edges[(df_edges_old.target == dst) & (df_edges_old.source == src)].index.values[0]
        if sidx == len(cutoffs) - 1:
            break
        eidx = sidx+1
        start = cutoffs[sidx]
        end = cutoffs[eidx]
        
        if linecolors == 'uniform':
            ax.plot(hb['x'].values[start:end], hb['y'].values[start:end], color='orange')
        elif linecolors == 'binary':
            ax.plot(hb['x'].values[start:end-((end-start)//2)], hb['y'].values[start:end-((end-start)//2)], color='orange')
            ax.plot(hb['x'].values[start+((end-start)//2):end], hb['y'].values[start+((end-start)//2):end], color='blue')
        elif linecolors == 'gradient':
            colorline(ax, hb['x'].values[start:end], hb['y'].values[start:end], cmap='PiYG_r')
        elif linecolors == 'translucent':
            colorline(ax, hb['x'].values[start:end], hb['y'].values[start:end], color='tab:orange', alpha=0.2)
        elif linecolors == 'translucentweighted':
            curr_link_val = (adjb[dst, src]/vmax)/5
            colorline(ax, hb['x'].values[start:end], hb['y'].values[start:end], color='tab:orange', alpha=curr_link_val)
        elif linecolors == 'difference':
            curr_link_val = (adjb[dst, src]/vmax)/2
            if curr_link_val < 0:
                colorline(ax, hb['x'].values[start:end], hb['y'].values[start:end], color='tab:blue', alpha=abs(curr_link_val))
            else:
                colorline(ax, hb['x'].values[start:end], hb['y'].values[start:end], color='tab:red', alpha=curr_link_val)
        elif linecolors == 'difference_halfline':
            curr_link_val = (adjb[dst, src]/vmax)/2
            if curr_link_val < 0:
                colorline(ax, hb['x'].values[start:end-((end-start)//2)], hb['y'].values[start:end-((end-start)//2)], color='tab:blue', alpha=abs(curr_link_val))
            else:
                colorline(ax, hb['x'].values[start:end-((end-start)//2)], hb['y'].values[start:end-((end-start)//2)], color='tab:red', alpha=curr_link_val)
        else:
            raise Exception(f"linecolors {linecolors} not supported")
            
            
    if jitter == False:
        ax.plot(df_coords['x'][:84], df_coords['y'][:84], '.', color='red')
        ax.plot(df_coords['x'][84:], df_coords['y'][84:], '.', color='blue')

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    return df_edges, df_coords, hb


def hammer_plot_matrix(self, mat1, mat2=None, diff=False, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(30, 10), layout="compressed")

    views = [(0, 1), (1, 2), (0, 2)]
    xlims = [(-0.1, 0.1), (-0.1, 0.07), (-0.1, 0.1)]
    ylims = [(-0.115, 0.085), (-0.075, 0.1), (-0.075, 0.1)]

    if mat2 is not None:
        adjb2 = (mat1+mat2)/2
    else:
        adjb2 = None

    if diff:
        if mat2 is None:
            mat3 = mat1
        else:
            mat3 = mat1 - mat2
    else:
        mat3 = mat1
    

    for i in range(3):
        x, y = views[i]
        
        df_edges, df_coords, hb = self._hammer_plot(mat3, x, y, ax[i], xlims[i], ylims[i], 
                                              adjb2=adjb2, **kwargs)
        ax[i].axis('off')
        
    fig.subplots_adjust(wspace=0.0)
    plt.show()


def hammer_plot(self, condition1, condition2=None, diff=False, **kwargs):
    mat1 = np.mean(self.condition_to_models(condition1, norm=True), axis=0)
    if condition2 is None:
        mat2 = None
    else:
        mat2 = np.mean(self.condition_to_models(condition2, norm=True), axis=0)

    self.hammer_plot_matrix(mat1, mat2, diff, **kwargs)

    

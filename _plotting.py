import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
from mne_connectivity import viz


################################################################################
# Heatmap
################################################################################

def heatmap_linkmatrix(self, lkm1, lkm2, status1='C1', status2='C2', hemi=False, 
                     overlay_nums=False, vmin=0, vmax=1, diffvmin=-1, diffvmax=1, figsize=(10, 3)):
    if hemi == False:
        c1_group_avg = lkm1.T
        c2_group_avg = lkm2.T

        fig, ax = plt.subplots(1, 3, figsize=figsize)

        im = ax[0].imshow(c1_group_avg, vmin=vmin, vmax=vmax)
        ax[0].set_title(f"{status1}")
        ax[0].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
        ax[0].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
        if overlay_nums == True:
            for srcidx, source in enumerate(self.target_lobes):
                for dstidx, target in enumerate(self.target_lobes):
                    ax[0].text(srcidx, dstidx, 
                        round(c1_group_avg[dstidx, srcidx], 2), ha="center", va="center", color="w")


        ax[1].imshow(c2_group_avg, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"{status2}")
        ax[1].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
        ax[1].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
        if overlay_nums == True:
            for srcidx, source in enumerate(self.target_lobes):
                for dstidx, target in enumerate(self.target_lobes):
                    ax[1].text(srcidx, dstidx, 
                        round(c2_group_avg[dstidx, srcidx], 2), ha="center", va="center", color="w")


        im2 = ax[2].imshow(c1_group_avg - c2_group_avg, vmin=diffvmin, vmax=diffvmax, cmap='seismic')
        ax[2].set_title(f"{status1}-{status2}")
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
        for i in range(4):
            c1_group_avg_hemi[i] = lkm1[i].T
            c2_group_avg_hemi[i] = lkm2[i].T
        
        fig, ax = plt.subplots(2, 6, figsize=figsize)

        for srchemi, dsthemi, i, j in [('lh', 'lh', 0, 0), ('rh', 'rh', 1, 1), 
                                        ('lh', 'rh', 0, 1), ('rh', 'lh', 1, 0)]:
            hemi_idx = self._get_hemi_idx(srchemi, dsthemi)
            im = ax[i, j].imshow(c1_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)
            ax[i, j].set_title(f"{status1}:{srchemi}->{dsthemi}")
            ax[i, j].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[i, j].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[i, j].text(srcidx, dstidx, 
                            round(c1_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")

            ax[i, j+2].imshow(c2_group_avg_hemi[hemi_idx], vmin=vmin, vmax=vmax)
            ax[i, j+2].set_title(f"{status2}:{srchemi}->{dsthemi}")
            ax[i, j+2].set_xticks(list(range(len(self.target_lobes))), self.target_lobes)
            ax[i, j+2].set_yticks(list(range(len(self.target_lobes))), self.target_lobes)
            if overlay_nums == True:
                for srcidx, source in enumerate(self.target_lobes):
                    for dstidx, target in enumerate(self.target_lobes):
                        ax[i, j+2].text(srcidx, dstidx, 
                            round(c2_group_avg_hemi[hemi_idx, dstidx, srcidx],2), ha="center", va="center", color="w")

            im2 = ax[i, j+4].imshow(c1_group_avg_hemi[hemi_idx] \
                                - c2_group_avg_hemi[hemi_idx],
                                vmin=diffvmin, vmax=diffvmax, cmap='seismic')
            ax[i, j+4].set_title(f"{status1}-{status2}:{srchemi}->{dsthemi}")
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
    

def heatmap(self, condition1, condition2, status1='C1', status2='C2', hemi=False, 
                     overlay_nums=False, vmin=0, vmax=1, diffvmin=-1, diffvmax=1, figsize=(10, 3)):
    if hemi == True:
        condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
        
    if self.conn is None:
        self.tabulate_links(verbose=False)


    ###################################
    if hemi == False:
        c1_group_avg, c2_group_avg = self.group_averages_whole(condition1, condition2)

        self.heatmap_linkmatrix(c1_group_avg, c2_group_avg, status1, status2, hemi, 
                     overlay_nums, vmin, vmax, diffvmin, diffvmax, figsize)
    else:
        c1_group_avg_hemi, c2_group_avg_hemi = self.group_averages_hemi(condition1, condition2)
        self.heatmap_linkmatrix(c1_group_avg_hemi, c2_group_avg_hemi, status1, status2, hemi, 
                     overlay_nums, vmin, vmax, diffvmin, diffvmax, figsize)


################################################################################
# Circle Plot
################################################################################

def circle_plot(self, condition1, condition2, status1='C1', status2='C2', hemi=False):
    if hemi == True:
        condition1, condition2 = self._check_hemis(condition1), self._check_hemis(condition2)
        
    if self.conn is None:
        self.tabulate_links(verbose=False)


    ###################################
    if hemi == False:
        c1_group_avg, c2_group_avg = self.group_averages_whole(condition1, condition2)
        
        
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
        c1_group_avg_hemi, c2_group_avg_hemi = self.group_averages_hemi(condition1, condition2)
            
        
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

    ax.arrow(midx, midy, 0.001*(mid_dx-midx), 0.001*(mid_dy-midy), width=0.01, color=color['inner'])
    
    if bidirectional:
        arrow = mpatches.FancyArrowPatch(posa, posb,
                                connectionstyle=f"arc3,rad={-(-int(inner))*arc}", color=color['outer'], **kw)
        ax.add_patch(arrow)
        mid_dx, mid_dy, midx, midy = _trace_bezier_midpoint(arrow)
        mid_dx, mid_dy = _push_wedge(pushfactor['outer'], mid_dx, mid_dy, centerxy)
        midx, midy = _push_wedge(pushfactor['outer'], midx, midy, centerxy)
        ax.arrow(midx, midy, 0.001*(midx-mid_dx), 0.001*(midy-mid_dy), width=0.01, color=color['outer'])
        

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

        for i in range(nverts):
            a = lkm[i, (i+1)%nverts]
            b = lkm[(i+1)%nverts, i]
            assert(a <= 1 and a >= 0 and b <= 1 and b >= 0)
            if a == 1:
                a = a - 0.00001
            if b == 1:
                b = b - 0.00001

            color_i = cmap(a)
            color_o = cmap(b)
            _add_connection(ax, (xs[i], ys[i]), (xs[(i+1)%nverts], ys[(i+1)%nverts]), 
                        bidirectional=True, arc=0.25, pushfactor={'inner':0.715, 'outer':1.147}, 
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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from dipy.workflows.base import IntrospectiveArgumentParser

import argparse
import numpy as np
import nibabel as nib
from dipy.viz import actor, window, widget
from dipy.viz import fvtk
from dipy.tracking.streamline import set_number_of_points


def build_argparser():
    description = "Visualize streamlines colored according to their loss."
    p = argparse.ArgumentParser(description=description)

    p.add_argument("tractogram", help="Tractogram file (.TRK)")

    return p



def create_hist(outlierness, colormap_name="jet"):
    import pylab as plt
    # Make histogram plot prettier.
    plt.rcParams['font.size'] = 24
    #plt.rcParams['font.weight'] = "bold"
    fig = plt.figure(figsize=(16, 4), dpi=300)
    ax = fig.add_subplot(111)
    #ax.set_title('Outlierness', fontsize="32", fontweight="bold")
    #ax.set_ylabel('Outlierness', fontsize="24")
    #n, bins, patches = ax.hist(outlierness, bins=np.linspace(0, 1, 101), linewidth=0.2, orientation="horizontal")
    n, bins, patches = ax.hist(outlierness, bins=np.linspace(0, 1, 101), linewidth=0)

    # Apply colormap to histogram.
    cm = plt.cm.get_cmap(colormap_name)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return fig


def create_hist_actor(outlierness, colormap_name="jet"):
    import vtk
    from dipy.viz import actor, utils

    import pylab as plt
    if not hasattr(plt, 'style'):
        print "Use Matplotlib >= 1.4 to have better colormap."
    else:
        plt.style.use('dark_background')

    fig = create_hist(outlierness, colormap_name)
    fig.gca().set_title("")
    arr = utils.matplotlib_figure_to_numpy(fig, dpi=300, transparent=False)
    figure_actor = actor.figure(arr, size=100, interpolation='linear')

    # Make sure the figure is center is in the middle of the image.
    transform = vtk.vtkTransform()
    transform.Translate(-np.array(figure_actor.GetCenter()))
    figure_actor.SetUserMatrix(transform.GetMatrix())
    return figure_actor, fig


def interactive_viewer(streamlines, outlierness):
    import vtk
    from dipy.viz import fvtk, actor, window, widget
    from dipy.data.fetcher import read_viz_icons

    colormap_name = "jet"
    stream_actor = actor.line(streamlines, colors=fvtk.create_colormap(outlierness, name=colormap_name))
    stream_actor.SetPosition(-np.array(stream_actor.GetCenter()))

    global threshold
    threshold = 0.8

    streamlines_color = np.zeros(len(streamlines), dtype="float32")
    streamlines_color[outlierness < threshold] = 1
    streamlines_color[outlierness >= threshold] = 0

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.Build()
    lut.SetTableValue(0, tuple(fvtk.colors.orange_red) + (1,))
    lut.SetTableValue(1, tuple(fvtk.colors.green) + (1,))
    lut.SetTableRange(0, 1)

    stream_split_actor = actor.line(streamlines, colors=streamlines_color, lookup_colormap=lut)
    stream_split_actor.SetPosition(-np.array(stream_split_actor.GetCenter()))
    hist_actor, hist_fig = create_hist_actor(outlierness, colormap_name=colormap_name)

    # Main renderder
    bg = (0, 0, 0)
    global screen_size
    screen_size = (0, 0)
    ren_main = window.Renderer()
    ren_main.background(bg)
    show_m = window.ShowManager(ren_main, size=(1066, 600), interactor_style="trackball")
    show_m.window.SetNumberOfLayers(2)
    ren_main.SetLayer(1)
    ren_main.InteractiveOff()

    # Outlierness renderer
    ren_outlierness = window.Renderer()
    show_m.window.AddRenderer(ren_outlierness)
    ren_outlierness.background(bg)
    ren_outlierness.SetViewport(0, 0.3, 0.5, 1)
    ren_outlierness.add(stream_actor)
    ren_outlierness.reset_camera_tight()

    ren_split = window.Renderer()
    show_m.window.AddRenderer(ren_split)
    ren_split.background(bg)
    ren_split.SetViewport(0.5, 0.3, 1, 1)
    ren_split.add(stream_split_actor)
    ren_split.SetActiveCamera(ren_outlierness.GetActiveCamera())

    # Histogram renderer
    ren_hist = window.Renderer()
    show_m.window.AddRenderer(ren_hist)
    ren_hist.projection("parallel")
    ren_hist.background(bg)
    ren_hist.SetViewport(0, 0, 1, 0.3)
    ren_hist.add(hist_actor)
    ren_hist.SetInteractive(False)

    def apply_threshold(obj, evt):
        global threshold
        new_threshold = np.round(obj.GetSliderRepresentation().GetValue(), decimals=2)
        obj.GetSliderRepresentation().SetValue(new_threshold)
        if threshold != new_threshold:
            threshold = new_threshold

            streamlines_color = np.zeros(len(streamlines), dtype=np.float32)
            streamlines_color[outlierness < threshold] = 1
            streamlines_color[outlierness >= threshold] = 0

            colors = []
            for color, streamline in zip(streamlines_color, streamlines):
                colors += [color] * len(streamline)

            scalars = stream_split_actor.GetMapper().GetInput().GetPointData().GetScalars()
            for i, c in enumerate(colors):
                scalars.SetValue(i, c)

            scalars.Modified()

    threshold_slider_rep = vtk.vtkSliderRepresentation3D()
    threshold_slider_rep.SetMinimumValue(0.)
    threshold_slider_rep.SetMaximumValue(1.)
    threshold_slider_rep.SetValue(threshold)
    threshold_slider_rep.SetLabelFormat("%0.2lf")
    threshold_slider_rep.SetLabelHeight(0.02)
    threshold_slider_rep.GetPoint1Coordinate().SetCoordinateSystemToWorld()
    x1, x2, y1, y2, z1, z2 = hist_actor.GetBounds()
    threshold_slider_rep.GetPoint1Coordinate().SetValue(x1*1., y1-5, 0)
    threshold_slider_rep.GetPoint2Coordinate().SetCoordinateSystemToWorld()
    threshold_slider_rep.GetPoint2Coordinate().SetValue(x2*1., y1-5, 0)
    threshold_slider_rep.SetEndCapLength(0.)
    threshold_slider_rep.SetEndCapWidth(0.)

    threshold_slider = vtk.vtkSliderWidget()
    threshold_slider.SetInteractor(show_m.iren)
    threshold_slider.SetRepresentation(threshold_slider_rep)
    threshold_slider.SetCurrentRenderer(ren_hist)
    threshold_slider.SetAnimationModeToJump()
    threshold_slider.EnabledOn()

    threshold_slider.AddObserver("InteractionEvent", apply_threshold)

    #ren_main
    def _place_buttons():
        sz = 30.0
        width, _ = ren_main.GetSize()

        # bds = np.zeros(6)
        # bds[0] = width - sz - 5
        # bds[1] = bds[0] + sz
        # bds[2] = 5
        # bds[3] = bds[2] + sz
        # bds[4] = bds[5] = 0.0
        # save_button.GetRepresentation().PlaceWidget(bds)

    def _window_callback(obj, event):
        ren_hist.reset_camera_tight(margin_factor=1.2)
        _place_buttons()

    show_m.add_window_callback(_window_callback)
    show_m.initialize()
    show_m.render()
    show_m.start()

    inliers = [s for s, keep in zip(streamlines, outlierness < threshold) if keep]
    outliers = [s for s, keep in zip(streamlines, outlierness >= threshold) if keep]
    return inliers, outliers


def main():
    parser = build_argparser()
    args = parser.parse_args()

    tractogram = nib.streamlines.load(args.tractogram).tractogram

    idx = np.arange(len(tractogram))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    tractogram = tractogram[idx[:300]]

    losses = tractogram.data_per_streamline['loss'][:, 0]
    losses = np.exp(-losses)

    interactive_viewer(tractogram.streamlines, losses)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import nibabel as nib
from dipy.viz import actor, window, widget
from dipy.viz import fvtk
from dipy.tracking.streamline import set_number_of_points

from learn2track.neurotools import TractographyData


def horizon(tractograms, data, affine):

    rng = np.random.RandomState(42)
    slicer_opacity = .8

    ren = window.Renderer()
    global centroid_actors
    centroid_actors = []
    for streamlines in tractograms:

        print(' Number of streamlines loaded {} \n'.format(len(streamlines)))
        colors = rng.rand(3)
        ren.add(actor.line(streamlines, colors, opacity=1., lod_points=10 ** 5))

    class SimpleTrackBallNoBB(window.vtk.vtkInteractorStyleTrackballCamera):
        def HighlightProp(self, p):
            pass

    style = SimpleTrackBallNoBB()
    # very hackish way
    style.SetPickColor(0, 0, 0)
    # style.HighlightProp(None)
    show_m = window.ShowManager(ren, size=(1200, 900), interactor_style=style)
    show_m.initialize()

    if data is not None:
        #from dipy.core.geometry import rodrigues_axis_rotation
        #affine[:3, :3] = np.dot(affine[:3, :3], rodrigues_axis_rotation((0, 0, 1), 45))

        image_actor = actor.slicer(data, affine)
        image_actor.opacity(slicer_opacity)
        image_actor.SetInterpolate(False)
        ren.add(image_actor)

        ren.add(fvtk.axes((10, 10, 10)))

        last_value = [10]
        def change_slice(obj, event):
            new_value = int(np.round(obj.get_value()))
            if new_value == image_actor.shape[1] - 1 or new_value == 0:
                new_value = last_value[0] + np.sign(new_value - last_value[0])

            image_actor.display(None, new_value, None)
            obj.set_value(new_value)
            last_value[0] = new_value

        slider = widget.slider(show_m.iren, show_m.ren,
                               callback=change_slice,
                               min_value=0,
                               max_value=image_actor.shape[1] - 1,
                               value=image_actor.shape[1] / 2,
                               label="Move slice",
                               right_normalized_pos=(.98, 0.6),
                               size=(120, 0), label_format="%0.lf",
                               color=(1., 1., 1.),
                               selected_color=(0.86, 0.33, 1.))

        slider.SetAnimationModeToJump()

    global size
    size = ren.GetSize()
    # ren.background((1, 0.5, 0))
    ren.background((0, 0, 0))
    global picked_actors
    picked_actors = {}

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            if data is not None:
                slider.place(ren)
            size = obj.GetSize()

    global centroid_visibility
    centroid_visibility = True

    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()


def add_noise_to_streamlines(streamlines, sigma, rng=np.random.RandomState(42)):
    # Add gaussian noise, N(0, self.sigma).
    noisy_streamlines = streamlines.copy()
    shape = noisy_streamlines._data.shape
    noisy_streamlines._data += sigma * rng.randn(*shape)
    return noisy_streamlines


def horizon_flow(input_files, noisy_streamlines_sigma=0., verbose=True):
    """ Horizon

    Parameters
    ----------
    input_files : variable string
    cluster : bool, optional
    cluster_thr : float, optional
    random_colors : bool, optional
    verbose : bool, optional
    length_lt : float, optional
    length_gt : float, optional
    clusters_lt : int, optional
    clusters_gt : int, optional
    noisy_streamlines_sigma : float, optional
    """

    filenames = input_files
    # glob(input_files)
    tractograms = []

    data = None
    affine = None
    for i, f in enumerate(filenames):
        if verbose:
            print('Loading file ...')
            print(f)
            print('\n')

        if f.endswith('.trk') or f.endswith('.tck'):
            streamlines = nib.streamlines.load(f).streamlines
            idx = np.arange(len(streamlines))
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            streamlines = streamlines[idx[:100]]

            if noisy_streamlines_sigma > 0. and i > 0:
                streamlines = add_noise_to_streamlines(streamlines, noisy_streamlines_sigma)

            tractograms.append(streamlines)

        if f.endswith('.npz'):
            tractography_data = TractographyData.load(f)
            # idx = np.arange(len(tractography_data.streamlines))
            # rng = np.random.RandomState(42)
            # rng.shuffle(idx)
            # tractography_data.streamlines = tractography_data.streamlines[idx[:200]]
            # tractograms.append(tractography_data.streamlines)

            M = 2
            # Take M streamlines per bundle
            for k in sorted(tractography_data.name2id.keys()):
                bundle_id = tractography_data.name2id[k]
                streamlines = tractography_data.streamlines[tractography_data.bundle_ids == bundle_id][:M].copy()
                streamlines._lengths = streamlines._lengths.astype("int64")
                streamlines = set_number_of_points(streamlines, nb_points=40)
                tractograms.append(streamlines)

            if hasattr(tractography_data, 'signal'):
                signal = tractography_data.signal.get_data()
                data = signal[:, :, :, 0]
                affine = np.eye(4)

        if f.endswith('.nii.gz') or f.endswith('.nii'):

            img = nib.load(f)
            data = img.get_data()
            affine = img.get_affine()
            if verbose:
                print(affine)

    # tmp save
    # tractogram = nib.streamlines.Tractogram(tractograms[0])
    # tractogram.apply_affine(img.affine)
    # nib.streamlines.save(tractogram, "tmp.tck")
    # exit()

    horizon(tractograms, data, affine)


if __name__ == '__main__':
    horizon_flow(input_files=sys.argv[1:])

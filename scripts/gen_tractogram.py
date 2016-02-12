#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
from os.path import join as pjoin

import theano
import theano.tensor as T

import nibabel as nib
from dipy.tracking.streamline import compress_streamlines
from dipy.tracking.streamline import values_from_volume

from smartlearner import Dataset
from smartlearner import utils as smartutils

from learn2track.rnn import RNN
from learn2track.utils import Timer, map_coordinates_3d_4d, normalize_dwi
from smartlearner.utils import load_dict_from_json_file


def build_argparser():
    DESCRIPTION = "Generate a tractogram from a LSTM model trained on ismrm2015 challenge data."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")

    p.add_argument('--wm-mask', type=str, help="streamlines will stop if going outside this mask (.nii|.nii.gz).")
    p.add_argument('--bvals', type=str, help='text file with the bvalues. Default: same name as the dwi file but with extension .bvals.')
    p.add_argument('--seeding-mask', type=str, help="streamlines will start from this mask (.nii|.nii.gz).")
    p.add_argument('--trk', type=str, help="perform some qualitative evaluation using a testset bundle (.trk).")

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite existing tractogram')

    return p

floatX = theano.config.floatX
NB_POINTS = 100


def track(model, dwi, seeds, step_size=0.5, max_nb_points=500, mask=None, mask_threshold=0.05, affine=None):
    inputs = T.tensor3('inputs')
    inputs.tag.test_value = map_coordinates_3d_4d(dwi, np.asarray(seeds)[:, None, :])
    next_directions = model.use(inputs)
    track_step = theano.function([inputs], next_directions[:, -1])

    streamlines = []
    sequences = np.asarray(seeds)[:, None, :]
    # streamlines_dwi = map_coordinates_3d_4d(dwi, sequences[:, [-1]])
    streamlines_dwi = values_from_volume(dwi, sequences[:, [-1]], affine)[0].astype(np.float32)

    for i in range(max_nb_points):
        if (i+1) % 10 == 0:
            print("{}/{}".format(i+1, max_nb_points))

        directions = track_step(streamlines_dwi)
        directions *= step_size

        done = []
        undone = np.arange(len(directions))

        # If a streamline goes outside the wm mask, mark is as done.
        if mask is not None:
            next_points = sequences[undone, -1] + directions[undone]

            # # next_points_voxel = np.round(next_points)
            # next_points_voxel = np.floor(next_points + 0.5)
            # for idx, voxel in list(zip(undone, next_points_voxel)):
            #     if tuple(voxel) not in allowed_voxels:
            #         done += [idx]
            #         undone.remove(idx)

            values = values_from_volume(mask, next_points.reshape((-1, 1, 3)), affine).flatten()
            # if values.ndim == 0:  # If there is only a single points, values is scalar instead of a vector.
            #     values = values.reshape((1,))

            done.extend(undone[values < mask_threshold])
            undone = undone[values >= mask_threshold]
            assert len(undone) + len(done) == len(directions)

        streamlines.extend([s for s in sequences[done]])

        if len(undone) == 0:
            break

        sequences = sequences[undone]
        points = sequences[:, [-1]] + directions[undone, None, :]
        sequences = np.concatenate([sequences, points], axis=1)

        streamlines_dwi = np.concatenate([streamlines_dwi[undone],
                                          values_from_volume(dwi, sequences[:, [-1]], affine)[0].astype(np.float32)
                                          # map_coordinates_3d_4d(dwi, sequences[:, [-1]])],
                                          ], axis=1)

    # Add remaining
    streamlines.extend([s for s in sequences])
    return streamlines


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    # Load experiments hyperparameters
    try:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
    except FileNotFoundError:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "..", "hyperparams.json"))

    with Timer("Loading model"):
        if hyperparams["model"] == "lstm":
            from learn2track.lstm import LSTM_Regression
            model_class = LSTM_Regression
        elif hyperparams["model"] == "lstm_extraction":
            from learn2track.lstm import LSTM_RegressionWithFeaturesExtraction
            model_class = LSTM_RegressionWithFeaturesExtraction

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    with Timer("Loading DWIs"):
        dwi = nib.load(args.dwi)

        # Load and parse bvals
        bvals_filename = args.bvals
        if bvals_filename is None:
            bvals_filename = args.dwi.split('.')[0] + ".bvals"

        bvals = list(map(float, open(bvals_filename).read().split()))
        bvals = np.round(bvals).astype(int)

        dwi = nib.load(args.dwi)
        weights = normalize_dwi(dwi, bvals)

    allowed_voxels = None
    if args.wm_mask is not None:
        with Timer("Loading WM mask"):
            wm_mask = nib.load(args.wm_mask).get_data()
            allowed_voxels = set(zip(*np.where(wm_mask)))

    with Timer("Generating seeds"):
        step_size = 0.5
        if args.trk is not None:
            trk = nib.streamlines.load(args.trk)
            # trk.tractogram.apply_affine(np.linalg.inv(trk.affine))  # We track in voxel space.
            # # step_size = np.mean([np.mean(np.sqrt(np.sum((pts[1:]-pts[:-1])**2, axis=1))) for pts in streamlines.points])
            # # seeds = [s[0] for s in streamlines.points[::10]]
            # assert np.all(np.min(trk.tractogram.streamlines._data, axis=0) >= 0)
            # max_coords = np.max(trk.tractogram.streamlines._data, axis=0)
            # assert max_coords[0] < dwi.shape[0]
            # assert max_coords[1] < dwi.shape[1]
            # assert max_coords[2] < dwi.shape[2]
            seeds = [s[0] for s in trk.streamlines]
            seeds += [s[-1] for s in trk.streamlines]

    with Timer("Tracking"):
        new_streamlines = track(model, weights, seeds, step_size, mask=wm_mask, affine=dwi.affine)

    with Timer("Saving streamlines"):
        new_streamlines = compress_streamlines(new_streamlines)
        tractogram = nib.streamlines.Tractogram(new_streamlines)
        # tractogram.apply_affine(trk.affine)  # Streamlines were generated in voxel space.
        # new_trk = nib.streamlines.TrkFile(tractogram, trk.header)
        # save_path = pjoin(args.name, "generated_streamlines.trk")
        #nib.streamlines.save(new_trk, save_path)
        save_path = pjoin(args.name, "generated_streamlines.tck")
        nib.streamlines.save(tractogram, save_path)

if __name__ == "__main__":
    main()

# def track(model, dwi, seeds, step_size=0.5, allowed_voxels=None, max_nb_points=500):
#     inputs = T.tensor3('inputs')
#     inputs.tag.test_value = map_coordinates_3d_4d(dwi, np.asarray(seeds)[:, None, :])
#     next_directions, stoppings = model.use(inputs)
#     track_step = theano.function([inputs], [next_directions[:, -1], stoppings[:, -1, 0]])

#     streamlines = []
#     sequences = np.asarray(seeds)[:, None, :]
#     streamlines_dwi = map_coordinates_3d_4d(dwi, sequences[:, [-1]])

#     for i in range(max_nb_points):
#         if (i+1) % 10 == 0:
#             print("{}/{}".format(i+1, max_nb_points))

#         directions, prob_stoppings = track_step(streamlines_dwi)
#         directions *= step_size

#         done = list(np.where(prob_stoppings > 0.5)[0])
#         undone = list(np.where(prob_stoppings <= 0.5)[0])

#         # If a streamline goes outside the wm mask, mark is as done.
#         if allowed_voxels is not None:
#             next_points = sequences[undone, -1] + directions[undone]
#             next_points_voxel = np.round(next_points)
#             for idx, voxel in zip(undone, next_points_voxel):
#                 if tuple(voxel) not in allowed_voxels:
#                     done += [idx]
#                     undone.remove(idx)

#         streamlines.extend([s for s in sequences[done]])

#         if len(undone) == 0:
#             break

#         sequences = sequences[undone]
#         points = sequences[:, [-1]] + directions[undone, None, :]
#         sequences = np.concatenate([sequences, points], axis=1)

#         streamlines_dwi = np.concatenate([streamlines_dwi[undone],
#                                           map_coordinates_3d_4d(dwi, sequences[:, [-1]])],
#                                          axis=1)

#     # Add remaining
#     streamlines.extend([s for s in sequences])
#     return streamlines


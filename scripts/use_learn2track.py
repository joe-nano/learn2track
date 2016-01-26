#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse

import nibabel as nib

import theano
import theano.tensor as T

from dipy.tracking.streamline import compress_streamlines

from learn2track.rnn import RNN
from learn2track.lstm import LSTM, LSTM_regression
from learn2track.utils import Timer, map_coordinates_3d_4d, normalize_dwi
from smartlearner.utils import load_dict_from_json_file

from smartlearner import Dataset

floatX = theano.config.floatX
NB_POINTS = 100


def buildArgsParser():
    DESCRIPTION = "Script to use a trained LSTM to do tractography."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('experiment', type=str, help="name of the experiment.")
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--wm_mask', type=str, help="streamlines will stop if going outside this mask (.nii|.nii.gz).")
    p.add_argument('--bvals', type=str, help='text file with the bvalues. Default: same name as the dwi file but with extension .bvals.')
    p.add_argument('--seeding_mask', type=str, help="streamlines will start from this mask (.nii|.nii.gz).")
    p.add_argument('--trk', type=str, help="perform some qualitative evaluation using a testset bundle (.trk).")

    return p


def load_testset_bundles(bundles_path):
    dataset_name = "ISMRM15_Challenge"

    bundles = {'testset': []}
    for f in os.listdir(bundles_path):
        if f.endswith("_testset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            data = np.load(pjoin(bundles_path, f))
            dataset = Dataset(data['inputs'].astype(floatX), data['targets'].astype(floatX), name=bundle_name, keep_on_cpu=True)
            bundles["testset"].append(dataset)

    testset_inputs = np.concatenate([b.inputs.get_value() for b in bundles["testset"]])
    testset_targets = np.concatenate([b.targets.get_value() for b in bundles["testset"]])
    testset = Dataset(testset_inputs, testset_targets, name=dataset_name+"_testset")

    return testset


def track(model, dwi, seeds, step_size=0.5, allowed_voxels=None, max_nb_points=500):
    inputs = T.tensor3('inputs')
    inputs.tag.test_value = map_coordinates_3d_4d(dwi, np.asarray(seeds)[:, None, :])
    next_directions, stoppings = model.use(inputs)
    track_step = theano.function([inputs], [next_directions[:, -1], stoppings[:, -1, 0]])

    streamlines = []
    sequences = np.asarray(seeds)[:, None, :]
    streamlines_dwi = map_coordinates_3d_4d(dwi, sequences[:, [-1]])

    for i in range(max_nb_points):
        if (i+1) % 10 == 0:
            print("{}/{}".format(i+1, max_nb_points))

        directions, prob_stoppings = track_step(streamlines_dwi)
        directions *= step_size

        done = list(np.where(prob_stoppings > 0.5)[0])
        undone = list(np.where(prob_stoppings <= 0.5)[0])

        # If a streamline goes outside the wm mask, mark is as done.
        if allowed_voxels is not None:
            next_points = sequences[undone, -1] + directions[undone]
            next_points_voxel = np.round(next_points)
            for idx, voxel in zip(undone, next_points_voxel):
                if tuple(voxel) not in allowed_voxels:
                    done += [idx]
                    undone.remove(idx)

        streamlines.extend([s for s in sequences[done]])

        if len(undone) == 0:
            break

        sequences = sequences[undone]
        points = sequences[:, [-1]] + directions[undone, None, :]
        sequences = np.concatenate([sequences, points], axis=1)

        streamlines_dwi = np.concatenate([streamlines_dwi[undone],
                                          map_coordinates_3d_4d(dwi, sequences[:, [-1]])],
                                         axis=1)

    # Add remaining
    streamlines.extend([s for s in sequences])
    return streamlines


def track_no_stopping(model, dwi, seeds, step_size=0.5, allowed_voxels=None, max_nb_points=500):
    inputs = T.tensor3('inputs')
    inputs.tag.test_value = map_coordinates_3d_4d(dwi, np.asarray(seeds)[:, None, :])
    next_directions = model.use(inputs)
    track_step = theano.function([inputs], next_directions[:, -1])

    streamlines = []
    sequences = np.asarray(seeds)[:, None, :]
    streamlines_dwi = map_coordinates_3d_4d(dwi, sequences[:, [-1]])

    for i in range(max_nb_points):
        if (i+1) % 10 == 0:
            print("{}/{}".format(i+1, max_nb_points))

        directions = track_step(streamlines_dwi)
        directions *= step_size

        done = []
        undone = list(range(len(directions)))

        # If a streamline goes outside the wm mask, mark is as done.
        if allowed_voxels is not None:
            next_points = sequences[undone, -1] + directions[undone]
            next_points_voxel = np.round(next_points)
            for idx, voxel in zip(undone, next_points_voxel):
                if tuple(voxel) not in allowed_voxels:
                    done += [idx]
                    undone.remove(idx)

        streamlines.extend([s for s in sequences[done]])

        if len(undone) == 0:
            break

        sequences = sequences[undone]
        points = sequences[:, [-1]] + directions[undone, None, :]
        sequences = np.concatenate([sequences, points], axis=1)

        streamlines_dwi = np.concatenate([streamlines_dwi[undone],
                                          map_coordinates_3d_4d(dwi, sequences[:, [-1]])],
                                         axis=1)

    # Add remaining
    streamlines.extend([s for s in sequences])
    return streamlines


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

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

    with Timer("Loading model"):
        meta = load_dict_from_json_file(pjoin(args.experiment, "meta.json"))
        if meta["name"] == RNN.__name__:
            model = RNN.load(args.experiment)
        elif meta["name"] == LSTM.__name__:
            model = LSTM.load(args.experiment)
            track_fct = track
        elif meta["name"] == LSTM_regression.__name__:
            model = LSTM_regression.load(args.experiment)
            track_fct = track_no_stopping
        else:
            raise ValueError("Unknown class: {}".format(meta["name"]))

    with Timer("Generating seeds"):
        step_size = 0.5
        if args.trk is not None:
            streamlines = nib.streamlines.load(args.trk, ref=args.dwi)
            #step_size = np.mean([np.mean(np.sqrt(np.sum((pts[1:]-pts[:-1])**2, axis=1))) for pts in streamlines.points])
            #seeds = [s[0] for s in streamlines.points[::10]]
            seeds = [s[0] for s in streamlines.points]
            seeds += [s[-1] for s in streamlines.points]

    with Timer("Tracking"):
        new_streamlines = track_fct(model, weights, seeds, step_size, allowed_voxels)

    with Timer("Saving streamlines"):
        new_streamlines = compress_streamlines(new_streamlines)
        s = nib.streamlines.Streamlines(new_streamlines)
        save_path = pjoin(args.experiment, "generated_streamlines.trk")
        nib.streamlines.save(s, save_path, ref=args.dwi)

if __name__ == "__main__":
    main()

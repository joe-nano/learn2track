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
import time

import dipy
import nibabel as nib
from dipy.tracking.streamline import compress_streamlines

from smartlearner import Dataset
from smartlearner import utils as smartutils

from learn2track import utils
from learn2track.utils import Timer, map_coordinates_3d_4d, normalize_dwi
from smartlearner.utils import load_dict_from_json_file


def build_argparser():
    DESCRIPTION = "Generate a tractogram from a LSTM model trained on ismrm2015 challenge data."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--out', type=str, default="tractogram.tck",
                   help="name of the output tractogram (.tck|.trk). Default: tractogram.tck")

    p.add_argument('--seeds', type=str, nargs="+", required=True,
                   help="use extermities of the streamlines in these tractograms (.trk|.tck) as seed points.")
    p.add_argument('--nb-seeds-per-voxel', type=int, default=1,
                   help="number of seeds per voxel, only if --seeds is a seeding mask (i.e. a nifti file). Default: 1")
    p.add_argument('--seeding-rng-seed', type=int, default=1234,
                   help="seed for the random generator responsible of generating the seeds in the voxels. Default: 1234")

    deviation_angle = p.add_mutually_exclusive_group()
    deviation_angle.add_argument('--theta', metavar='ANGLE', type=float,
                                 help="Maximum angle between 2 steps (in degree). [45]")
    deviation_angle.add_argument('--curvature', metavar='RADIUS', type=float,
                                 help='Minimum radius of curvature R in mm. Replaces --theta.')

    p.add_argument('--min-length', type=int, help="minimum length (in mm) for a streamline. Default: 10 mm", default=10)
    p.add_argument('--max-length', type=int, help="maximum length (in mm) for a streamline. Default: 300 mm", default=300)
    p.add_argument('--step-size', type=float, help="step size between two consecutive points in a streamlines (in mm). Default: 0.5mm", default=0.5)
    p.add_argument('--mask', type=str,
                   help="if provided, streamlines will stop if going outside this mask (.nii|.nii.gz).")
    p.add_argument('--mask-threshold', type=float, default=0.05,
                   help="streamlines will be terminating if they pass through a voxel with a value from the mask lower than this value. Default: 0.05")

    p.add_argument('--enable-backward-tracking', action="store_true",
                   help="if specified, both senses of the direction obtained from the seed point will be explored.")


    p.add_argument('--append-previous-direction', action="store_true",
                   help="if specified, the target direction of the last timestep will be concatenated to the input of the current timestep. (0,0,0) will be used for the first timestep.")

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite existing tractogram')

    return p

floatX = theano.config.floatX


def track(model, dwi, seeds, step_size=0.5, max_nb_points=1000, theta=0.78, mask=None, mask_threshold=0.05, enable_backward_tracking=False):
    streamlines_dwi = np.zeros((len(seeds), dwi.shape[-1]), dtype=np.float32)

    # Forward tracking
    # Prepare some data container and reset the model.
    sequences = seeds.copy()
    if sequences.ndim == 2:
        sequences = sequences[:, None, :]

    directions = np.zeros((len(sequences), 3), dtype=np.float32)
    last_directions = np.zeros((len(sequences), 3), dtype=np.float32)

    streamlines_lengths = np.zeros(len(seeds), dtype=np.int16)
    undone = np.ones(len(sequences), dtype=bool)

    model.seq_reset(batch_size=len(seeds))

    # Tracking
    for i in range(max_nb_points):
        if (i+1) % 100 == 0:
            print("pts: {}/{}".format(i+1, max_nb_points))

        streamlines_dwi[undone] = map_coordinates_3d_4d(dwi, sequences[undone, -1, :])
        directions[undone] = model.seq_next(streamlines_dwi[undone])

        # If a streamline makes a turn to tight, stop it.
        if sequences.shape[1] > 1:
            angles = np.arccos(np.sum(last_directions * directions, axis=1))  # Normed directions.
            model.seq_squeeze(tokeep=angles[undone] <= theta)
            undone = np.logical_and(undone, angles <= theta)

        last_directions[undone] = directions[undone].copy()

        # Make a step
        directions[undone] = directions[undone] * step_size
        sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)
        streamlines_lengths[:] += undone[:]

        # If a streamline goes outside the wm mask, mark is as done.
        if mask is not None:
            last_point_values = map_coordinates_3d_4d(mask, sequences[:, -1, :])
            model.seq_squeeze(tokeep=last_point_values[undone] >= mask_threshold)
            undone = np.logical_and(undone, last_point_values >= mask_threshold)

        if undone.sum() == 0:
            break

    # Trim sequences to obtain the streamlines.
    streamlines = [s[:l] for s, l in zip(sequences, streamlines_lengths)]

    if not enable_backward_tracking:
        return streamlines

    # Backward tracking
    # Reset everything
    streamlines_dwi = np.zeros((len(sequences), dwi.shape[-1]), dtype=np.float32)

    sequences = seeds.copy()
    if sequences.ndim == 2:
        sequences = sequences[:, None, :]

    directions = np.zeros((len(sequences), 3), dtype=np.float32)
    last_directions = np.zeros((len(sequences), 3), dtype=np.float32)

    streamlines_lengths = np.zeros(len(seeds), dtype=np.int16)
    undone = np.ones(len(sequences), dtype=bool)

    model.seq_reset(batch_size=len(seeds))

    # Tracking
    print("Tracking backward...")
    for i in range(max_nb_points):
        if (i+1) % 100 == 0:
            print("pts: {}/{}".format(i+1, max_nb_points))

        streamlines_dwi[undone] = map_coordinates_3d_4d(dwi, sequences[undone, -1, :])
        directions[undone] = model.seq_next(streamlines_dwi[undone])
        directions[undone] = directions[undone] * step_size

        if i == 0:
            # Follow the opposite direction obtained from the seed points (and only for that point).
            directions[undone] *= -1

        # If a streamline makes a turn to tight, stop it.
        if sequences.shape[1] > 1:
            angles = np.arccos(np.sum(last_directions * directions, axis=1))  # Normed directions.
            model.seq_squeeze(tokeep=angles[undone] <= theta)
            undone = np.logical_and(undone, angles <= theta)

        last_directions[undone] = directions[undone]

        # Make a step
        directions[undone] = directions[undone] * step_size
        sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)
        streamlines_lengths[:] += undone[:]

        # If a streamline goes outside the wm mask, mark is as done.
        if mask is not None:
            last_point_values = map_coordinates_3d_4d(mask, sequences[:, -1, :])
            model.seq_squeeze(tokeep=last_point_values[undone] >= mask_threshold)
            undone = np.logical_and(undone, last_point_values >= mask_threshold)

        if undone.sum() == 0:
            break

    # Trim sequences to obtain the streamlines.
    streamlines = [np.r_[s[::-1], seq[1:l]] for s, seq, l in zip(streamlines, sequences, streamlines_lengths)]
    return streamlines


def batch_track(model, dwi, seeds, step_size=0.5, max_nb_points=500, theta=0.78, mask=None, mask_threshold=0.05, enable_backward_tracking=False):
    batch_size = len(seeds)
    while True:
        try:
            time.sleep(1)
            print("Trying to track {:,} streamlines at the same time.".format(batch_size))
            tractogram = nib.streamlines.Tractogram()

            for start in range(0, len(seeds), batch_size):
                print("{:,} / {:,}".format(start, len(seeds)))
                end = start+batch_size
                new_streamlines = track(model=model, dwi=dwi, seeds=seeds[start:end], step_size=step_size,
                                        max_nb_points=max_nb_points, theta=theta,
                                        mask=mask, mask_threshold=mask_threshold,
                                        enable_backward_tracking=enable_backward_tracking)
                new_streamlines = compress_streamlines(new_streamlines)
                tractogram.streamlines.extend(new_streamlines)

            return tractogram

        except MemoryError:
            print("{:,} streamlines is too much!".format(batch_size))
            batch_size //= 2
            if batch_size < 0:
                raise MemoryError("Might needs a bigger graphic card!")

        except RuntimeError as e:
            if "out of memory" in e.args[0]:
                print("{:,} streamlines is too much!".format(batch_size))
                batch_size //= 2
                if batch_size < 0:
                    raise MemoryError("Might needs a bigger graphic card!")

            else:
                raise e


def get_max_angle_from_curvature(curvature, step_size):
    """
    Parameters
    ----------
    curvature: float
        Minimum radius of curvature in mm.
    step_size: float
        The tracking step size in mm.

    Return
    ------
    theta: float
        The maximum deviation angle in radian,
        given the radius curvature and the step size.
    """
    theta = 2. * np.arcsin(step_size / (2. * curvature))
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return theta



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
        if "classification" in hyperparams and hyperparams["classification"]:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Softmax
                model_class = LSTM_Softmax
            elif hyperparams["model"] == "lstm_hybrid":
                from learn2track.lstm import LSTM_Hybrid
                model_class = LSTM_Hybrid

        elif "regression" in hyperparams and hyperparams["regression"]:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Regression
                model_class = LSTM_Regression
            elif hyperparams["model"] == "lstm_extraction":
                from learn2track.lstm import LSTM_RegressionWithFeaturesExtraction
                model_class = LSTM_RegressionWithFeaturesExtraction
            if hyperparams["model"] == "gru":
                from learn2track.gru import GRU_Regression
                model_class = GRU_Regression

                if args.append_previous_direction:
                    raise NameError("Not implemented")
                    from learn2track.gru import GRU_RegressionWithScheduledSampling
                    model_class = GRU_RegressionWithScheduledSampling

        else:
            from learn2track.gru import GRU_Regression
            model_class = GRU_Regression

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

        if args.append_previous_direction:
            model.scheduled_sampling_rate.var.set_value(1)

    with Timer("Loading DWIs"):
        dwi = nib.load(args.dwi)

        # Load gradients table
        bvals_filename = args.dwi.split('.')[0] + ".bvals"
        bvecs_filename = args.dwi.split('.')[0] + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        weights = utils.resample_dwi(dwi, bvals, bvecs).astype(np.float32)  # Resample to 100 directions

    mask = None
    if args.mask is not None:
        with Timer("Loading mask"):
            mask = nib.load(args.mask).get_data()

    with Timer("Generating seeds"):
        seeds = []
        for filename in args.seeds:
            if filename.endswith('.trk') or filename.endswith('.tck'):
                tfile = nib.streamlines.load(filename)
                # Send the streamlines to voxel since that's where we'll track.
                tfile.tractoram.apply_affine(np.linalg.inv(dwi.affine))

                # Use extremities of the streamlines as seeding points.
                seeds += [s[0] for s in tfile.streamlines]
                seeds += [s[-1] for s in tfile.streamlines]

            else:
                # Assume it is a binary mask.
                rng = np.random.RandomState(args.seeding_rng_seed)
                nii_seeds = nib.load(filename)
                indices = np.array(np.where(nii_seeds.get_data())).T
                for idx in indices:
                    seeds.extend(idx + rng.rand(args.nb_seeds_per_voxel, 3))

        seeds = np.array(seeds)

    with Timer("Tracking"):
        voxel_sizes = np.asarray(dwi.header.get_zooms()[:3])
        if not np.all(voxel_sizes == dwi.header.get_zooms()[0]):
            print("* Careful voxel are anisotropic {}!".format(tuple(voxel_sizes)))
        # Since we are tracking in voxel space, convert step_size (in mm) to voxel.
        step_size = np.float32(args.step_size / voxel_sizes.max())
        # Also convert max length (in mm) to voxel.
        max_nb_points = int(args.max_length / step_size)

        if args.theta is not None:
            theta = np.deg2rad(args.theta)
        elif args.curvature is not None and args.curvature > 0:
            theta = get_max_angle_from_curvature(args.curvature, step_size)
        else:
            theta = np.deg2rad(45)

        print("Angle: {}".format(theta))

        tractogram = batch_track(model, weights, seeds,
                                 step_size=step_size,
                                 max_nb_points=max_nb_points,
                                 theta=theta,
                                 mask=mask,
                                 mask_threshold=args.mask_threshold,
                                 enable_backward_tracking=args.enable_backward_tracking)

    with Timer("Saving streamlines"):
        # Flush streamlines that has no points.
        tractogram = tractogram[np.array(list(map(len, tractogram))) > 0]
        tractogram.apply_affine(dwi.affine)  # Streamlines were generated in voxel space.
        # Remove small streamlines
        lengths = dipy.tracking.streamline.length(tractogram.streamlines)
        tractogram = tractogram[lengths >= args.min_length]

        save_path = pjoin(experiment_path, args.out)
        nib.streamlines.save(tractogram, save_path)

    print("{:,} streamlines (compressed) were generated with an average length of {:.2f} mm.".format(len(tractogram), lengths.mean()))

if __name__ == "__main__":
    main()

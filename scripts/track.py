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
from learn2track.utils import Timer
from smartlearner.utils import load_dict_from_json_file

from learn2track import neurotools


def build_argparser():
    DESCRIPTION = "Generate a tractogram from a LSTM model trained on ismrm2015 challenge data."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--ref-dwi', type=str, help="[Experimental] will be used to normalized the diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--ref-dwi2', type=str, help="[Experimental] will be used to normalized the diffusion weighted images (.nii|.nii.gz).")
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
    p.add_argument('--step-size', type=float, help="step size between two consecutive points in a streamlines (in mm). Default: use model's output as-is")
    p.add_argument('--mask', type=str,
                   help="if provided, streamlines will stop if going outside this mask (.nii|.nii.gz).")
    p.add_argument('--mask-threshold', type=float, default=0.05,
                   help="streamlines will be terminating if they pass through a voxel with a value from the mask lower than this value. Default: 0.05")

    p.add_argument('--backward-tracking-algo', type=int,
                   help="if specified, both senses of the direction obtained from the seed point will be explored. Default: 0, (i.e. only do tracking one direction) ", default=0)

    p.add_argument('--batch-size', type=int, help="number of streamlines to process at the same time. Default: the biggest possible")

    p.add_argument('--dilate-mask', action="store_true",
                   help="if specified, apply binary dilation on the tracking mask.")

    p.add_argument('--append-previous-direction', action="store_true",
                   help="if specified, the target direction of the last timestep will be concatenated to the input of the current timestep. (0,0,0) will be used for the first timestep.")

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite existing tractogram')

    return p

floatX = theano.config.floatX


def track(model, dwi, seeds, step_size=None, max_nb_points=1000, theta=0.78, mask=None, mask_affine=None, mask_threshold=0.05, backward_tracking_algo=0):
    # Forward tracking
    # Prepare some data container and reset the model.
    sequences = seeds.copy()
    if sequences.ndim == 2:
        sequences = sequences[:, None, :]

    directions = np.zeros((len(seeds), 3), dtype=np.float32)
    all_directions = np.zeros((len(seeds), 0, 3), dtype=np.float32)
    last_directions = np.zeros((len(seeds), 3), dtype=np.float32)

    streamlines_lengths = np.zeros(len(seeds), dtype=np.int16)
    undone = np.ones(len(sequences), dtype=bool)

    model.seq_reset(batch_size=len(seeds))
    stopping_types_count = {'curv': 0,
                            'mask': 0}

    # Tracking
    for i in range(max_nb_points):
        if (i+1) % 100 == 0:
            print("pts: {}/{}".format(i+1, max_nb_points))

        directions[undone] = model.seq_next(sequences[undone, -1, :])   # Get next unnormalized directions
        normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))  # Normed directions.
        # print(np.sqrt(np.sum(directions[undone]**2, axis=1, keepdims=True)))

        # If a streamline makes a turn to tight, stop it.
        if sequences.shape[1] > 1:
            # TEND-like approach
            # directions[undone] = 0.5*last_directions[undone] + 0.5*directions[undone]

            angles = np.arccos(np.sum(last_directions * normalized_directions, axis=1))
            model.seq_squeeze(tokeep=angles[undone] <= theta)
            stopping_types_count['curv'] += np.sum(angles[undone] > theta)
            undone = np.logical_and(undone, angles <= theta)

        if step_size is not None:
            directions[undone] = normalized_directions[undone] * step_size

        all_directions = np.concatenate([all_directions, directions[:, None, :]], axis=1)
        last_directions[undone] = normalized_directions[undone].copy()

        # Make a step
        directions[np.logical_not(undone)] = np.nan  # Help debugging
        sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)
        streamlines_lengths[:] += undone[:]

        # If a streamline goes outside the wm mask, mark it as done.
        if mask is not None:
            last_point_values = neurotools.map_coordinates_3d_4d(mask, sequences[:, -1, :], affine=mask_affine, order=1)
            model.seq_squeeze(tokeep=last_point_values[undone] > mask_threshold)
            stopping_types_count['mask'] += np.sum(last_point_values[undone] <= mask_threshold)
            undone = np.logical_and(undone, last_point_values > mask_threshold)

        sequences[np.logical_not(undone), i+1] = np.nan  # Help debugging

        if undone.sum() == 0:
            break

    print("Max length: {}".format(streamlines_lengths.max()))
    print("Forward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(stopping_types_count['mask'],
                                                                                            stopping_types_count['curv'],
                                                                                            undone.sum()))

    # Trim sequences to obtain the streamlines.
    streamlines = [s[:nb_pts] for s, nb_pts in zip(sequences, streamlines_lengths)]

    norm = lambda x: np.sqrt(np.sum(x**2, axis=1))
    avg_step_sizes = [np.mean(norm(s[1:]-s[:-1])) for s in streamlines]
    print("Average step sizes: {:.2f} mm.".format(np.nanmean(avg_step_sizes)))

    if backward_tracking_algo == 0:
        return streamlines

    if backward_tracking_algo == 1:
        # Reset everything and start tracking from the opposite direction.

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

            directions[undone] = model.seq_next(sequences[undone, -1, :])   # Get next unnormalized directions
            normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))  # Normed directions.

            if i == 0:
                # Follow the opposite direction obtained from the seed points (and only for that point).
                directions[undone] *= -1

            # If a streamline makes a turn to tight, stop it.
            if sequences.shape[1] > 1:
                angles = np.arccos(np.sum(last_directions * normalized_directions, axis=1))
                model.seq_squeeze(tokeep=angles[undone] <= theta)
                undone = np.logical_and(undone, angles <= theta)

            last_directions[undone] = normalized_directions[undone].copy()

            # Make a step
            if step_size is not None:
                directions[undone] = normalized_directions[undone] * step_size  # TODO: Model should have learned the step size.
            directions[np.logical_not(undone)] = np.nan  # Help debugging
            sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)
            streamlines_lengths[:] += undone[:]

            # If a streamline goes outside the wm mask, mark it as done.
            if mask is not None:
                last_point_values = neurotools.map_coordinates_3d_4d(mask, sequences[:, -1, :], affine=mask_affine, order=1)
                model.seq_squeeze(tokeep=last_point_values[undone] > mask_threshold)
                undone = np.logical_and(undone, last_point_values > mask_threshold)

            if undone.sum() == 0:
                break

        # Trim sequences to obtain the streamlines.
        streamlines = [np.r_[s[::-1], seq[1:l]] for s, seq, l in zip(streamlines, sequences, streamlines_lengths)]
        return streamlines

    elif backward_tracking_algo == 2:
        # Reset everything, kickstart the model with the first half of the streamline and resume tracking.

        # Reverse streamline
        forward_streamlines_lengths = streamlines_lengths.copy()
        r_directions = np.zeros((len(sequences), streamlines_lengths.max(), 3), dtype=sequences.dtype)
        for i, l in enumerate(streamlines_lengths):
            r_directions[i, :l-1] = -all_directions[i, :l-1][::-1]

        # old_streamlines_lengths = streamlines_lengths  # DEBUG
        # old_sequences = sequences  # DEBUG
        new_sequences = sequences[range(len(sequences)), streamlines_lengths-1].copy()
        sequences = new_sequences[:, None, :]

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

            resuming = i < forward_streamlines_lengths-1

            directions[undone] = model.seq_next(sequences[undone, -1, :])   # Get next unnormalized directions
            normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))  # Normed directions.

            # If a streamline makes a turn to tight, stop it.
            if sequences.shape[1] > 1:
                # TEND-like approach
                # directions[undone] = 0.5*last_directions[undone] + 0.5*directions[undone]

                angles = np.arccos(np.sum(last_directions * normalized_directions, axis=1))
                model.seq_squeeze(tokeep=np.logical_or(angles[undone] <= theta, resuming[undone]))
                undone = np.logical_or(np.logical_and(undone, angles <= theta), resuming)

            # Overwrite directions for all streamlines still being kickstarted (resumed).
            if i < r_directions.shape[1]:
                directions[resuming] = r_directions[resuming, i, :]
            else:
                if step_size is not None:
                    directions[undone] = normalized_directions[undone] * step_size  # TODO: Model should have learned the step size.
            last_directions[undone] = normalized_directions[undone].copy()

            # Make a step
            directions[np.logical_not(undone)] = np.nan  # Help debugging
            sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)
            streamlines_lengths[:] += undone[:]

            # If a streamline goes outside the wm mask, mark it as done.
            if mask is not None:
                last_point_values = neurotools.map_coordinates_3d_4d(mask, sequences[:, -1, :], affine=mask_affine, order=1)
                model.seq_squeeze(tokeep=np.logical_or(last_point_values[undone] > mask_threshold, resuming[undone]))
                undone = np.logical_or(np.logical_and(undone, last_point_values > mask_threshold), resuming)

            if undone.sum() == 0:
                break

        # Trim sequences to obtain the streamlines.
        streamlines = [s[:l] for s, l in zip(sequences, streamlines_lengths)]
        return streamlines


def batch_track(model, dwi, seeds, step_size=None, max_nb_points=500, theta=0.78, mask=None, mask_affine=None, mask_threshold=0.05, backward_tracking_algo=0, batch_size=None):
    if batch_size is None:
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
                                        mask=mask, mask_affine=mask_affine, mask_threshold=mask_threshold,
                                        backward_tracking_algo=backward_tracking_algo)

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

    with Timer("Loading DWIs"):
        # Load gradients table
        bvals_filename = args.dwi.split('.')[0] + ".bvals"
        bvecs_filename = args.dwi.split('.')[0] + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        if hyperparams["use_sh_coeffs"]:
            # Use 45 spherical harmonic coefficients to represent the diffusion signal.
            weights = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)
        else:
            # Resample the diffusion signal to have 100 directions.
            weights = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

        affine_rasmm2dwivox = np.linalg.inv(dwi.affine)

        if args.ref_dwi is not None:
            # Load gradients table
            ref_bvals_filename = args.ref_dwi.split('.')[0] + ".bvals"
            ref_bvecs_filename = args.ref_dwi.split('.')[0] + ".bvecs"
            ref_bvals, ref_bvecs = dipy.io.gradients.read_bvals_bvecs(ref_bvals_filename, ref_bvecs_filename)

            ref_dwi = nib.load(args.ref_dwi)
            ref_dwi2 = nib.load(args.ref_dwi2)
            ref_weights = neurotools.resample_dwi(ref_dwi, ref_bvals, ref_bvecs).astype(np.float32)  # Resample to 100 directions
            ref_weights2 = neurotools.resample_dwi(ref_dwi2, ref_bvals, ref_bvecs).astype(np.float32)  # Resample to 100 directions
            #affine_rasmm2dwivox = np.linalg.inv(dwi.affine)

            print ("Minmax:      ", weights.min(), weights.max())
            print ("Minmax (ref):", ref_weights.min(), ref_weights.max())

            import pylab as plt
            # plt.hist(weights[weights!=0], normed=True, bins=np.linspace(0, 1), alpha=0.5)
            # plt.hist(ref_weights[ref_weights!=0], normed=True, bins=np.linspace(0, 1), alpha=0.5)

            # Normalization over all voxels and all directions
            # non_zero = weights > 0
            # std = weights[non_zero].std()
            # mean = weights[non_zero].mean()
            # ref_std = ref_weights[ref_weights!=0].std()
            # ref_mean = ref_weights[ref_weights!=0].mean()
            # standardized_weights = (weights[non_zero] - mean) / std
            # weights[non_zero] = (standardized_weights * ref_std) + ref_mean

            non_zero = weights > 0
            std = weights.std()
            mean = weights.mean()
            ref_std = ref_weights.std()
            ref_mean = ref_weights.mean()
            standardized_weights = (weights - mean) / std
            weights = (standardized_weights * ref_std) + ref_mean

            # # Normalization over all voxels
            # std = weights.std(axis=-1, keepdims=True)
            # mean = weights.mean(axis=-1, keepdims=True)
            # ref_std = ref_weights.std(axis=-1, keepdims=True)
            # ref_mean = ref_weights.mean(axis=-1, keepdims=True)
            # standardized_weights = (weights - mean) / std
            # weights = (standardized_weights * ref_std) + ref_mean
            # print (weights.sum())
            # weights[np.isnan(weights)] = 0

            # # Normalization over all directions
            # std = weights.std(axis=(0,1,2), keepdims=True)
            # mean = weights.mean(axis=(0,1,2), keepdims=True)
            # ref_std = ref_weights.std(axis=(0,1,2), keepdims=True)
            # ref_mean = ref_weights.mean(axis=(0,1,2), keepdims=True)
            # standardized_weights = (weights - mean) / std
            # weights = (standardized_weights * ref_std) + ref_mean
            # print (weights.sum())
            # weights[np.isnan(weights)] = 0

            # for i in range(weights.shape[-1]):
            #     std = weights[..., i][weights[..., i]>1e-2].std()
            #     mean = weights[..., i][weights[..., i]!=0].mean()
            #     ref_std = ref_weights[..., i][ref_weights[..., i]!=0].std()
            #     ref_mean = ref_weights[..., i][ref_weights[..., i]!=0].mean()
            #     standardized_weights = (weights[..., i][weights[..., i]!=0] - mean) / std
            #     weights[..., i][weights[..., i]!=0] = (standardized_weights * ref_std) + ref_mean
            #     print (weights.sum())

            # weights[np.isnan(weights)] = 0
            # weights[ref_weights==0] = 0

            plt.figure()
            # plt.hist(weights[weights!=0], normed=True, bins=np.linspace(0, 1), alpha=0.5)
            # plt.hist(ref_weights[ref_weights!=0], normed=True, bins=np.linspace(0, 1), alpha=0.5)

            # plt.show()

    with Timer("Loading model"):
        kwargs = {}
        if hyperparams["model"] == "gru_regression":
            from learn2track.models import GRU_Regression
            model_class = GRU_Regression
            volume_manager = neurotools.VolumeManager()
            volume_manager.register(weights)
            kwargs['volume_manager'] = volume_manager

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path), **kwargs)  # Create new instance and restore model.
        print(str(model))

    mask = None
    if args.mask is not None:
        with Timer("Loading mask"):
            mask_nii = nib.load(args.mask)
            mask = mask_nii.get_data()
            # Compute the affine allowing to evaluate the mask at some coordinates correctly.

            mask_affine = np.dot(affine_rasmm2dwivox, mask_nii.affine)
            if args.dilate_mask:
                import scipy
                mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

    with Timer("Generating seeds"):
        seeds = []

        for filename in args.seeds:
            if filename.endswith('.trk') or filename.endswith('.tck'):
                tfile = nib.streamlines.load(filename)
                # Send the streamlines to voxel since that's where we'll track.
                tfile.tractogram.apply_affine(affine_rasmm2dwivox)

                # Use extremities of the streamlines as seeding points.
                seeds += [s[0] for s in tfile.streamlines]
                seeds += [s[-1] for s in tfile.streamlines]

            else:
                # Assume it is a binary mask.
                rng = np.random.RandomState(args.seeding_rng_seed)
                nii_seeds = nib.load(filename)
                seeds_affine = np.dot(affine_rasmm2dwivox, nii_seeds.affine)

                indices = np.array(np.where(nii_seeds.get_data())).T
                for idx in indices:
                    seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(args.nb_seeds_per_voxel, 3))
                    seeds_in_voxel = nib.affines.apply_affine(seeds_affine, seeds_in_voxel)
                    seeds.extend(seeds_in_voxel)

        seeds = np.array(seeds, dtype=theano.config.floatX)

    with Timer("Tracking in the diffusion voxel space"):
        voxel_sizes = np.asarray(dwi.header.get_zooms()[:3])
        if not np.all(voxel_sizes == dwi.header.get_zooms()[0]):
            print("* Careful voxel are anisotropic {}!".format(tuple(voxel_sizes)))
        # Since we are tracking in diffusion voxel space, convert step_size (in mm) to voxel.

        if args.step_size is not None:
            step_size = np.float32(args.step_size / voxel_sizes.max())
            # Also convert max length (in mm) to voxel.
            max_nb_points = int(args.max_length / step_size)
        else:
            step_size = None
            max_nb_points = args.max_length

        if args.theta is not None:
            theta = np.deg2rad(args.theta)
        elif args.curvature is not None and args.curvature > 0:
            theta = get_max_angle_from_curvature(args.curvature, step_size)
        else:
            theta = np.deg2rad(45)

        print("Angle: {}".format(np.rad2deg(theta)))
        print("Step size (vox): {}".format(step_size))
        print("Max nb. points: {}".format(max_nb_points))

        tractogram = batch_track(model, weights, seeds,
                                 step_size=step_size,
                                 max_nb_points=max_nb_points,
                                 theta=theta,
                                 mask=mask, mask_affine=mask_affine,
                                 mask_threshold=args.mask_threshold,
                                 backward_tracking_algo=args.backward_tracking_algo,
                                 batch_size=args.batch_size)

    with Timer("Saving streamlines"):
        # Flush streamlines that has no points.
        tractogram = tractogram[np.array(list(map(len, tractogram))) > 0]
        tractogram.affine_to_rasmm = dwi.affine
        # tractogram.apply_affine(dwi.affine)  # Streamlines were generated in diffusion voxel space.
        # Remove small streamlines
        lengths = dipy.tracking.streamline.length(tractogram.streamlines)
        tractogram = tractogram[lengths >= args.min_length]
        lengths = lengths[lengths >= args.min_length]

        save_path = pjoin(experiment_path, args.out)
        try:  # Create dirs, if needed.
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

        nib.streamlines.save(tractogram, save_path)

    print("{:,} streamlines (compressed) were generated.".format(len(tractogram)))
    print("Average length: {:.2f} mm.".format(lengths.mean()))
    print("Minimum length: {:.2f} mm. Maximum length: {:.2f}".format(lengths.min(), lengths.max()))

if __name__ == "__main__":
    main()

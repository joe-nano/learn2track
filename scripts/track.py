#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import os

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
from os.path import join as pjoin

import theano
import time

import dipy
import nibabel as nib
from dipy.tracking.streamline import compress_streamlines

from smartlearner import views
from smartlearner import utils as smartutils

from learn2track import datasets, batch_schedulers
from learn2track.factories import loss_factory
from learn2track.utils import Timer

from learn2track import neurotools

floatX = theano.config.floatX

# Constant
STOPPING_MASK =       int('00000001', 2)
STOPPING_LENGTH =     int('00000010', 2)
STOPPING_CURVATURE =  int('00000100', 2)
STOPPING_LIKELIHOOD = int('00001000', 2)


def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    return ((flags & ref_flag) >> np.log2(ref_flag).astype(np.uint8)).astype(bool)

def count_flags(flags, ref_flag):
    """ Counts how many flags have the `ref_flag` set. """
    return is_flag_set(flags, ref_flag).sum()


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

    p.add_argument('--filter-threshold', type=float,
                   help="If specified, only streamlines with a loss value lower than the specified value will be kept.")

    p.add_argument('--batch-size', type=int, help="number of streamlines to process at the same time. Default: the biggest possible")

    p.add_argument('--dilate-mask', action="store_true",
                   help="if specified, apply binary dilation on the tracking mask.")

    p.add_argument('--discard-stopped-by-curvature', action="store_true",
                   help='if specified, discard streamlines having a too high curvature (i.e. tracking stopped because of that).')
    p.add_argument('--append-previous-direction', action="store_true",
                   help="if specified, the target direction of the last timestep will be concatenated to the input of the current timestep. (0,0,0) will be used for the first timestep.")

    p.add_argument('--pft-nb-retry', type=int,
                   help="Number of retry to use for the particle filtering tractography inspired by Girard et al. (2014) NeuroImage. Default: don't use PFT at all.")

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite existing tractogram')

    return p


def compute_loss_errors(streamlines, model, hyperparams):
    # Create dummy dataset for these new streamlines.
    tracto_data = neurotools.TractographyData(None, None, None)
    tracto_data.add(streamlines, bundle_name="Generated")
    tracto_data.subject_id = 0
    dataset = datasets.TractographyDataset([tracto_data], "Generated", keep_on_cpu=True)

    loss = loss_factory(hyperparams, model, dataset)
    if hyperparams['model'] == 'gru_multistep':
        batch_scheduler = batch_schedulers.MultistepSequenceBatchScheduler(dataset,
                                                                           batch_size=1000,
                                                                           k=1,
                                                                           noisy_streamlines_sigma=None,
                                                                           use_data_augment=False,  # Otherwise it doubles the number of losses :-/
                                                                           seed=1234,
                                                                           shuffle_streamlines=False,
                                                                           normalize_target=False)
    else:
        batch_scheduler = batch_schedulers.TractographyBatchScheduler(dataset,
                                                                      batch_size=1000,
                                                                      noisy_streamlines_sigma=None,
                                                                      use_data_augment=False,  # Otherwise it doubles the number of losses :-/
                                                                      seed=1234,
                                                                      shuffle_streamlines=False,
                                                                      normalize_target=hyperparams['normalize'])

    loss_view = views.LossView(loss=loss, batch_scheduler=batch_scheduler)
    return loss_view.losses.view()


def make_is_outside_mask(mask, affine, threshold=0):
    """ Makes a function that checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    mask : 3D array
        3D image defining a mask. The interior of the mask is defined by voxels with value higher or equal to `threshold`.
    affine : ndarray of shape (4, 4)
        Matrix representing the affine transformation that aligns streamlines coordinates on top of `mask`.
    threshold : float
        Voxels value higher or equal to this threshold are considered as part of the interior of the mask.

    Returns
    -------
    function
    """
    def _is_outside_mask(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        outside : 1D array of shape (n_streamlines,)
            Array telling whether a streamline last coordinate is outside the mask.
        """
        last_coordinates = streamlines[:, -1, :]
        mask_values = neurotools.map_coordinates_3d_4d(mask, last_coordinates, affine=affine, order=1)
        return mask_values < threshold

    return _is_outside_mask


def make_is_too_long(max_length):
    """ Makes a function that checks which streamlines are exceedingly long.

    Parameters
    ----------
    max_length : int
        Maximum number of points a streamline can have.

    Returns
    -------
    function

    Notes
    -----
    That's what she said!
    """

    def _is_too_long(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        too_long : 1D array of shape (n_streamlines,)
            Array telling wheter a streamline is too long or not.
        """
        # np.isfinite(streamlines[:, :, 0]) > max_length
        return np.asarray([streamlines.shape[1] > max_length] * len(streamlines))

    return _is_too_long


def make_is_too_curvy(max_theta):
    """ Makes a function that checks which streamlines are exceedingly curvy.

    Parameters
    ----------
    max_theta : float
        Maximum angle, in degree, two consecutive segments can have with each other.

    Returns
    -------
    function

    Notes
    -----
    That's what she said!
    """
    max_theta = np.deg2rad(max_theta)  # Internally use radian.

    def _is_too_curvy(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        too_long : 1D array of shape (n_streamlines,)
            Array telling wheter a streamline is too long or not.
        """
        if streamlines.shape[1] < 3:
            # Not enough segments to test curvature.
            return [False] * len(streamlines)

        last_segments = streamlines[:, -1] - streamlines[:, -2]
        before_last_segments = streamlines[:, -2] - streamlines[:, -3]

        # Normalized segments.
        last_segments /= np.sqrt(np.sum(last_segments**2, axis=1, keepdims=True))
        before_last_segments /= np.sqrt(np.sum(before_last_segments**2, axis=1, keepdims=True))

        # Compute angles.
        angles = np.arccos(np.sum(last_segments * before_last_segments, axis=1))
        return angles > max_theta

    return _is_too_curvy


def make_is_stopping(stopping_criteria):
    """ Makes a function that checks which streamlines should we stop tracking.

    Parameters
    ----------
    stopping_criteria : dict
        Dictionnary containing all stopping criteria to check. The key is one the
        flags constant variable defined above. The value can be any function
        that expect `streamlines` as input and returns a boolean array indicating
        which streamlines should be stopped..
    """

    def _is_stopping(streamlines, to_check=None):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines coordinates.
        to_check : 1D array, optional
            Only check specific streamlines.
            If array of bool, check only streamlines marked as True.
            If array of int, check only streamlines at specified index.
            By default, check every coordinate.

        Returns
        -------
        undone : 1D array
            Array containing the indices of ongoing streamlines.
        done : 1D array
            Array containing the indices of streamlines that should be stopped.
        flags : 1D array
            Array containing a flag explaining why a streamline should be stopped.
        """

        if to_check is None:
            idx = np.arange(len(streamlines))
        elif isinstance(to_check, np.ndarray) and to_check.dtype == np.bool:
            assert len(to_check) == len(streamlines)
            idx = np.where(to_check)[0]
        else:
            idx = to_check

        undone = np.ones(len(idx), dtype=bool)
        flags = np.zeros(len(idx), dtype=np.uint8)
        for flag, stopping_criterion in stopping_criteria.items():
            done = stopping_criterion(streamlines[idx])
            undone[done] = False
            flags[done] |= flag

        done = np.logical_not(undone)
        return idx[np.where(undone)[0]], idx[np.where(done)[0]], flags[done]

    return _is_stopping


def track(model, dwi, seeds, step_size, is_stopping):

    """
    Parameters
    ----------
    is_stopping : function
        Tells whether a streamlines should stop being tracked.
        This function expects a list of streamlines and returns a 3-tuple:
          the indices of the streamlines that are undone,
          the indices of the streamlines that are done,
          the reasons why the streamlines should be stopped.
    """
    # Prepare some data container and reset the model.
    sequences = seeds.copy()
    if sequences.ndim == 2:
        sequences = sequences[:, None, :]

    seq_next = model.make_sequence_generator()
    states = model.get_init_states(batch_size=len(seeds))

    directions = np.zeros((len(seeds), 3), dtype=np.float32)
    undone = np.ones(len(sequences), dtype=bool)
    stopping_flags = np.zeros(len(seeds), dtype=np.uint8)

    # Tracking
    i = 0
    while len(undone) > 0:
        if (i+1) % 100 == 0:
            print("pts: {}/{}".format(i+1, is_stopping.max_nb_points))

        directions[:] = np.nan  # Help debugging
        # Get next unnormalized directions
        assert not np.any(np.isnan(sequences[undone, -1, :]))
        directions[undone], new_states = seq_next(x_t=sequences[undone, -1, :],
                                                  states=[s[undone] for s in states])

        # Update states for all undone streamlines.
        for state, new_state in zip(states, new_states):
            state[undone, :] = new_state[:, :]

        if step_size is not None:
            normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))  # Normed directions.
            directions[undone] = normalized_directions[undone] * step_size

        # Make a step
        sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)

        # Check which streamlines should be stopped.
        undone, done, flags = is_stopping(sequences, to_check=undone)
        stopping_flags[done] = flags
        i += 1

    streamlines_lengths = sequences.shape[1] - np.sum(np.isnan(sequences[..., 0]), axis=1)
    print("Max length: {}".format(streamlines_lengths.max()))
    print("Forward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(count_flags(stopping_flags, STOPPING_MASK),
                                                                                            count_flags(stopping_flags, STOPPING_CURVATURE),
                                                                                            count_flags(stopping_flags, STOPPING_LENGTH)))
    # Trim sequences to get the final streamlines.
    streamlines = [s[:nb_pts] for s, nb_pts in zip(sequences, streamlines_lengths)]

    # Compress stremalines and add metadata.
    streamlines = compress_streamlines(streamlines)
    tractogram = nib.streamlines.Tractogram(streamlines, data_per_streamline={"stopping_flags": stopping_flags})

    return tractogram

def pft_track(model, dwi, seeds, step_size, is_stopping, nb_retry=1):

    """
    Parameters
    ----------
    is_stopping : function
        Tells whether a streamlines should stop being tracked.
        This function expects a list of streamlines and returns a 3-tuple:
          the indices of the streamlines that are undone,
          the indices of the streamlines that are done,
          the reasons why the streamlines should be stopped.
    """
    # Prepare some data container and reset the model.
    print("Using PFT algorithm")

    def _track_step(sequences, states):
        # Get next unnormalized directions
        directions, new_states = seq_next(x_t=sequences[:, -1, :], states=states)

        if step_size is not None:
            normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))  # Normed directions.
            directions = normalized_directions * step_size

        # Make a step
        sequences = np.concatenate([sequences, sequences[:, [-1], :] + directions[:, None, :]], axis=1)

        # Check which streamlines should be stopped.
        undone, done, flags = is_stopping(sequences)

        return sequences, new_states, undone, done, flags


    sequences = seeds.copy()
    if sequences.ndim == 2:
        sequences = sequences[:, None, :]

    seq_next = model.make_sequence_generator()
    states = model.get_init_states(batch_size=len(seeds))
    undone = np.arange(len(sequences))
    stopping_flags = np.zeros(len(seeds), dtype=np.uint8)

    # Tracking
    i = 0
    while len(undone) > 0:
        if (i+1) % 1 == 0:
            print("pts: {}/{}".format(i+1, is_stopping.max_nb_points))

        assert not np.any(np.isnan(sequences[undone]))
        candidates = _track_step(sequences[undone], states=[s[undone] for s in states])
        new_sequences, new_states, new_undone, new_done, new_flags = candidates

        for j in range(nb_retry):
            if len(new_done) == 0:
                break

            print(".", end="")
            sys.stdout.flush()

            assert not np.any(np.isnan(sequences[undone][new_done]))
            retry_candidates = _track_step(sequences[undone][new_done], states=[s[undone][new_done] for s in states])
            retry_new_sequences, retry_new_states, retry_undone, retry_done, retry_flags = retry_candidates

            # Keep new_sequences up-to-date
            new_sequences[new_done] = retry_new_sequences

            # Keep new_states up-to-date.
            for new_state, retry_new_state in zip(new_states, retry_new_states):
                new_state[new_done, :] = retry_new_state[:, :]

            new_undone = np.concatenate([new_undone, new_done[retry_undone]])
            new_done = new_done[retry_done]
            new_flags = retry_flags

        # Update states for all undone streamlines.
        for state, new_state in zip(states, new_states):
            state[undone, :] = new_state[:, :]

        # Update sequences
        sequences = np.concatenate([sequences, (np.nan*np.ones((len(sequences), 1, 3))).astype(np.float32)], axis=1)
        sequences[undone] = new_sequences

        done = undone[new_done]
        undone = undone[new_undone]
        stopping_flags[done] = new_flags

        i += 1

    streamlines_lengths = sequences.shape[1] - np.sum(np.isnan(sequences[..., 0]), axis=1)
    print("Max length: {}".format(streamlines_lengths.max()))
    print("Forward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(count_flags(stopping_flags, STOPPING_MASK),
                                                                                            count_flags(stopping_flags, STOPPING_CURVATURE),
                                                                                            count_flags(stopping_flags, STOPPING_LENGTH)))
    # Trim sequences to get the final streamlines.
    streamlines = [s[:nb_pts] for s, nb_pts in zip(sequences, streamlines_lengths)]

    # Compress stremalines and add metadata.
    streamlines = compress_streamlines(streamlines)
    tractogram = nib.streamlines.Tractogram(streamlines, data_per_streamline={"stopping_flags": stopping_flags})

    return tractogram


def track_old(model, dwi, seeds, step_size=None, max_nb_points=1000, theta=0.78, mask=None, mask_affine=None, mask_threshold=0.05, backward_tracking_algo=0, discard_stopped_by_curvature=False):
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

    stopped_curv = np.zeros(len(sequences), dtype=bool)

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
            stopped_curv[undone] = angles[undone] > theta
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

    assert stopping_types_count['curv'] == np.sum(stopped_curv)
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
        if discard_stopped_by_curvature:
            streamlines = [s for s, discard in zip(streamlines, stopped_curv) if not discard]

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
                stopped_curv[undone] = angles[undone] > theta
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
                stopped_curv[undone] = np.logical_and(angles[undone] > theta, np.logical_not(resuming[undone]))
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

        print("Max length: {}".format(streamlines_lengths.max()))
        print("Tracking stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(len(sequences) - (stopped_curv.sum() + undone.sum()),
                                                                                            stopped_curv.sum(),
                                                                                            undone.sum()))

        if discard_stopped_by_curvature:
            streamlines = [s for s, discard in zip(streamlines, stopped_curv) if not discard]

        return streamlines


def batch_track(model, dwi, seeds, step_size=None, max_nb_points=500, theta=0.78, mask=None, mask_affine=None, mask_threshold=0.05, backward_tracking_algo=0, batch_size=None,
                discard_stopped_by_curvature=False, is_stopping=None, **kwargs):
    if batch_size is None:
        batch_size = len(seeds)

    while True:
        try:
            time.sleep(1)
            print("Trying to track {:,} streamlines at the same time.".format(batch_size))
            tractogram = None#nib.streamlines.Tractogram()

            for start in range(0, len(seeds), batch_size):
                print("{:,} / {:,}".format(start, len(seeds)))
                end = start+batch_size
                if kwargs.get('pft_nb_retry') is not None:
                    batch_tractogram = pft_track(model=model, dwi=dwi, seeds=seeds[start:end], step_size=step_size, is_stopping=is_stopping, nb_retry=kwargs['pft_nb_retry'])
                else:
                    batch_tractogram = track(model=model, dwi=dwi, seeds=seeds[start:end], step_size=step_size, is_stopping=is_stopping)

                if tractogram is None:
                    tractogram = batch_tractogram
                else:
                    tractogram += batch_tractogram

                # new_streamlines = track(model=model, dwi=dwi, seeds=seeds[start:end], step_size=step_size,
                #                         max_nb_points=max_nb_points, theta=theta,
                #                         mask=mask, mask_affine=mask_affine, mask_threshold=mask_threshold,
                #                         backward_tracking_algo=backward_tracking_algo,
                #                         discard_stopped_by_curvature=discard_stopped_by_curvature)
                # new_streamlines = compress_streamlines(new_streamlines)
                # tractogram.streamlines.extend(new_streamlines)

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
        dwi_name = args.dwi
        if dwi_name.endswith(".gz"):
            dwi_name = dwi_name[:-3]
        if dwi_name.endswith(".nii"):
            dwi_name = dwi_name[:-4]
        bvals_filename = dwi_name + ".bvals"
        bvecs_filename = dwi_name + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        if hyperparams["use_sh_coeffs"]:
            # Use 45 spherical harmonic coefficients to represent the diffusion signal.
            weights = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)
        else:
            # Resample the diffusion signal to have 100 directions.
            weights = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

        affine_rasmm2dwivox = np.linalg.inv(dwi.affine)

    with Timer("Loading model"):
        if hyperparams["model"] == "gru_regression":
            from learn2track.models import GRU_Regression
            model_class = GRU_Regression
        elif hyperparams['model'] == 'gru_mixture':
            from learn2track.models import GRU_Mixture
            model_class = GRU_Mixture
        elif hyperparams['model'] == 'gru_multistep':
            from learn2track.models import GRU_Multistep_Gaussian
            model_class = GRU_Multistep_Gaussian
        else:
            raise ValueError("Unknown model!")

        kwargs = {}
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

            # affine_maskvox2dwivox = mask_vox => rasmm space => dwi_vox
            affine_maskvox2dwivox = np.dot(affine_rasmm2dwivox, mask_nii.affine)
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

                # affine_seedsvox2dwivox = mask_vox => rasmm space => dwi_vox
                affine_seedsvox2dwivox = np.dot(affine_rasmm2dwivox, nii_seeds.affine)

                indices = np.array(np.where(nii_seeds.get_data())).T
                for idx in indices:
                    seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(args.nb_seeds_per_voxel, 3))
                    seeds_in_voxel = nib.affines.apply_affine(affine_seedsvox2dwivox, seeds_in_voxel)
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



        is_outside_mask = make_is_outside_mask(mask, affine_maskvox2dwivox, threshold=args.mask_threshold)
        is_too_long = make_is_too_long(max_nb_points)
        is_too_curvy = make_is_too_curvy(np.rad2deg(theta))
        is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                        STOPPING_LENGTH: is_too_long,
                                        STOPPING_CURVATURE: is_too_curvy})

        is_stopping.max_nb_points = max_nb_points  # Small hack

        tractogram = batch_track(model, weights, seeds,
                                 step_size=step_size,
                                 is_stopping=is_stopping,
                                 batch_size=args.batch_size,
                                 pft_nb_retry=args.pft_nb_retry)

        # tractogram = batch_track(model, weights, seeds,
        #                          step_size=step_size,
        #                          max_nb_points=max_nb_points,
        #                          theta=theta,
        #                          mask=mask, mask_affine=affine_maskvox2dwivox,
        #                          mask_threshold=args.mask_threshold,
        #                          backward_tracking_algo=args.backward_tracking_algo,
        #                          batch_size=args.batch_size,
        #                          discard_stopped_by_curvature=args.discard_stopped_by_curvature)

    nb_streamlines = len(tractogram)
    print("Generated {:,} (compressed) streamlines".format(nb_streamlines))
    with Timer("Cleaning streamlines", newline=True):
        # Flush streamlines that has no points.
        tractogram = tractogram[np.array(list(map(len, tractogram))) > 0]
        print("Removed {:,} empty streamlines".format(nb_streamlines - len(tractogram)))

        # Remove small streamlines
        nb_streamlines = len(tractogram)
        lengths = dipy.tracking.streamline.length(tractogram.streamlines)
        tractogram = tractogram[lengths >= args.min_length]
        lengths = lengths[lengths >= args.min_length]
        if len(lengths) > 0:
            print("Average length: {:.2f} mm.".format(lengths.mean()))
            print("Minimum length: {:.2f} mm. Maximum length: {:.2f}".format(lengths.min(), lengths.max()))
        print("Removed {:,} streamlines smaller than {:.2f} mm".format(nb_streamlines - len(tractogram),
                                                                       args.min_length))
        if args.discard_stopped_by_curvature:
            nb_streamlines = len(tractogram)
            stopping_curvature_flag_is_set = is_flag_set(tractogram.data_per_streamline['stopping_flags'], STOPPING_CURVATURE)
            tractogram = tractogram[stopping_curvature_flag_is_set]
            print("Removed {:,} streamlines stopped for having a curvature higher than {:.2f} degree".format(nb_streamlines - len(tractogram),
                                                                                                             np.rad2deg(theta)))

        if args.filter_threshold is not None:
            # Remove streamlines that produces a reconstruction error higher than a certain threshold.
            nb_streamlines = len(tractogram)
            losses = compute_loss_errors(tractogram.streamlines, model, hyperparams)
            print("Mean loss: {:.4f} ± {:.4f}".format(np.mean(losses), np.std(losses, ddof=1) / np.sqrt(len(losses))))
            tractogram = tractogram[losses <= args.filter_threshold]
            print("Removed {:,} streamlines producing a loss lower than {:.2f} mm".format(nb_streamlines - len(tractogram),
                                                                                          args.filter_threshold))

    with Timer("Saving {:,} (compressed) streamlines".format(len(tractogram))):
        # Streamlines have been generated in the voxel space so the add the
        # affine of the dwi so we send them in rasmm.
        tractogram.affine_to_rasmm = dwi.affine
        save_path = pjoin(experiment_path, args.out)
        try:  # Create dirs, if needed.
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

        nib.streamlines.save(tractogram, save_path)

if __name__ == "__main__":
    main()

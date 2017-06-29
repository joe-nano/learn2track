#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import textwrap

import nibabel as nib
import numpy as np
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.core.gradients import gradient_table
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking.streamline import set_number_of_points

from learn2track.neurotools import TractographyData
from learn2track.utils import Timer


def build_argparser():
    DESCRIPTION = textwrap.dedent(
        """ Script to generate training data from a list of streamlines bundle files.

            This results in a .npz file containing the following keys:\n"
            'coords': ndarray of shape (N, 3)
                Coordinates of each point of every streamlines expressed in voxel space.
                N is the total number points of all streamlines.
            'offsets': ndarray of shape (M,) with dtype int64
                Index of the beginning of each streamline. M is the total number of streamlines.
            'lengths': ndarray of shape (M,) with dtype int16
                Number of points of each streamline. M is the total number of streamlines.
            'bundle_ids': ndarray of shape (M,) with dtype int16
                ID of the bundle each streamline belongs to
            'name2id': dict
                Mapping between bundle names and bundle IDs.
            'signal': :class:`Nifti1Image` object (from nibabel)
                Diffusion signal
            'gradients': :class:`GradientTable` object (from dipy)
                Diffusion gradients information
        """)
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('signal', help='Diffusion signal (.nii|.nii.gz).')
    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of streamlines bundle files.')
    p.add_argument('--bvals', help='File containing diffusion gradient lengths (Default: guess it from `signal`).')
    p.add_argument('--bvecs', help='File containing diffusion gradient directions (Default: guess it from `signal`).')
    p.add_argument('--out', metavar='FILE', default="dataset.npz", help='output filename (.npz). Default: dataset.npz')
    p.add_argument('--dtype', type=str, default="float32", help="'float16' or 'float32'. Default: 'float32'")

    subparsers = p.add_subparsers(title="Subcommands", help="'subsample-streamlines' downsamples every bundle using QuickBundles")
    subsampler = subparsers.add_parser('subsample-streamlines')

    subsampler.add_argument('--min-distance', type=int, default=2,
                            help="Minimal distance for 2 streamlines to be considered different (in mm). Default: 2")
    subsampler.add_argument('--clustering-threshold', type=int, default=6, help="Clustering threshold for QB. Default: 6")

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')

    return p


def remove_similar_streamlines(streamlines, threshold=5):
    # Simple trick to make it faster than using 40-60 points
    sample_10_streamlines = set_number_of_points(streamlines, 10)
    distance_matrix = distance_matrix_mdf(sample_10_streamlines, sample_10_streamlines)

    current_id = 0
    while True:
        indices = np.where(distance_matrix[current_id] < threshold)[0]

        it = 0
        if len(indices) > 1:
            for k in indices:
                # Every streamlines similar to yourself (excluding yourself)
                # should be deleted from the set of desired streamlines
                if not current_id == k:
                    streamlines.pop(k - it)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=0)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=1)
                    it += 1

        current_id += 1
        # Once you reach the end of the remaining streamlines
        if current_id >= len(streamlines):
            break

    return streamlines


def subsample_wrapper(streamlines, cluster_id, threshold, verbose):
    subsample_streamlines = remove_similar_streamlines(streamlines, threshold=threshold)
    if verbose:
        print("cluster #{}: {} / {}".format(cluster_id, len(streamlines), len(subsample_streamlines)))
    return subsample_streamlines


def main():
    parser = build_argparser()
    args = parser.parse_args()

    signal = nib.load(args.signal)
    signal.get_data()  # Forces loading volume in-memory.
    basename = re.sub('(\.gz|\.nii.gz)$', '', args.signal)
    bvals = basename + '.bvals' if args.bvals is None else args.bvals
    bvecs = basename + '.bvecs' if args.bvecs is None else args.bvecs

    gradients = gradient_table(bvals, bvecs)
    tracto_data = TractographyData(signal, gradients)

    # Compute matrix that brings streamlines back to diffusion voxel space.
    rasmm2vox_affine = np.linalg.inv(signal.affine)

    # Retrieve data.
    with Timer("Retrieving data", newline=args.verbose):
        for filename in sorted(args.bundles):
            if args.verbose:
                print("{}".format(filename))

            # Load streamlines
            tfile = nib.streamlines.load(filename)
            if args.subsample_streamlines:
                original_streamlines = tfile.streamlines
                output_streamlines = []

                qb = QuickBundles(original_streamlines, dist_thr=args.clustering_threshold, pts=20)
                for i in range(len(qb.centroids)):
                    temp_streamlines = qb.label2tracks(original_streamlines, i)
                    output_streamlines.extend(subsample_wrapper(temp_streamlines, i, args.min_distance, args.verbose))

                print("Total difference: {} / {}".format(len(original_streamlines), len(output_streamlines)))
                new_tractogram = nib.streamlines.Tractogram(output_streamlines,
                                                            affine_to_rasmm=tfile.tractogram.affine_to_rasmm)
                tfile.tractogram = new_tractogram

            tfile.tractogram.apply_affine(rasmm2vox_affine)

            # Add streamlines to the TractogramData
            bundle_name = os.path.splitext(os.path.basename(filename))[0]
            tracto_data.add(tfile.streamlines, bundle_name)

    if args.verbose:
        diff = tracto_data.streamlines._data - tracto_data.streamlines._data.astype(args.dtype)
        precision_error = np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))
        avg_precision_error = precision_error / len(tracto_data.streamlines._data)
        print("Precision error: {} (avg. {})".format(precision_error, avg_precision_error))

    # Save streamlines coordinates using either float16 or float32.
    tracto_data.streamlines._data = tracto_data.streamlines._data.astype(args.dtype)

    # Save dataset
    tracto_data.save(args.out)


if __name__ == '__main__':
    main()

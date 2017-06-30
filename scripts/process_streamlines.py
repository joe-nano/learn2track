#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import textwrap

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table

from learn2track.neurotools import TractographyData, subsample_streamlines
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
                output_streamlines = subsample_streamlines(original_streamlines, args.clustering_threshold,
                                                           args.min_distance)

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

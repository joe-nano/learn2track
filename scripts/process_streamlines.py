#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
import textwrap

import nibabel as nib
from learn2track.utils import Timer


def build_argparser():
    DESCRIPTION = textwrap.dedent(
        """ Script to generate training data from a list of streamlines bundle files.

            This results in a .npz file containing the following keys:\n"
            'coords': ndarray of shape (N, 3)
                Coordinates of each point of every streamlines. N is the total number points of all streamlines.
            'offsets': ndarray of shape (M,) with dtype int64
                Index of the beginning of each streamline. M is the total number of streamlines.
            'lengths': ndarray of shape (M,) with dtype int16
                Number of points of each streamline. M is the total number of streamlines.
            'bundle_ids': ndarray of shape (M,) with dtype int16
                ID of the bundle each streamline belongs to
            'bundle_names': ndarray of shape (B,) with dtype str
                Name of each bundle in order of their bundle ID. B is the number of bundles.
        """)
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of ground truth bundle files.')
    p.add_argument('--ref', metavar='FILE',
                   help='If provided, streamlines will be transformed to fit in the voxel grid defined by this reference anatomy.')
    p.add_argument('--out', metavar='FILE', default="dataset.npz", help='output filename (.npz). Default: dataset.npz')
    p.add_argument('--dtype', type=str, default="float32", help="'float16' or 'float32'. Default: 'float32'")

    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')
    # p.add_argument('-f', '--force', action='store_true', help='overwrite existing file.')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    precision_error = 0
    streamlines = nib.streamlines.ArraySequence()
    bundle_ids = np.zeros((0,), dtype=np.int16)
    bundle_names = []

    affine = None
    if args.ref:
        affine = nib.load(args.ref).affine  # vox->rasmm
        affine = np.linalg.inv(affine)  # rasmm->vox

    # Retrieve data.
    with Timer("Retrieving data", newline=args.verbose):
        for i, filename in enumerate(sorted(args.bundles)):
            if args.verbose:
                print("{}".format(filename))

            tfile = nib.streamlines.load(filename)
            if affine is not None:
                tfile.tractogram.apply_affine(affine)

            # Concatenate the points of the streamlines.
            streamlines.extend(tfile.streamlines)

            # Keep the bundle ID for every streamline.
            size = len(bundle_ids)
            new_size = size + len(tfile.streamlines)
            bundle_ids.resize((new_size,))
            bundle_ids[size:new_size] = i

            # Keep the bundle name associated to every bundle ID.
            bundle_name = os.path.splitext(os.path.basename(filename))[0]
            bundle_names.append(bundle_name)

    if args.verbose:
        diff = streamlines._data - streamlines._data.astype(args.dtype)
        precision_error = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
        print("Precision error: {} (avg. {})".format(precision_error, precision_error/len(streamlines._data)))

    # Save dataset
    np.savez(args.out,
             coords=streamlines._data.astype(args.dtype),
             offsets=streamlines._offsets,
             lengths=streamlines._lengths.astype(np.int16),
             bundle_ids=bundle_ids,
             bundle_names=bundle_names)

if __name__ == '__main__':
    main()

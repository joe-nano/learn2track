#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse

import nibabel as nib
from nibabel.streamlines import CompactList
from nibabel.streamlines.compact_list import save_compact_list, load_compact_list

from learn2track.utils import save_bundle
from learn2track.utils import map_coordinates_3d_4d
from dipy.tracking.streamline import set_number_of_points


NB_POINTS = 100


def buildArgsParser():
    DESCRIPTION = ("Script to generate training data from a list of streamlines bundle files."
                   " Each streamline is added twice: once flip (i.e. the points are reverse) and once as-is."
                   " Since we cannot predict the next direction for the last point we don't use it for training."
                   " The same thing goes for the last point in the reversed order.")
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('dwi', type=str, help='Nifti file containing the original HCP dwi used in the ISMRM2015 challenge.')
    p.add_argument('--bvals', type=str, help='text file containing the bvalues used in the dwi. By default, uses "subject1.bvals" in the same folder as the dwi file.')
    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of ground truth bundle files.')
    p.add_argument('--out', metavar='DIR', type=str, help='output folder where to put generated training data. Default: along the bundles')
    p.add_argument('--nb-points', type=int,
                   help="if specified, force all streamlines to have the same number of points. This is achieved using linear interpolation along the curve."
                        "Default: number of points per streamline is not modified.")

    p.add_argument('--sanity-check', action='store_true', help='perform sanity check on the data and quit.')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')

    return p


def generate_training_data(tractogram, dwi, bvals):
    """
    Generates the training data given a list of streamlines, a dwi and
    its associated bvals.

    The training data consist of a list of $N$ sequences of $M_i$ inputs $x_ij$ and $M_i$
    targets $y_ij$, where $N$ is the number of streamlines and $M_i$ is the number of
    3D points of streamline $i$. The input $x_ij$ is a vector (32 dimensions) corresponding
    to the dwi data that have been trilinearly interpolated at the coordinate of the $j$-th
    point of the $i$-th streamline. The target $y_ij$ corresponds the (x, y, z) direction
    leading from the 3D point $j$ to the 3D point $j+1$ of streamline $i$.

    Parameters
    ----------
    tractogram : `nib.streamlines.Tractogram` object
        contains the points coordinates of N streamlines
    dwi : `nib.NiftiImage` object
        diffusion weighted image
    bvals : list of int
        represents the b-value used for each direction

    Returns
    -------
    inputs : nib.streamlines.CompactList` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.CompactList` object
        the direction leading from any 3D point to the next in every streamline.
    """
    # Indices of bvals sorted
    sorted_bvals_idx = np.argsort(bvals)
    nb_b0s = int(np.sum(bvals == 0))
    dwi_weights = dwi.get_data().astype("float32")

    inputs = []
    targets = []

    # Get the first b0.
    b0 = dwi_weights[..., [sorted_bvals_idx[0]]]

    # Keep only b-value greater than 0
    weights = dwi_weights[..., sorted_bvals_idx[nb_b0s:]]

    # Make sure in every voxels weights are lower than ones from the b0. (should not happen)
    # weights_normed = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.isnan(weights_normed)] = 0.

    for i, streamline in enumerate(tractogram.streamlines):
        # Get diffusion weights for every points along the streamlines (the inputs).
        # The affine is provided in order to bring the streamlines back to voxel space.
        weights_interpolated = map_coordinates_3d_4d(weights_normed, streamline, affine=dwi.affine)

        # We don't need the last point though (nothing to predict from it).
        inputs.append(weights_interpolated[:-1])
        # Also add the flip version of the streamline (except its last point).
        inputs.append(weights_interpolated[::-1][:-1])

        # Get streamlines directions (the targets)
        directions = streamline[1:, :] - streamline[:-1, :]
        targets.append(directions)
        targets.append(-directions)  # Flip directions

    return CompactList(inputs), CompactList(targets)


def do_sanity_check_on_data(streamlines, dwi):
    min_pts = np.inf*np.ones(3)
    max_pts = -np.inf*np.ones(3)
    for streamline_points in streamlines.points:
        min_pts = np.minimum(min_pts, streamline_points.min(axis=0))
        max_pts = np.maximum(max_pts, streamline_points.max(axis=0))

    if np.any(min_pts < np.zeros(3)-0.5) or np.any(max_pts >= np.array(dwi.shape[:3])+0.5):
        print ("Streamlines are not in voxel space!")
        print ("min_pts", min_pts)
        print ("max_pts", max_pts)
        return False

    return True


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print (args)

    dwi = nib.load(args.dwi)

    # Load and parse bvals
    bvals_filename = args.bvals
    if bvals_filename is None:
        bvals_filename = args.dwi[:-len('.nii')] + ".bvals"
        if args.dwi.endswith('.nii.gz'):
            bvals_filename = args.dwi[:-len('.nii.gz')] + ".bvals"

    bvals = list(map(float, open(bvals_filename).read().split()))
    bvals = np.round(bvals).astype(int)

    assert dwi.shape[-1] == len(bvals)

    if args.sanity_check:
        for no, bundle in enumerate(args.bundles):
            streamlines = nib.streamlines.load(bundle, ref=dwi)
            print("Checking {0} ({1:,} streamlines)...".format(bundle, len(streamlines)), end=" ")
            is_ok = do_sanity_check_on_data(streamlines, dwi)
            print("OK" if is_ok else "FAIL")
    else:
        for bundle in args.bundles:
            if args.verbose:
                print("Processing {0}...".format(bundle))

            tractogram = nib.streamlines.load(bundle).tractogram  # Load streamlines in RASmm.

            if args.nb_points is not None:
                tractogram.streamlines = set_number_of_points(tractogram.streamlines, nb_points=args.nb_points)

            inputs, targets = generate_training_data(tractogram, dwi, bvals)

            # Dump data as numpy array
            if args.out is None:  # Put training data along the streamlines.
                path = os.path.splitext(bundle)[0] + ".npz"
            else:
                filename = os.path.splitext(os.path.basename(bundle))[0]
                path = os.path.join(args.out, filename + ".npz")

            save_bundle(path, inputs, targets)


if __name__ == '__main__':
    main()

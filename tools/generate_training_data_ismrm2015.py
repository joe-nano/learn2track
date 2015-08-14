#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse

import nibabel as nib

from utils import map_coordinates_3d_4d
from dipy.tracking.streamline import set_number_of_points


NB_POINTS = 100


def buildArgsParser():
    DESCRIPTION = "Script to generate training data from a list of streamlines bundle files."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('dwi', type=str, help='Nifti file containing the original HCP dwi used in the ISMRM2015 challenge.')
    p.add_argument('--bvals', type=str, help='text file containing the bvalues used in the dwi. By default, uses "subject1.bvals" in the same folder as the dwi file.')
    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of ground truth bundle files.')
    #p.add_argument('--out', metavar='DIR', type=str, help='output folder where to put generated training data. Default: along the bundles')

    p.add_argument('--sanity-check', action='store_true', help='perform sanity check on the data and quit.')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')

    return p


def generate_training_data(streamlines, dwi, bvals):
    """
    Generate the training data given a list of streamlines, a dwi and
    its associated bvals.

    Parameters
    ----------
    streamlines : `nib.Streamlines` object
        contains the points coordinates of N streamlines
    dwi : `nib.NiftiImage` object
        diffusion weighted image
    bvals : list of int
        represents the b-value used for each direction

    Returns
    -------
    inputs: list of ndarrays (N, 271)
        if 'original_hcp/dwi.nii.gz' is used:
          `inputs` contains 271 diffusion weights (90 directions with b1000,
          90 directions with b2000, 90 directions with b3000
          and 1 b0 [out of the 18 b0s available])
        if 'ground_truth/dwi.nii.gz' is used:
          `inputs`  contains 33 diffusion weights (32 directions with b1000,
          and 1 b0)
    targets: list of ndarrays (N, 3)
        contains directions
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
    #weights_normed = np.minimum(weights, b0)
    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.isnan(weights_normed)] = 0.

    for i, streamline_points in enumerate(set_number_of_points(streamlines.points, nb_points=NB_POINTS)):
        # Get diffusion weights for every points along the streamlines (the inputs)
        weights_interpolated = map_coordinates_3d_4d(weights_normed, streamline_points)
        inputs.append(weights_interpolated)

        # Get streamlines directions (the targets)
        directions = streamline_points[1:, :] - streamline_points[:-1, :]
        targets.append(directions)

    return inputs, targets


def do_sanity_check_on_data(streamlines, dwi):
    min_pts = np.inf*np.ones(3)
    max_pts = -np.inf*np.ones(3)
    for streamline_points in streamlines.points:
        min_pts = np.minimum(min_pts, streamline_points.min(axis=0))
        max_pts = np.maximum(max_pts, streamline_points.max(axis=0))

    if np.any(min_pts < np.zeros(3)-0.5) or np.any(max_pts >= np.array(dwi.shape[:3])+0.5):
        print "Streamlines are not in voxel space!"
        print "min_pts", min_pts
        print "max_pts", max_pts
        return False

    return True


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print args

    dwi = nib.load(args.dwi)

    # Load and parse bvals
    bvals_filename = args.bvals
    if bvals_filename is None:
        bvals_filename = args.dwi.split('.')[0] + ".bvals"

    bvals = map(float, open(bvals_filename).read().split())
    bvals = np.round(bvals).astype(int)

    assert dwi.shape[-1] == len(bvals)

    if args.sanity_check:
        for no, bundle in enumerate(args.bundles):
            streamlines = nib.streamlines.load(bundle, ref=dwi)
            print "Checking {0}...".format(bundle)
            is_ok = do_sanity_check_on_data(streamlines, dwi)
            print "OK" if is_ok else "FAIL"
    else:
        for bundle in args.bundles:
            if args.verbose:
                print "Processing {0}...".format(bundle)
            streamlines = nib.streamlines.load(bundle, ref=dwi)

            inputs, targets = generate_training_data(streamlines, dwi, bvals)

            # Dump data as numpy array
            filename = os.path.splitext(bundle)[0]
            np.savez(filename + ".npz", inputs=inputs, targets=targets)


if __name__ == '__main__':
    main()

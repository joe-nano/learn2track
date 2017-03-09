#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import textwrap

import nibabel as nib
from nibabel.streamlines.tractogram import Tractogram

from learn2track.utils import Timer

import numpy as np


def build_argparser():
    DESCRIPTION = textwrap.dedent(
        """ Script to filter out of bounds streamlines from the volume.
            Outputs two files, one for the filtered streamlines and another for the removed streamlines.
        """)
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('signal', help='Diffusion signal (.nii|.nii.gz).')
    p.add_argument('filename', type=str, help='streamlines file (.tck)')
    p.add_argument('--out_prefix', help='Output filenames prefix. If not given, will use streamlines `filename` as prefix.')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    signal = nib.load(args.signal)
    data = signal.get_data()

    # Compute matrix that brings streamlines back to diffusion voxel space.
    rasmm2vox_affine = np.linalg.inv(signal.affine)

    # Retrieve data.
    with Timer("Retrieving data"):
        print("Loading {}".format(args.filename))

        # Load streamlines (already in RASmm space)
        tfile = nib.streamlines.load(args.filename)
        tfile.tractogram.apply_affine(rasmm2vox_affine)

        # tfile.tractogram.apply_affine(rasmm2vox_affine)
        tractogram = Tractogram(streamlines=tfile.streamlines, affine_to_rasmm=signal.affine)

    with Timer("Filtering streamlines"):

        # Get volume bounds
        x_max = data.shape[0]
        y_max = data.shape[1]
        z_max = data.shape[2]

        mask = np.ones((len(tractogram),)).astype(bool)

        for i,s in enumerate(tractogram.streamlines):

            # Identify streamlines out of bounds
            oob_test = np.logical_or.reduce((s[:, 0] < 0., s[:, 0] > x_max,  # Out of bounds on axis X
                                             s[:, 1] < 0., s[:, 1] > y_max,  # Out of bounds on axis Y
                                             s[:, 2] < 0., s[:, 2] > z_max))  # Out of bounds on axis Z

            if np.any(oob_test):
                mask[i] = False

        tractogram_filtered = tractogram[mask]
        tractogram_removed = tractogram[np.logical_not(mask)]

        print("Kept {} streamlines and removed {} streamlines".format(len(tractogram_filtered), len(tractogram_removed)))

    with Timer("Saving filtered and removed streamlines"):
        base_filename = args.out_prefix
        if args.out_prefix is None:
            base_filename = args.filename[:-4]

        tractogram_filtered_filename = "{}_filtered.tck".format(base_filename)
        tractogram_removed_filename = "{}_removed.tck".format(base_filename)

        # Save streamlines
        nib.streamlines.save(tractogram_filtered, tractogram_filtered_filename)
        nib.streamlines.save(tractogram_removed, tractogram_removed_filename)

if __name__ == '__main__':
    main()

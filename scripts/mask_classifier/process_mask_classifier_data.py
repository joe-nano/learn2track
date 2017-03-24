#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import textwrap

import nibabel as nib
import numpy as np
import os
import re
from dipy.core.gradients import gradient_table

from learn2track.neurotools import MaskClassifierData


def build_argparser():
    DESCRIPTION = textwrap.dedent(
        """ Script to generate training data from diffusion data and binary mask.

            This results in a .npz file containing the following keys:\n"
            'mask': :class: numpy.ndarray
                3D binary mask
            'signal': :class:`Nifti1Image` object (from nibabel)
                Diffusion signal
            'gradients': :class:`GradientTable` object (from dipy)
                Diffusion gradients information
        """)
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('signal', help='Diffusion signal (.nii|.nii.gz).')
    p.add_argument('mask', help='Binary mask (.nii|.nii.gz).')
    p.add_argument('--bvals', help='File containing diffusion gradient lengths (Default: guess it from `signal`).')
    p.add_argument('--bvecs', help='File containing diffusion gradient directions (Default: guess it from `signal`).')
    p.add_argument('--out', metavar='FILE', default="dataset.npz", help='output filename (.npz). Default: dataset.npz')

    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')
    p.add_argument('-f', '--force', action='store_true', help='overwrite existing file.')

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

    mask = nib.load(args.mask)
    mask.get_data()

    positive_coords = np.argwhere(mask.get_data() > 0)
    negative_coords = np.argwhere(mask.get_data() <= 0)

    mask_classifier_data = MaskClassifierData(signal, gradients, mask, positive_coords, negative_coords)

    # Save dataset
    if os.path.isfile(args.out) and not args.force:
        print("File already exists! Use -f|--force to overwrite it.")
        return
    mask_classifier_data.save(args.out)

if __name__ == '__main__':
    main()

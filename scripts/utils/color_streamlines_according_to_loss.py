#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# # Hack so you don't have to put the library containing this script in the PYTHONPATH.
# sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import nibabel as nib
from nibabel.streamlines import ArraySequence

from learn2track.utils import Timer


def build_parser():
    DESCRIPTION = ("Color streamlines according to loss.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('tractogram', help='tractogram (TRK) to score (must contained data_per_streamline["loss"]).')
    p.add_argument('--out', help='output filename (TRK).')
    p.add_argument('--normalization', help='log or norm.', default="norm")

    p.add_argument('-f', '--force', action='store_true', help='force.')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    with Timer("Loading streamlines"):
        trk = nib.streamlines.load(args.tractogram)
        losses = trk.tractogram.data_per_streamline['loss']
        del trk.tractogram.data_per_streamline['loss']  # Not supported in MI-Brain for my version.

    with Timer("Coloring streamlines"):
        viridis = plt.get_cmap('viridis')

        vmin = None
        vmax = None

        if args.normalization == "norm":
            cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        elif args.normalization == "norm":
            cNorm  = colors.LogNorm()
        else:
            raise ValueError("Unkown normalization: {}".format(args.normalization))

        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=viridis)
        print(scalarMap.get_clim())
        streamlines_colors = scalarMap.to_rgba(-losses[:, 0], bytes=True)[:, :-1]

        # from dipy.viz import fvtk
        # streamlines_colors = fvtk.create_colormap(-losses[:, 0]) * 255
        colors_per_point = ArraySequence([np.tile(c, (len(s), 1)) for s, c in zip(trk.tractogram.streamlines, streamlines_colors)])
        trk.tractogram.data_per_point['color'] = colors_per_point

    with Timer("Saving streamlines"):
        if args.out is None:
           args.out = args.tractogram[:-4] + "_color" + args.tractogram[-4:]

        nib.streamlines.save(trk, args.out)


if __name__ == "__main__":
    main()

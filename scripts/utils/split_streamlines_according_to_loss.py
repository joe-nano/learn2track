#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from dipy.workflows.base import IntrospectiveArgumentParser

import pylab as plt
import argparse
import numpy as np
import nibabel as nib
from dipy.viz import actor, window, widget
from dipy.viz import fvtk
from dipy.tracking.streamline import set_number_of_points


def build_argparser():
    description = "Split streamlines according to their loss."
    p = argparse.ArgumentParser(description=description)

    p.add_argument("tractogram", help="Tractogram file (.TRK)")
    p.add_argument("--intervals", nargs="+", type=float, required=True, help="Interval(s) where to split.")

    p.add_argument("--stats-only", action="store_true", help="Only print stats then exit.")


    return p



def create_hist(outlierness, colormap_name="jet"):
    # Make histogram plot prettier.
    plt.rcParams['font.size'] = 18
    #plt.rcParams['font.weight'] = "bold"
    # fig = plt.figure(figsize=(16, 4), dpi=300)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    #ax.set_title('Outlierness', fontsize="32", fontweight="bold")
    #ax.set_ylabel('Outlierness', fontsize="24")
    #n, bins, patches = ax.hist(outlierness, bins=np.linspace(0, 1, 101), linewidth=0.2, orientation="horizontal")
    n, bins, patches = ax.hist(outlierness, bins=np.linspace(0, np.max(outlierness), 101), linewidth=0)

    # Apply colormap to histogram.
    cm = plt.cm.get_cmap(colormap_name)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return fig


def main():
    parser = build_argparser()
    args = parser.parse_args()

    tractogram = nib.streamlines.load(args.tractogram).tractogram
    losses = tractogram.data_per_streamline['loss'][:, 0]
    nb_streamlines = len(tractogram)

    print("Nb. streamlines: {:,}".format(nb_streamlines))
    print("Stats on losses: min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(losses.min(), losses.max(), losses.mean(), losses.std()))
    fig = create_hist(losses)
    plt.show()

    if args.stats_only:
        sys.exit()

    filename = args.tractogram[:-4]
    counts = []
    remaining = tractogram
    for split in sorted(args.intervals):
        losses = remaining.data_per_streamline['loss'][:, 0]
        tractogram = remaining[losses <= split].copy()
        remaining = remaining[losses > split].copy()
        counts.append(len(tractogram))

        nib.streamlines.save(tractogram, filename + "_{}.tck".format(split))

    if len(remaining) > 0:
        counts.append(len(remaining))
        nib.streamlines.save(remaining, filename + "_remaining.tck")

    print("Counts: {} vs. {}".format(nb_streamlines, np.sum(counts)))


if __name__ == "__main__":
    main()

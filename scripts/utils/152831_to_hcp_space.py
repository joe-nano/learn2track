import os
import pickle
import argparse
import numpy as np

import nibabel as nib
from nibabel.streamlines import Field


def build_argparser():
    DESCRIPTION = "Convert TRK tractograms (Laurent 152831 space) -> (Standard HCP space)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('box', help='cropping box used in Jasmeen data (.pkl).')
    p.add_argument('tractograms', metavar="tractogram", nargs="+", help='tractograms to convert (.trk).')
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output files.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    try:
        box = pickle.load(open(args.box))
        corner = box[0]
    except:
        parser.error("Could not load cropping box: {}".format(args.box))

    for tractogram in args.tractograms:
        if nib.streamlines.detect_format(tractogram) is not nib.streamlines.TrkFile:
            print("Skipping non TRK file: '{}'".format(tractogram))
            continue

        output_filename = tractogram[:-4] + '.hcp.trk'
        if os.path.isfile(output_filename) and not args.force:
            print("Skipping existing file: '{}'. Use -f to overwrite.".format(output_filename))
            continue

        trk = nib.streamlines.load(tractogram)
        hdr = trk.header
        affine = hdr[Field.VOXEL_TO_RASMM].copy()
        sizes = hdr[Field.VOXEL_SIZES]

        translation = np.eye(4)
        translation[:-1, -1] = -np.array(corner) * sizes

        # Translate streamlines.
        trk.tractogram.apply_affine(translation)
        trk.tractogram.affine_to_rasmm = np.eye(4)

        # Replace translation.
        hdr[Field.VOXEL_TO_RASMM][:-1, -1] = affine[:-1, -1] - corner * sizes

        nib.streamlines.save(trk.tractogram, output_filename, header=hdr)

if __name__ == '__main__':
    main()


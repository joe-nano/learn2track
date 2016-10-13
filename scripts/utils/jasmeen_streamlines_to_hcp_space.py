#!/usr/bin/env python

import pickle
import argparse
import numpy as np

import nibabel as nib


def build_argparser():
    DESCRIPTION = "Send Jasmeen's tractograms to HCP native space. File will be generated with the '.hcp.trk' extension."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('box', help='pickle file containing cropping information. (Ask Maxime Descoteaux)')
    p.add_argument('ref', help='HCP anatomical image used to get header information (.nii|.nii.gz)')
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms (.trk).')
    p.add_argument('--debug', action="store_true", help='if specified, also save streamlines in TCK format.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    box_min, box_max = pickle.load(open(args.box))
    ref = nib.load(args.ref)

    for tractogram in args.tractograms:
        trk = nib.streamlines.load(tractogram)
        trk.header['dimensions'] = ref.shape

        translation = np.eye(4)
        translation[:3, 3] = np.array(box_min) * np.array(ref.header.get_zooms())
        trk.tractogram.apply_affine(translation)
        trk.tractogram.affine_to_rasmm = np.eye(4)

        nib.streamlines.save(trk, tractogram[:-4] + '.hcp.trk')

        if args.debug:
            # Useful for debugging in MI-Brain
            trk = nib.streamlines.load(tractogram[:-4] + '.hcp.trk')
            nib.streamlines.save(trk.tractogram, tractogram[:-4] + '.hcp.tck')

if __name__ == '__main__':
    main()


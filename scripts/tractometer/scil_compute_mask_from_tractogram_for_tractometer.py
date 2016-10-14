import os
import argparse
from os.path import join as pjoin
from subprocess import check_call

import scipy.ndimage
import nibabel as nib


def build_argparser():
    DESCRIPTION = "Create binary mask from a tractogram."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('ref', help='reference anatomy (.nii|.nii.gz).')
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms.')
    p.add_argument('--out', help='destination folder. Default: along the tractogram.')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Create output directory if needed.
    try:
        os.makedirs(args.out)
    except:
        pass

    for tractogram in args.tractograms:
        mask_filename = tractogram[:-4] + ".nii.gz"
        if args.out is not None:
            filename = os.path.basename(tractogram)
            mask_filename = pjoin(args.out, filename[:-4] + ".nii.gz")

        check_call(["scil_compute_density_map_from_streamlines.py", tractogram, args.ref, mask_filename, "--binary", "-f"])

        nii = nib.load(mask_filename)
        mask = nii.get_data()
        mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)
        nib.save(nib.Nifti1Image(mask, nii.affine), mask_filename)

if __name__ == '__main__':
    main()


import argparse
import os

import nibabel as nib
import numpy as np
from nibabel.streamlines import Field
from nibabel.streamlines.tck import TckFile
from nibabel.streamlines.trk import TrkFile


def build_argparser():
    DESCRIPTION = "Merge tractograms (TCK/TRK)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms (.trk | .tck).')
    p.add_argument('-o', '--output', type=str, help="Output filename. Default: tractogram(.tck|.trk)")
    p.add_argument('-f', '--force', action="store_true", help='overwrite existing output file.')
    return p


# List of all supported formats
FORMATS = {TrkFile: ".trk",
           TckFile: ".tck"
           }


def get_filetype(tractograms):
    file_type = None
    for tractogram in tractograms:
        if file_type is None:
            file_type = nib.streamlines.detect_format(tractogram)
        else:
            assert nib.streamlines.detect_format(tractogram) is file_type, "All files must be of the same type!"
    return file_type


def validate_and_merge_trks(filenames):
    header = None
    all_tractograms = None
    for filename in filenames:

        tractogram = nib.streamlines.load(filename)

        if header is None:
            header = tractogram.header
            all_tractograms = tractogram.tractogram
        else:
            all_matching = True
            all_matching = all_matching and np.allclose(header[Field.VOXEL_TO_RASMM], tractogram.header[Field.VOXEL_TO_RASMM])
            all_matching = all_matching and np.allclose(header[Field.VOXEL_SIZES], tractogram.header[Field.VOXEL_SIZES])
            all_matching = all_matching and np.array_equal(header[Field.DIMENSIONS], tractogram.header[Field.DIMENSIONS])
            all_matching = all_matching and np.array_equal(header[Field.VOXEL_ORDER], tractogram.header[Field.VOXEL_ORDER])

            assert all_matching, "Headers do not match, aborting!"
            all_tractograms += tractogram.tractogram
    return all_tractograms, header


def validate_and_merge_tcks(filenames):
    all_tractograms = None
    for filename in filenames:
        tractogram = nib.streamlines.load(filename)

        if all_tractograms is None:
            all_tractograms = tractogram
        else:
            assert np.allclose(tractogram.affine_to_rasmm, all_tractograms.affine_to_rasmm), "Affines should be the same!"
            all_tractograms += tractogram
    return all_tractograms


def main():
    parser = build_argparser()
    args = parser.parse_args()

    file_type = get_filetype(args.tractograms)
    assert file_type in FORMATS.keys(), "Usupported filetype: {}; Suported filetypes: {}".format(file_type, FORMATS.values())

    if args.output:
        output_filename = args.output
        assert nib.streamlines.detect_format(output_filename) is file_type, "Output file must be of the same type!"
    else:
        output_filename = "tractogram" + FORMATS[file_type]

    if os.path.isfile(output_filename) and not args.force:
        print("Output file already exists: '{}'. Use -f to overwrite.".format(output_filename))
        return

    if file_type is TrkFile:
        all_tractograms, header = validate_and_merge_trks(args.tractograms)
        nib.streamlines.save(all_tractograms, output_filename, header=header)
    else:
        all_tractograms, header = validate_and_merge_tcks(args.tractograms)
        nib.streamlines.save(all_tractograms, output_filename, header=header)


if __name__ == '__main__':
    main()

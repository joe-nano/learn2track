import argparse
import os

import nibabel as nib
from nibabel.streamlines.tck import TckFile
from nibabel.streamlines.trk import TrkFile

from learn2track.neurotools import TractographyData


def build_argparser():
    DESCRIPTION = "Merge tractograms (TCK/TRK)."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('tractograms', metavar='tractogram', nargs="+", help='list of tractograms (.trk | .tck | .npz).')
    return p


class NpzFile:
    pass


# List of all supported formats
FORMATS = {TrkFile: ".trk",
           TckFile: ".tck",
           NpzFile: ".npz"
           }


def get_filetype(tractogram):
    """ Get the filetype of a single tractogram file """

    # Try nibabel formats
    file_type = nib.streamlines.detect_format(tractogram)

    # Try custom formats
    if file_type is None:
        if os.path.splitext(tractogram)[1] == '.npz':
            file_type = NpzFile

    return file_type


def count_nibabel_streamlines(filename):
    tractogram = nib.streamlines.load(filename)
    return len(tractogram.streamlines)


def count_npz_streamlines(filename):
    tracto_data = TractographyData.load(filename)
    return len(tracto_data.streamlines)


def main():
    parser = build_argparser()
    args = parser.parse_args()

    streamline_count = 0

    for filename in args.tractograms:
        file_type = get_filetype(filename)
        assert file_type in FORMATS.keys(), "Usupported filetype: {}; Suported filetypes: {}".format(file_type, FORMATS.values())

        if file_type in [TckFile, TrkFile]:
            streamline_count += count_nibabel_streamlines(filename)
        elif file_type == NpzFile:
            streamline_count += count_npz_streamlines(filename)

    print("Total # of streamlines: {}".format(streamline_count))


if __name__ == '__main__':
    main()

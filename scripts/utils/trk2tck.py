import argparse
import nibabel as nib


def build_argparser():
    DESCRIPTION = "Convert tractograms (TRK -> TCK)."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('tractograms', metavar='bundle', nargs="+", help='list of tractograms.')
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    for tractogram in args.tractograms:
        trk = nib.streamlines.load(tractogram)
        nib.streamlines.save(trk.tractogram, tractogram[:-4] + '.tck')

if __name__ == '__main__':
    main()


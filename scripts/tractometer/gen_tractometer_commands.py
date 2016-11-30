import argparse
from os.path import join as pjoin


def buildArgsParser():
    p = argparse.ArgumentParser()
    p.add_argument('files', type=str, nargs='+', help='path of tractogram file.')
    p.add_argument('--save-ib', action="store_true", help='add option to save segmented IBs.')
    return p

if __name__ == "__main__":
    parser = buildArgsParser()
    args = parser.parse_args()

    template = "python ~/research/src/tractometer/scripts/score_ismrm.py {0} ismrm15_challenge/scoring_data/ {1}/attributes.json ~/research/src/tractometer/metadata/ismrm_challenge_2015/gt_bundles_attributes.json {1}/ 5 -v --save_tracts --save_vb"

    if args.save_ib:
        template += " --save_ib"

    for f in args.files:
        folder = pjoin(*(f.split("/")[:-2]))
        print(template.format(f, folder))

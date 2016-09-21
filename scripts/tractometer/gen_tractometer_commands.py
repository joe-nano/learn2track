import argparse


def buildArgsParser():
    p = argparse.ArgumentParser()
    p.add_argument('files', type=str, nargs='+', help='path of tractogram file.')
    return p

if __name__ == "__main__":
    parser = buildArgsParser()
    args = parser.parse_args()

    template = "~/env/miccai2016_tractometer/bin/python ~/research/src/tractometer/scripts/score_ismrm.py {0} ismrm15_challenge/scoring_data/ experiments/{1}/tractometer/attributes.json ~/research/src/tractometer/metadata/ismrm_challenge_2015/gt_bundles_attributes.json experiments/{1}/tractometer/ 5 -v --save_tracts --save_vb"

    for f in args.files:
        folder = f.split("/")[1]
        print(template.format(f, folder))

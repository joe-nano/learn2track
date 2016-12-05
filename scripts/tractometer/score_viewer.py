#!/usr/bin/env python

from __future__ import division

from pdb import set_trace as dbg

import re
import os
import argparse
import pickle
from texttable import Texttable


def sort_nicely(col):
    """ Sort the given list in the way that humans expect.
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key[col])]
    return alphanum_key

###############
# Script part #
###############
DESCRIPTION = """
Score viewer for the Tractometer.

Metrics:
--------

Average Bundles overlap (OL): proportion of the voxels within the volume
of a ground truth bundle that is traversed by at least one valid streamline
associated with the bundle. This value shows how well the tractography
result recovers the original volume of the bundle.

Average Bundles overreach (OR): fraction of voxels outside the volume of a
ground truth bundle that is traversed by at least one valid streamline
associated with the bundle over the total number of voxels within the ground
truth bundle. This value shows how much the valid connections extend beyond
the ground truth bundle volume.
(See http://www.tractometer.org/ismrm_2015_challenge/evaluation)

Average Bundles overreach (ORn): fraction of voxels outside the volume of a
ground truth bundle that is traversed by at least one valid streamline
associated with the bundle over the total number of voxels within the ground
truth bundle. This value shows how much the valid connections extend beyond
the ground truth bundle volume.
(See http://www.tractometer.org/ismrm_2015_challenge/evaluation)

Average F1-Score (F1): f1-score (a.k.a. dice coefficient) between voxels that
are traversed by at least one valid streamline and the voxels within the volume
of a ground truth bundle.
Basically, it is  $(2 * precision * recall) / (precision + recall)$
where $precision=(1-OR)$ and $recall=OL$.

Invalid Bundles (IB): number of unexpected pairs of ROIs connected by at least
one streamline,

Valid Bundles (VB): number of expected pairs of ROIs connected by at least one
streamline,

No Connections (NC): fraction of streamlines not connecting any pairs of ROIs,

Valid Connections Wrong Path (VCWP): fraction of streamlines connecting
expected pairs of ROIs, but exiting corresponding bundle mask,

Invalid Connections (IC): fraction of streamlines connecting unexpected pairs
of ROIs,

Valid Connections (VC): fraction of streamlines connecting expected pairs of
ROIs,

Valid Connection to Connection Ratio (VCCR): VC/(VCWP+IC+VC).

File Naming Convention:
-----------------
The parameter-value naming convention is: an underscore symbol (_) to
separate pairs of parameters and values, and a dash symbol (-) to separate each
parameter of its value. E.g 'param1-value1_param2_value2.pkl' or
'description_param1-value1_param2_value2.pkl'.
"""

METRICS = ['OL', 'OR', 'ORn', 'F1', 'IB', 'VB', 'NC', 'VCWP', 'IC', 'VC', 'VCCR']


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('--scores', dest='scores', action='store', type=str,
                   metavar='FILES', nargs='+', default=['scores/'],
                   help="Scoring file(s) or folder containing scoring files." +
                   " %(default)s")
    p.add_argument('--sort', dest='sort_by', action='store', metavar='IDs',
                   type=int, nargs='+', default=[], help="Sort rows by " +
                   "column's ID in ascending (+ID) or descending (-ID) " +
                   "order. %(default)s")
    p.add_argument('--split', dest='is_split', action='store_true',
                   help="Use the naming convention to " +
                   "identify tracking parameters (see description).")
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    scoring_files = args.scores
    sort_by = args.sort_by

    if len(scoring_files) == 1 and os.path.isdir(scoring_files[0]):
        scoring_files = [os.path.join(scoring_files[0], f)
                         for f in os.listdir(scoring_files[0])]

    scoring_files = filter(lambda f: f.endswith('.json'), scoring_files)

    if len(scoring_files) == 0:
        parser.error('Need a least one scoring file/folder!')

    if len([no for no in sort_by if no == 0]) > 0:
        parser.error('Column ID are starting at 1.')

    headers_params = set()
    headers_scores = set()
    params = []
    scores = []

    # Retrieves scores
    for scoring_file in scoring_files:
        infos = scoring_file.split('/')[-1][:-4]
        if args.is_split:
            ind = 0 if infos.find('-') < infos.find('_') else 1
            param = dict([tuple(param.split('-'))
                          for param in infos.split('_')[ind:]])
        else:
            param = {"filename": infos}
        # score = pickle.load(open(scoring_file))
        score = json.load(open(scoring_file))

        # Compute the VCCR metric, CSR=1-NC
        # In VCCR, VCWP are considered as IC [Girard et al., NeuroImage, 2014]
        if score['VC'] > 0:
            score['VCCR'] = score['VC'] / (score['VC'] +
                                           score['IC'] +
                                           score['VCWP'])
        else:
            score['VCCR'] = 0

        # Keep only scalar metrics.
        for k in score.keys():
            if k not in METRICS:
                del score[k]

        headers_params |= set(param.keys())
        headers_scores |= set(score.keys())

        scores.append(score)
        params.append(param)

    nbr_cols = len(headers_params) + len(headers_scores)
    if len([no for no in sort_by if abs(no) > nbr_cols]) > 0:
        parser.error('The maximum column ID is {0}.'.format(nbr_cols))

    table = Texttable(max_width=0)
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['a'] * nbr_cols)
    table.set_cols_align(['c'] * nbr_cols)

    # Headers
    headers_params = sorted(headers_params)
    headers_scores = list(headers_scores)
    headers_scores = [headers_scores[headers_scores.index(e)] for e in METRICS if e in headers_scores]

    headers = headers_params + headers_scores
    table.header([str(i) + "\n" + h for i, h in enumerate(headers, start=1)])

    # Data
    for param, score in zip(params, scores):
        data = []
        for header in headers_params:
            data.append(param.get(header, '-'))

        for header in headers_scores:
            data.append(score.get(header, '-'))

        table.add_row(data)

    # Sort
    for col in reversed(sort_by):
        table._rows = sorted(table._rows,
                             key=sort_nicely(abs(col) - 1),
                             reverse=col < 0)

    print(table.draw())


if __name__ == "__main__":
    main()

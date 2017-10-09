#!/usr/bin/env python

from __future__ import division

import argparse
import json
import logging
import os
import pickle

import learn2track_metrics as metrics
import numpy as np
import tractometer.pipeline_helper as helper
from tractometer.pipeline_helper import mkdir
from tractometer.utils.attribute_computer import get_attribs_for_file, \
    load_attribs

###############
# Script part #
###############
DESCRIPTION = 'Scoring script for learn2track journal paper'


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram', action='store',
                   metavar='TRACTS', type=str, help='Tractogram file')

    p.add_argument('base_dir', action='store',
                   metavar='BASE_DIR', type=str,
                   help='base directory for scoring data')

    p.add_argument('metadata_file', action='store',
                   metavar='SUBMISSIONS_ATTRIBUTES', type=str,
                   help='attributes file of the submissions. ' +
                        'Needs to contain the orientation.\n' +
                        'Normally, use metadata/ismrm_challenge_2015/' +
                        'anon_submissions_attributes.json.\n' +
                        'Can be computed with ' +
                        'ismrm_compute_submissions_attributes.py.')

    p.add_argument('basic_bundles_attribs', action='store',
                   metavar='GT_ATTRIBUTES', type=str,
                   help='attributes of the basic bundles. ' +
                        'Same format as SUBMISSIONS_ATTRIBUTES')

    p.add_argument('out_dir', action='store',
                   metavar='OUT_DIR', type=str,
                   help='directory where to send score files')

    # Only version 5 is supported for now

    p.add_argument('version', action='store',
                   metavar='ALGO_VERSION', choices=range(1, 6),
                   type=int,
                   help='version of the algorithm to use.\n' +
                        'choices:\n  1: VC: auto_extract -> VCWP candidates ' +
                        '-> IC -> remove from VCWP -> rest = NC\n' +
                        '  2: Extract NC from whole, then do as 1. (NOT IMPLEMENTED)\n' +
                        '  3: Classical pipeline, no auto_extract\n' +
                        '  4: Do as 1, but assign ICs to as many IB as they can.\n' +
                        '  5: VC: auto_extract -> IC: length threshold -> QB -> ' +
                        'singleton removal -> nearest regions classification.')

    p.add_argument('--save_tracts', action='store_true',
                   help='save the segmented streamlines')
    p.add_argument('--save_ib', action='store_true',
                   help='save IB independently.')
    p.add_argument('--save_vb', action='store_true',
                   help='save VB independently.')
    p.add_argument('--save_vcwp', action='store_true',
                   help='save VCWP independently.')

    # Other
    p.add_argument('-f', dest='is_forcing', action='store_true',
                   required=False, help='overwrite output files')
    p.add_argument('-v', dest='is_verbose', action='store_true',
                   required=False, help='produce verbose output')

    return p


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}

        return json.JSONEncoder(self, obj)


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': '), cls=NumpyEncoder))


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    tractogram = args.tractogram
    base_dir = args.base_dir
    attribs_file = args.metadata_file
    out_dir = args.out_dir

    isForcing = args.is_forcing
    isVerbose = args.is_verbose

    if isVerbose:
        helper.VERBOSE = True
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(tractogram):
        parser.error('"{0}" must be a file!'.format(tractogram))

    if not os.path.isdir(base_dir):
        parser.error('"{0}" must be a directory!'.format(base_dir))

    if not os.path.isfile(attribs_file):
        parser.error('"{0}" must be a file!'.format(attribs_file))

    if not os.path.isfile(args.basic_bundles_attribs):
        parser.error('"{0}" is not a file!'.format(args.basic_bundles_attribs))

    if out_dir is not None:
        out_dir = mkdir(out_dir + "/").replace("//", "/")

    # Launch main
    masks_dir = base_dir + "/masks/"
    bundles_dir = os.path.join(base_dir, "bundles")
    scores_dir = mkdir(out_dir + "/scores/")
    scores_filename = scores_dir + tractogram.split('/')[-1][:-4] + ".pkl"

    if os.path.isfile(scores_filename):
        if isForcing:
            os.remove(scores_filename)
        else:
            print("Skipping... {0}".format(scores_filename))
            return

    if not args.save_tracts and (args.save_ib or args.save_vb or args.save_vcwp):
        parser.error("Cannot save IBs, VBs, or VCWP if save_tracts is not set.")

    if args.version != 5:
        parser.error("Algorithm version {} is not currently implemented.".format(args.version))

    tracts_attribs = get_attribs_for_file(attribs_file, os.path.basename(tractogram))
    basic_bundles_attribs = load_attribs(args.basic_bundles_attribs)

    if not args.save_tracts:
        scores = metrics.score_from_files(tractogram, masks_dir, bundles_dir,
                                          tracts_attribs, basic_bundles_attribs)
    else:
        segments_dir = mkdir(out_dir + "/segmented/")
        base_name = os.path.splitext(os.path.basename(tractogram))[0]
        scores = metrics.score_from_files(tractogram, masks_dir, bundles_dir,
                                          tracts_attribs, basic_bundles_attribs,
                                          True, args.save_ib, args.save_vb,
                                          args.save_vcwp,
                                          segments_dir, base_name, isVerbose)

    if scores is not None:
        pickle.dump(scores, open(scores_filename, 'wb'))
        save_dict_to_json_file(scores_filename[:-4] + '.json', scores)

    if isVerbose:
        print(scores)


if __name__ == "__main__":
    main()

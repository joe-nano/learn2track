#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import argparse
import numpy as np

import nibabel as nib

from learn2track.utils import Timer, StreamlinesData


def buildArgsParser():
    DESCRIPTION = "Script to split a dataset while making sure the split is similar every bundle."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('dataset', help='training data (.npz).')
    p.add_argument('--split', type=float, nargs=3, help='respectively the sizes of the split for trainset, validset and testset.', default=[0.8, 0.1, 0.1])
    p.add_argument('--split_type', type=str, choices=["percentage", "count"], help='type of the split, either use percents or fixed counts.', default="percentage")
    p.add_argument('--seed', type=int, help='seed to use to shuffle data.', default=1234)
    p.add_argument('--delete', action="store_true", help='delete bundle file after being splitted.')
    p.add_argument('--leave-one-out', metavar="BUNDLE_ID", nargs="+", type=str,
                   help='list of bundle_id (use comma to merge multiple into one), from which a leave-one-out split will be generated (i.e. remove one bundle from the trainset and generate a testset and a validset from it.')

    p.add_argument('--list-bundles-name', action="store_true",
                   help='if specified, list the bundles name and quit.')

    p.add_argument('-v', '--verbose', action="store_true", help='verbose mode')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    data = StreamlinesData.load(args.dataset)
    streamlines = data.streamlines

    if args.list_bundles_name:
        for i, name in enumerate(data.bundle_names):
            print("{}: {}".format(i, name))

        return

    if args.leave_one_out is not None:
        with Timer("Splitting {} using a leave-one-out strategy".format(args.dataset), newline=True):
            for bundle in args.leave_one_out:
                rng = np.random.RandomState(args.seed)
                train_data = StreamlinesData(data.bundle_names)
                valid_data = StreamlinesData(data.bundle_names)
                test_data = StreamlinesData(data.bundle_names)

                bundle_ids_to_exclude = list(map(int, bundle.split(',')))
                missing_bundles_name = [data.bundle_names[i] for i in bundle_ids_to_exclude]

                if args.verbose:
                    print("Leaving out {}...".format(", ".join(missing_bundles_name)))

                include = np.ones(len(data.bundle_ids), dtype=bool)
                exclude = np.zeros(len(data.bundle_ids), dtype=bool)
                for i in bundle_ids_to_exclude:
                    include = np.logical_and(include, data.bundle_ids != i)
                    exclude = np.logical_or(exclude, data.bundle_ids == i)

                include_idx = np.where(include)[0]
                exclude_idx = np.where(exclude)[0]
                rng.shuffle(include_idx)
                rng.shuffle(exclude_idx)

                trainset_indices = include_idx
                validset_indices = exclude_idx[:len(exclude_idx)//2]
                testset_indices = exclude_idx[len(exclude_idx)//2:]

                train_data.add(streamlines[trainset_indices], data.bundle_ids[trainset_indices])
                valid_data.add(streamlines[validset_indices], data.bundle_ids[validset_indices])
                test_data.add(streamlines[testset_indices], data.bundle_ids[testset_indices])

                filename = "missing_{}.npz".format("_".join(missing_bundles_name))
                with Timer("Saving dataset: {}".format(filename[:-4])):
                    train_data.save(filename[:-4] + "_trainset.npz")
                    valid_data.save(filename[:-4] + "_validset.npz")
                    test_data.save(filename[:-4] + "_testset.npz")

    else:
        rng = np.random.RandomState(args.seed)
        train_data = StreamlinesData(data.bundle_names)
        valid_data = StreamlinesData(data.bundle_names)
        test_data = StreamlinesData(data.bundle_names)

        with Timer("Splitting {} as follow {} using {}".format(args.dataset, args.split, args.split_type), newline=args.verbose):
            for i, name in enumerate(data.bundle_names):
                if args.verbose:
                    print("Splitting bundle {}...".format(name))

                indices = np.where(data.bundle_ids == i)[0]
                nb_examples = len(indices)
                rng.shuffle(indices)

                if args.split_type == "percentage":
                    trainset_size = int(np.round(args.split[0] * nb_examples))
                    validset_size = int(np.round(args.split[1] * nb_examples))
                    testset_size = int(np.round(args.split[2] * nb_examples))
                    # Make sure the splits sum to nb_examples
                    testset_size += nb_examples - (trainset_size + validset_size + testset_size)
                elif args.split_type == "count":
                    raise NotImplementedError("Split type `count` not implemented yet!")

                assert trainset_size + validset_size + testset_size == nb_examples

                trainset_indices = indices[:trainset_size]
                validset_indices = indices[trainset_size:-testset_size]
                testset_indices = indices[-testset_size:]

                train_data.add(streamlines[trainset_indices], data.bundle_ids[trainset_indices])
                valid_data.add(streamlines[validset_indices], data.bundle_ids[validset_indices])
                test_data.add(streamlines[testset_indices], data.bundle_ids[testset_indices])

        with Timer("Saving"):
            train_data.save(args.dataset[:-4] + "_trainset.npz")
            valid_data.save(args.dataset[:-4] + "_validset.npz")
            test_data.save(args.dataset[:-4] + "_testset.npz")

        if args.delete:
            os.remove(args.dataset)


if __name__ == '__main__':
    main()

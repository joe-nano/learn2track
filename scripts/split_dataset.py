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

    p.add_argument('-v', '--verbose', action="store_true", help='verbose mode')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    data = StreamlinesData.load(args.dataset)
    streamlines = data.streamlines
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import os

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import argparse
import numpy as np

from learn2track.utils import Timer
from learn2track.neurotools import MaskClassifierData


def buildArgsParser():
    DESCRIPTION = """Script to split a dataset while making sure the split is similar for positive and negative examples.

        Examples
        --------
        split_dataset.py subject1.npz --split 0.8 0.1 0.1 --seed 1234 --delete
    """
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('dataset', help='training data (.npz).')
    p.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help='respectively the sizes of the split for trainset, validset and testset. Default: %(default)s')
    p.add_argument('--seed', type=int, default=1234, help='seed to use to shuffle data. Default: %(default)s')
    p.add_argument('--delete', action="store_true", help='if specified, delete input file after being splitted.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    data = MaskClassifierData.load(args.dataset)
    positive_coords = data.positive_coords
    negative_coords = data.negative_coords

    rng = np.random.RandomState(args.seed)

    with Timer("Splitting {} using split: {}".format(args.dataset, args.split)):
        nb_positive_examples = positive_coords.shape[0]
        nb_negative_examples = negative_coords.shape[0]

        positive_indices = np.arange(nb_positive_examples)
        negative_indices = np.arange(nb_negative_examples)

        rng.shuffle(positive_indices)
        rng.shuffle(negative_indices)

        train_positive_size = int(np.round(args.split[0] * nb_positive_examples))
        train_negative_size = int(np.round(args.split[0] * nb_negative_examples))

        valid_positive_size = int(np.round(args.split[1] * nb_positive_examples))
        valid_negative_size = int(np.round(args.split[1] * nb_negative_examples))

        test_positive_size = int(np.round(args.split[2] * nb_positive_examples))
        test_negative_size = int(np.round(args.split[2] * nb_negative_examples))

        # Make sure the splits sum to nb_examples
        test_positive_size += nb_positive_examples - (train_positive_size + valid_positive_size + test_positive_size)
        test_negative_size += nb_negative_examples - (train_negative_size + valid_negative_size + test_negative_size)

        assert train_positive_size + valid_positive_size + test_positive_size == nb_positive_examples
        assert train_negative_size + valid_negative_size + test_negative_size == nb_negative_examples

        train_positive_indices = positive_indices[:train_positive_size]
        valid_positive_indices = positive_indices[train_positive_size:train_positive_size+valid_positive_size]
        test_positive_indices = positive_indices[train_positive_size+valid_positive_size:]

        train_negative_indices = negative_indices[:train_negative_size]
        valid_negative_indices = negative_indices[train_negative_size:train_negative_size+valid_negative_size]
        test_negative_indices = negative_indices[train_negative_size+valid_negative_size:]

        train_data = MaskClassifierData(data.signal, data.gradients, data.mask, positive_coords[train_positive_indices],
                                        negative_coords[train_negative_indices])
        valid_data = MaskClassifierData(data.signal, data.gradients, data.mask, positive_coords[valid_positive_indices],
                                        negative_coords[valid_negative_indices])
        test_data = MaskClassifierData(data.signal, data.gradients, data.mask, positive_coords[test_positive_indices],
                                       negative_coords[test_negative_indices])

    with Timer("Saving"):
        train_data.save(args.dataset[:-4] + "_trainset.npz")
        valid_data.save(args.dataset[:-4] + "_validset.npz")
        test_data.save(args.dataset[:-4] + "_testset.npz")

    if args.delete:
        os.remove(args.dataset)


if __name__ == '__main__':
    main()

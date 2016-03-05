#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import argparse
import numpy as np

import nibabel as nib

from learn2track.utils import Timer


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


class Dataset(object):
    def __init__(self, bundle_names):
        self.streamlines = nib.streamlines.ArraySequence()
        self.bundle_ids = np.zeros((0,), dtype=np.int16)
        self.bundle_names = bundle_names

    def add(self, streamlines, bundle_ids):
        self.streamlines.extend(streamlines)
        size = len(self.bundle_ids)
        new_size = size + len(bundle_ids)
        self.bundle_ids.resize((new_size,))
        self.bundle_ids[size:new_size] = bundle_ids

    def save(self, filename):
        np.savez(filename,
                 coords=self.streamlines._data.astype(np.float32),
                 offsets=self.streamlines._offsets,
                 lengths=self.streamlines._lengths.astype(np.int16),
                 bundle_ids=self.bundle_ids,
                 bundle_names=self.bundle_names)


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    dataset = np.load(args.dataset)
    streamlines = nib.streamlines.ArraySequence()
    streamlines._data = dataset['coords']
    streamlines._offsets = dataset['offsets']
    streamlines._lengths = dataset['lengths']

    trainset = Dataset(dataset['bundle_names'])
    validset = Dataset(dataset['bundle_names'])
    testset = Dataset(dataset['bundle_names'])

    with Timer("Splitting {} as follow {} using {}".format(args.dataset, args.split, args.split_type), newline=args.verbose):
        for i, name in enumerate(dataset['bundle_names']):
            if args.verbose:
                print("Splitting bundle {}...".format(name))

            indices = np.where(dataset['bundle_ids'] == i)[0]
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

            trainset.add(streamlines[trainset_indices], dataset['bundle_ids'][trainset_indices])
            validset.add(streamlines[validset_indices], dataset['bundle_ids'][validset_indices])
            testset.add(streamlines[testset_indices], dataset['bundle_ids'][testset_indices])

    with Timer("Saving"):
        trainset.save(args.dataset[:-4] + "_trainset.npz")
        validset.save(args.dataset[:-4] + "_validset.npz")
        testset.save(args.dataset[:-4] + "_testset.npz")

    if args.delete:
        os.remove(args.dataset)


if __name__ == '__main__':
    main()

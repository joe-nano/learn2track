#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import argparse


def buildArgsParser():
    DESCRIPTION = "Script to generate training data from a list of streamlines bundle files."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of training data files (.npz).')
    p.add_argument('--split', type=float, nargs=3, help='respectively the sizes of the split for trainset, validset and testset.', default=[0.8, 0.1, 0.1])
    p.add_argument('--split_type', type=str, choices=["percentage", "count"], help='type of the split, either use percents or fixed counts.', default="percentage")
    p.add_argument('--seed', type=int, help='seed to use to shuffle data.', default=1234)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    rng = np.random.RandomState(args.seed)

    for bundle in args.bundles:
        print("Splitting {} as follow {} using {}".format(bundle, args.split, args.split_type))
        data = np.load(bundle)
        inputs = data["inputs"]
        targets = data["targets"]

        nb_examples = len(inputs)
        indices = np.arange(nb_examples)
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
        assert len(inputs[:trainset_size]) == trainset_size
        assert len(inputs[trainset_size:-testset_size]) == validset_size
        assert len(inputs[-testset_size:]) == testset_size
        assert len(targets[:trainset_size]) == trainset_size
        assert len(targets[trainset_size:-testset_size]) == validset_size
        assert len(targets[-testset_size:]) == testset_size

        np.savez(bundle[:-4] + "_trainset.npz",
                 inputs=inputs[:trainset_size],
                 targets=targets[:trainset_size])
        np.savez(bundle[:-4] + "_validset.npz",
                 inputs=inputs[trainset_size:-testset_size],
                 targets=targets[trainset_size:-testset_size])
        np.savez(bundle[:-4] + "_testset.npz",
                 inputs=inputs[-testset_size:],
                 targets=targets[-testset_size:])


if __name__ == '__main__':
    main()

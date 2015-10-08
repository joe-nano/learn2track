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

    p.add_argument('--pad', action="store_true", help='make every sequence examples the same length.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    rng = np.random.RandomState(args.seed)

    # Find the longest streamlines in term of number of points.
    if args.pad:
        max_nb_points = -np.inf
        for bundle in args.bundles:
            data = np.load(bundle)
            inputs = data["inputs"]
            max_nb_points = max(max_nb_points, max(map(len, inputs)))

    for bundle in args.bundles:
        print("Splitting {} as follow {} using {}".format(bundle, args.split, args.split_type))
        data = np.load(bundle)

        if not args.pad:
            inputs = data["inputs"]
            targets = data["targets"]
        else:
            # Pad each streamline with NaN as needed.
            nb_examples = len(data["inputs"])
            input_dim = data["inputs"][0].shape[1]
            target_dim = data["targets"][0].shape[1]
            inputs = np.nan * np.ones((nb_examples, max_nb_points, input_dim), dtype="float32")
            targets = np.nan * np.ones((nb_examples, max_nb_points, target_dim), dtype="float32")

            for i, (x, y) in enumerate(zip(data['inputs'], data['targets'])):
                inputs[i, :len(x)] = x[:]
                targets[i, :len(y)] = y[:]

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

        trainset_indices = indices[:trainset_size]
        validset_indices = indices[trainset_size:-testset_size]
        testset_indices = indices[-testset_size:]

        np.savez(bundle[:-4] + "_trainset.npz",
                 inputs=inputs[trainset_indices],
                 targets=targets[trainset_indices])
        np.savez(bundle[:-4] + "_validset.npz",
                 inputs=inputs[validset_indices],
                 targets=targets[validset_indices])
        np.savez(bundle[:-4] + "_testset.npz",
                 inputs=inputs[testset_indices],
                 targets=targets[testset_indices])


if __name__ == '__main__':
    main()

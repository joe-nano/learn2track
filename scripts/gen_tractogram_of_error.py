#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse
import itertools
import theano
import nibabel as nib
from nibabel.streamlines import ArraySequence

from smartlearner import utils as smartutils

from learn2track.utils import Timer, load_ismrm2015_challenge_contiguous, log_variables

from learn2track.losses import L2DistanceForSequences
from learn2track.batch_schedulers import SequenceBatchScheduler


def buildArgsParser():
    DESCRIPTION = ("Script to eval a LSTM model on a dataset"
                   " (ismrm2015_challenge) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    # General options (optional)
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dataset', type=str, help='folder containing training data (.npz files).')
    p.add_argument('--append-previous-direction', action="store_true",
                   help="if specified, the target direction of the last timestep will be concatenated to the input of the current timestep. (0,0,0) will be used for the first timestep.")

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def generate_tractogram_of_error(model, dataset, append_previous_direction=False):
    loss = L2DistanceForSequences(model, dataset)
    loss.losses  # Hack to generate update dict in loss :(
    batch_scheduler = SequenceBatchScheduler(dataset, batch_size=50, append_previous_direction=append_previous_direction)

    predict, losses, targets, masks = log_variables(batch_scheduler, model.regression_out, loss.L2_error_per_item, dataset.symb_targets*1, dataset.symb_mask*1)

    timesteps_loss = ArraySequence([l[:int(m.sum())] for l, m in zip(losses, masks)])
    timesteps_prediction = ArraySequence([p[:int(m.sum())] for p, m in zip(predict, masks)])
    timesteps_targets = ArraySequence([t[:int(m.sum())] for t, m in zip(targets, masks)])

    streamlines = []
    colors = []
    for d, p, l in zip(timesteps_targets, timesteps_prediction, timesteps_loss):
        d = np.r_[[(0, 0, 0)], d]
        pts = np.cumsum(d, axis=0)
        pts -= pts.mean(0)

        streamline = np.zeros(((len(pts)-1)*3+1, 3))
        streamline[::3] = pts
        streamline[1:-1:3] = pts[:-1] + p
        streamline[2:-1:3] = pts[:-1]
        streamlines.append(streamline)

        color = np.zeros_like(streamline)
        color[:] = (0, 0, 255.)
        color[1:-1:3, 0] = l/2.*255.
        color[1:-1:3, 2] = (1-l/2.)*255.
        colors.append(color)

    tractogram = nib.streamlines.Tractogram(streamlines, data_per_point={"colors": colors})
    return tractogram


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    # Load experiments hyperparameters
    try:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
    except FileNotFoundError:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "..", "hyperparams.json"))

    with Timer("Loading dataset"):
        trainset, validset, testset = load_ismrm2015_challenge_contiguous(args.dataset, hyperparams['classification'])
        print("Datasets:", len(trainset), len(validset), len(testset))

    with Timer("Loading model"):
        if hyperparams["classification"]:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Softmax
                model_class = LSTM_Softmax
            elif hyperparams["model"] == "lstm_hybrid":
                from learn2track.lstm import LSTM_Hybrid
                model_class = LSTM_Hybrid

        else:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Regression
                model_class = LSTM_Regression
            elif hyperparams["model"] == "gru":
                from learn2track.gru import GRU_Regression
                model_class = GRU_Regression

                if args.append_previous_direction:
                    from learn2track.gru import GRU_RegressionWithScheduledSampling
                    model_class = GRU_RegressionWithScheduledSampling

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    tractogram_file = pjoin(experiment_path, "tractogram_{}.trk")
    for name, dataset in [("trainset", trainset), ("validset", validset), ("testset", testset)]:
        if not os.path.isfile(tractogram_file.format(name)) or args.force:
            tractogram = generate_tractogram_of_error(model, dataset, args.append_previous_direction)
            nib.streamlines.save(tractogram, tractogram_file.format(name))
        else:
            print("Tractogram already exists. (use --force to generate it again)")


if __name__ == "__main__":
    main()

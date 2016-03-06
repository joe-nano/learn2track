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

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def get_regression_results(model, dataset, append_previous_direction=False):
    loss = L2DistanceForSequences(model, dataset)
    loss.losses  # Hack to generate update dict in loss :(
    batch_scheduler = SequenceBatchScheduler(dataset, batch_size=50, append_previous_direction=append_previous_direction)
    losses, masks = log_variables(batch_scheduler, loss.L2_error_per_item, dataset.symb_mask*1)

    timesteps_loss = ArraySequence([l[:int(m.sum())] for l, m in zip(losses, masks)])
    sequences_mean_loss = np.array([l.mean() for l in timesteps_loss])

    results = {"type": "L2",
               "timesteps_loss_sum": float(timesteps_loss._data.sum()),
               "timesteps_loss_avg": float(timesteps_loss._data.mean()),
               "timesteps_loss_std": float(timesteps_loss._data.std()),
               "sequences_mean_loss_avg": float(sequences_mean_loss.mean()),
               "sequences_mean_loss_stderr": float(sequences_mean_loss.std(ddof=1)/np.sqrt(len(sequences_mean_loss)))}

    return results


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

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    with Timer("Loading dataset"):
        trainset, validset, testset = load_ismrm2015_challenge_contiguous(args.dataset, hyperparams['classification'])
        print("Datasets:", len(trainset), len(validset), len(testset))

    results_file = pjoin(experiment_path, "results.json")

    if not os.path.isfile(results_file) or args.force:
        if hyperparams['regression']:
            results = {}
            results['trainset'] = get_regression_results(model, trainset, hyperparams.get("append_previous_direction", False))
            results['validset'] = get_regression_results(model, validset, hyperparams.get("append_previous_direction", False))
            results['testset'] = get_regression_results(model, testset, hyperparams.get("append_previous_direction", False))

        smartutils.save_dict_to_json_file(results_file, results)
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = smartutils.load_dict_from_json_file(results_file)

    for dataset in ['trainset', 'validset', 'testset']:
        print("L2 error on {} (per timestep): {:.2f} ± {:.2f}".format(dataset, results[dataset]['timesteps_loss_avg'], results[dataset]['timesteps_loss_std']))
        print("L2 error on {} : {:.2f} ± {:.2f}".format(dataset, results[dataset]['sequences_mean_loss_avg'], results[dataset]['sequences_mean_loss_stderr']))


if __name__ == "__main__":
    main()

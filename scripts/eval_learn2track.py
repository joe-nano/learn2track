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
import time
from nibabel.streamlines import ArraySequence

from smartlearner import utils as smartutils

from learn2track.utils import Timer, load_ismrm2015_challenge_contiguous, log_variables

from learn2track import utils
from learn2track.losses import L2DistanceForSequences
from learn2track.batch_schedulers import SequenceBatchScheduler, StreamlinesBatchScheduler


def buildArgsParser():
    DESCRIPTION = ("Script to eval a LSTM model on a dataset"
                   " (ismrm2015_challenge) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    # General options (optional)
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--dwi',
                   help='if specified, file containing a diffusion weighted image (.nii|.nii.gz). Otherwise, information is obtained from hyperparams.json')
    p.add_argument('--dataset', type=str,
                   help='if specified, folder containing training data (.npz files). Otherwise, information is obtained from hyperparams.json.')

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def get_regression_results(model, dataset, batch_size):
    loss = L2DistanceForSequences(model, dataset)
    batch_scheduler = StreamlinesBatchScheduler(dataset, batch_size=batch_size,
                                                # patch_shape=args.neighborhood_patch,
                                                noisy_streamlines_sigma=None,
                                                nb_updates_per_epoch=None,
                                                seed=1234)

    loss.losses  # Hack to generate update dict in loss :(
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


def batch_get_regression_results(model, dataset, batch_size=None):
    if batch_size is None:
        batch_size = len(dataset)

    while True:
        try:
            time.sleep(1)
            print("Trying to evaluate {:,} streamlines at the same time.".format(batch_size))
            return get_regression_results(model, dataset, batch_size), batch_size

        except MemoryError:
            print("{:,} streamlines is too much!".format(batch_size))
            batch_size //= 2
            if batch_size < 0:
                raise MemoryError("Might needs a bigger graphic card!")

        except OSError as e:
            if "allocate memory" in e.args[0]:
                print("{:,} streamlines is too much!".format(batch_size))
                batch_size //= 2
                if batch_size < 0:
                    raise MemoryError("Might needs a bigger graphic card!")

            else:
                raise e

        except RuntimeError as e:
            if "out of memory" in e.args[0] or "allocation failed" in e.args[0]:
                print("{:,} streamlines is too much!".format(batch_size))
                batch_size //= 2
                if batch_size < 0:
                    raise MemoryError("Might needs a bigger graphic card!")

            else:
                raise e


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
        if "classification" in hyperparams and hyperparams["classification"]:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Softmax
                model_class = LSTM_Softmax
            elif hyperparams["model"] == "lstm_hybrid":
                from learn2track.lstm import LSTM_Hybrid
                model_class = LSTM_Hybrid

        elif "regression" in hyperparams and hyperparams["regression"]:
            if hyperparams["model"] == "lstm":
                from learn2track.lstm import LSTM_Regression
                model_class = LSTM_Regression
            elif hyperparams["model"] == "lstm_extraction":
                from learn2track.lstm import LSTM_RegressionWithFeaturesExtraction
                model_class = LSTM_RegressionWithFeaturesExtraction
            if hyperparams["model"] == "gru":
                from learn2track.gru import GRU_Regression
                model_class = GRU_Regression

                if args.append_previous_direction:
                    raise NameError("Not implemented")
                    from learn2track.gru import GRU_RegressionWithScheduledSampling
                    model_class = GRU_RegressionWithScheduledSampling

        else:
            from learn2track.gru import GRU_Regression
            model_class = GRU_Regression

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    with Timer("Loading dataset"):
        dataset_file = args.dataset if args.dataset is not None else hyperparams['dataset']
        dwi_file = args.dwi if args.dwi is not None else hyperparams['dwi']
        trainset, validset, testset = utils.load_streamlines_dataset(dwi_file, dataset_file)
        print("Datasets:", len(trainset), len(validset), len(testset))

    results_file = pjoin(experiment_path, "results.json")

    if not os.path.isfile(results_file) or args.force:
        results = {}

        with Timer("Evaluating trainset"):
            results['trainset'], batch_size = batch_get_regression_results(model, trainset)
        with Timer("Evaluating validset"):
            results['validset'], _ = batch_get_regression_results(model, validset, batch_size=batch_size)
        with Timer("Evaluating testset"):
            results['testset'], _ = batch_get_regression_results(model, testset, batch_size=batch_size)

        smartutils.save_dict_to_json_file(results_file, results)
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = smartutils.load_dict_from_json_file(results_file)

    for dataset in ['trainset', 'validset', 'testset']:
        print("L2 error on {} (per timestep): {:.2f} ± {:.2f}".format(dataset, results[dataset]['timesteps_loss_avg'], results[dataset]['timesteps_loss_std']))
        print("L2 error on {} : {:.2f} ± {:.2f}".format(dataset, results[dataset]['sequences_mean_loss_avg'], results[dataset]['sequences_mean_loss_stderr']))


if __name__ == "__main__":
    main()

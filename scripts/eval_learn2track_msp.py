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
from learn2track.losses import L2DistanceForSequences, MultistepMultivariateGaussianLossForSequences
from learn2track.batch_schedulers import SequenceBatchScheduler, StreamlinesBatchScheduler, MultistepSequenceBatchScheduler


def buildArgsParser():
    DESCRIPTION = ("Script to eval a GRU multistep prediction model on a dataset"
                   " (ismrm2015_challenge) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    # General options (optional)
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--dwi',
                   help='if specified, file containing a diffusion weighted image (.nii|.nii.gz). Otherwise, information is obtained from hyperparams.json')
    p.add_argument('--dataset', type=str,
                   help='if specified, folder containing training data (.npz files). Otherwise, information is obtained from hyperparams.json.')
    p.add_argument('--batch-size', type=int,
                   help='if specified, will use try this batch_size first and will reduce it if needed.')

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def get_regression_results(model, dataset, batch_size, k=10, Ms=[1, 10]):
    model.k = k

    results = []
    for M in Ms:
        model.m = M
        loss = MultistepMultivariateGaussianLossForSequences(model, dataset, nb_samples=M, target_size=3)
        batch_scheduler = MultistepSequenceBatchScheduler(dataset,
                                                          batch_size=batch_size,
                                                          k=k,
                                                          noisy_streamlines_sigma=None,
                                                          nb_updates_per_epoch=None,
                                                          seed=1234)

        loss.losses  # Hack to generate update dict in loss :(
        # k_nlls_per_seq.shape = [(batch_size, K)] * n_batches
        # masks.shape = [(batch_size, seq_len)] * n_batches
        # l2_errors.shape = [(batch_size, seq_len)] * n_batches
        k_nlls_per_seq, masks, l2_error = log_variables(batch_scheduler, model, loss.k_nlls_per_seq, dataset.symb_mask*1, loss.L2_error_per_item)

        # Concatenate all batches
        # k_nlls_per_seq : (dataset_len, K)
        k_nlls_per_seq = np.concatenate(k_nlls_per_seq, axis=0)

        masks = list(itertools.chain(*masks))
        l2_error = list(itertools.chain(*l2_error))
        timesteps_l2_error = ArraySequence([l[:int(m.sum())] for l, m in zip(l2_error, masks)])
        sequences_mean_l2_error = np.array([l.mean() for l in timesteps_l2_error])

        results += [{"M": M,
                     "nll_per_k": [
                        {"k": 1,
                         "nll_mean": np.mean(k_nlls_per_seq[:, 1-1], axis=0),
                         "nll_stderr": np.std(k_nlls_per_seq[:, 1-1], axis=0)/np.sqrt(len(k_nlls_per_seq))
                         },
                        {"k": 5,
                         "nll_mean": np.mean(k_nlls_per_seq[:, 5-1], axis=0),
                         "nll_stderr": np.std(k_nlls_per_seq[:, 5-1], axis=0)/np.sqrt(len(k_nlls_per_seq))
                         },
                        {"k": 10,
                         "nll_mean": np.mean(k_nlls_per_seq[:, 10-1], axis=0),
                         "nll_stderr": np.std(k_nlls_per_seq[:, 10-1], axis=0)/np.sqrt(len(k_nlls_per_seq))
                         }
                     ],
                     "nll_mean": np.mean(np.mean(k_nlls_per_seq, axis=1), axis=0),
                     "nll_stderr": np.std(np.mean(k_nlls_per_seq, axis=1), axis=0)/np.sqrt(len(k_nlls_per_seq))
                     }
                    ]

    results = {"NLL_per_M": results,
               "L2_error": {
                   "timesteps_loss_sum": float(timesteps_l2_error._data.sum()),
                   "timesteps_loss_avg": float(timesteps_l2_error._data.mean()),
                   "timesteps_loss_std": float(timesteps_l2_error._data.std()),
                   "sequences_mean_loss_avg": float(sequences_mean_l2_error.mean()),
                   "sequences_mean_loss_stderr": float(sequences_mean_l2_error.std(ddof=1)/np.sqrt(len(sequences_mean_l2_error)))
                   }
               }

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
            if "allocate memory" in str(e):
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

    with Timer("Loading dataset"):
        dataset_file = args.dataset if args.dataset is not None else hyperparams['dataset']
        dwi_file = args.dwi if args.dwi is not None else hyperparams['dwi']
        trainset, validset, testset = utils.load_streamlines_dataset(dwi_file, dataset_file)
        print("Datasets:", len(trainset), len(validset), len(testset))

    with Timer("Loading model"):
        from learn2track.gru import GRU_Multistep_Gaussian
        model_class = GRU_Multistep_Gaussian
        kwargs = {"dwi": trainset.volume,
                  'seed': 1234  # Temp
                  }

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path), **kwargs)  # Create new instance and restore model.
        print(str(model))

    results_file = pjoin(experiment_path, "results.json")

    if not os.path.isfile(results_file) or args.force:
        results = {}

        with Timer("Evaluating validset"):
            results['validset'], batch_size = batch_get_regression_results(model, validset, batch_size=args.batch_size)
        with Timer("Evaluating testset"):
            results['testset'], _ = batch_get_regression_results(model, testset, batch_size=batch_size)
        with Timer("Evaluating trainset"):
            results['trainset'], _ = batch_get_regression_results(model, trainset, batch_size=batch_size)

        smartutils.save_dict_to_json_file(results_file, results)
    else:
        print("Loading saved results... (use --force to re-run evaluation)")
        results = smartutils.load_dict_from_json_file(results_file)

    for dataset in ['trainset', 'validset', 'testset']:
        print("\n-= {} =-".format(dataset))
        print("L2 error (per timestep): {:.2f} ± {:.2f}".format(results[dataset]["L2_error"]['timesteps_loss_avg'], results[dataset]["L2_error"]['timesteps_loss_std']))
        print("L2 error: {:.2f} ± {:.2f}".format(results[dataset]["L2_error"]['sequences_mean_loss_avg'], results[dataset]["L2_error"]['sequences_mean_loss_stderr']))

        for results_per_M in results[dataset]["NLL_per_M"]:
            print("With ensemble of {}".format(results_per_M['M']))
            print("  NLL (avg. over all k): {:.2f} ± {:.2f}".format(results_per_M["nll_mean"], results_per_M["nll_stderr"]))
            for results_per_k in results_per_M["nll_per_k"]:
                print("  NLL (k={}): {:.2f} ± {:.2f}".format(results_per_k["k"], results_per_k["nll_mean"], results_per_k["nll_stderr"]))


if __name__ == "__main__":
    main()

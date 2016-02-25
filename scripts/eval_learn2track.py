#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import shutil
import numpy as np
from os.path import join as pjoin
import argparse
import itertools
import theano
import nibabel as nib

import pylab as plt
from time import sleep

import theano.tensor as T

from smartlearner import Trainer, tasks, Dataset
from smartlearner import tasks
from smartlearner.status import Status
from smartlearner import stopping_criteria
from smartlearner import views
from smartlearner import utils as smartutils
from smartlearner.optimizers import SGD, AdaGrad, Adam
from smartlearner.direction_modifiers import ConstantLearningRate, DirectionClipping


from learn2track import utils
from learn2track.utils import Timer, load_ismrm2015_challenge, load_ismrm2015_challenge_contiguous
from learn2track.lstm import LSTM_Regression, LSTM_RegressionWithFeaturesExtraction, LSTM_Softmax, LSTM_Hybrid
from learn2track.gru import GRU_Regression, GRU_Softmax, GRU_Hybrid
from learn2track.factories import ACTIVATION_FUNCTIONS
from learn2track.factories import WEIGHTS_INITIALIZERS, weigths_initializer_factory
from learn2track.factories import optimizer_factory
#from learn2track.view import RegressionError

from learn2track.losses import L2DistanceWithBinaryCrossEntropy, L2DistanceForSequences, NLLForSequenceOfDirections, ErrorForSequenceOfDirections
from learn2track.losses import ErrorForSequenceWithClassTarget, NLLForSequenceWithClassTarget
from learn2track.batch_schedulers import BundlesBatchScheduler, SequenceBatchScheduler
from learn2track.batch_schedulers import BundlesBatchSchedulerWithClassTarget, SequenceBatchSchedulerWithClassTarget


def buildArgsParser():
    DESCRIPTION = ("Script to eval a LSTM model on a dataset"
                   " (ismrm2015_challenge) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    # General options (optional)
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dataset', type=str, help='folder containing training data (.npz files).')

    # Task
    # task = p.add_argument_group("Task (required)")
    # task = task.add_mutually_exclusive_group(required=True)
    # task.add_argument('--regression', action="store_true", help='consider this problem as a regression task.')
    # task.add_argument('--classification', action="store_true", help='consider this problem as a classification task.')

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def log(batch_scheduler, *symb_vars):
    # Gather updates from the optimizer and the batch scheduler.
    f = theano.function([],
                        symb_vars,
                        givens=batch_scheduler.givens,
                        name="compute_loss",
                        on_unused_input='ignore')

    log = [[] for _ in range(len(symb_vars))]
    for _ in batch_scheduler:
        for i, e in enumerate(f()):
            log[i].append(e.copy())

    return [list(itertools.chain(*l)) for l in log]


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
                model_class = LSTM_Softmax
            elif hyperparams["model"] == "lstm_hybrid":
                model_class = LSTM_Hybrid

        else:
            if hyperparams["model"] == "lstm":
                model_class = LSTM_Regression
            elif hyperparams["model"] == "gru":
                model_class = GRU_Regression

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    # if args.classification and args.model == "lstm_hybrid":
    #     mask = trainset.symb_mask
    #     targets_directions = smartutils.sharedX(sphere.vertices)[T.cast(trainset.symb_targets[:, :, 0], dtype="int32")]
    #     reconstruction_error = T.sum(((model.directions - targets_directions)**2), axis=2)
    #     avg_reconstruction_error_per_sequence = T.sum(reconstruction_error*mask, axis=1)  # / T.sum(mask, axis=1)
    #     # avg_reconstruction_error_monitor = views.MonitorVariable(T.mean(avg_reconstruction_error_per_sequence))
    #     avg_reconstruction_error_monitor = views.MonitorVariable(T.sum(avg_reconstruction_error_per_sequence))
    #     avg_reconstruction_error = tasks.AveragePerEpoch(avg_reconstruction_error_monitor)
    #     trainer.append_task(avg_reconstruction_error)
    #     trainer.append_task(tasks.Print("Avg. reconstruction error:     : {}", avg_reconstruction_error))

    # Print NLL mean/stderror.
    if hyperparams['regression']:
        train_loss = L2DistanceForSequences(model, trainset)
        train_loss.losses
        train_batch_scheduler = SequenceBatchScheduler(trainset, batch_size=50)
        # train_error = views.LossView(loss=train_loss, batch_scheduler=train_batch_scheduler)
        # train_error.losses

        predict, losses, targets, masks = log(train_batch_scheduler, model.regression_out, train_loss.L2_error_per_item, trainset.symb_targets*1, trainset.symb_mask*1)

        A = np.array([(losses[i]*masks[i]).sum()/masks[i].sum() for i in range(len(masks))])

        # losses, masks = list(zip(*losses_and_masks))
        # losses = list(itertools.chain(*losses))
        # masks = list(itertools.chain(*masks))
        from nibabel.streamlines import ArraySequence
        losses_per_streamline = ArraySequence([l[:m.sum()] for l, m in zip(losses, masks)])
        predict_per_streamline = ArraySequence([p[:m.sum()] for p, m in zip(predict, masks)])
        targets_per_streamline = ArraySequence([t[:m.sum()] for t, m in zip(targets, masks)])
        # valid_loss = L2DistanceForSequences(model, validset)
        # valid_batch_scheduler = SequenceBatchScheduler(validset, batch_size=50)
        # valid_error = views.LossView(loss=valid_loss, batch_scheduler=valid_batch_scheduler)
        # valid_error.losses

        streamlines = []
        colors = []
        for d, p, l in zip(targets_per_streamline, predict_per_streamline, losses_per_streamline):
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

        print(losses_per_streamline[0])
        print(sum(map(sum, losses_per_streamline)) / sum(map(len, losses_per_streamline)))
        print(np.mean(np.array(list(map(sum, losses_per_streamline))) / np.array(list(map(len, losses_per_streamline)))))

        tractogram = nib.streamlines.Tractogram(streamlines, data_per_point={"colors": colors})
        nib.streamlines.save(tractogram[[0]], 'eval_{}_single.trk'.format(hyperparams['model']))
        nib.streamlines.save(tractogram, 'eval_{}.trk'.format(hyperparams['model']))

    # elif args.classification:
    #     valid_loss = ErrorForSequenceWithClassTarget(model, validset)
    #     valid_batch_scheduler = SequenceBatchSchedulerWithClassTarget(validset, batch_size=50)
    #     error = views.LossView(loss=valid_loss, batch_scheduler=valid_batch_scheduler)
    #     trainer.append_task(tasks.Print("Validset - Error        : {0:.2%} ± {1:.2f}", error.mean, error.stderror))
    #     lookahead_loss = error.mean

    # Plot some graphs
    # plt.figure()
    # plt.subplot(121)
    # plt.title("Loss")
    # plt.plot(logger.get_variable_history(0), label="Train")
    # plt.plot(logger.get_variable_history(1), label="Valid")
    # plt.legend()

    # plt.subplot(122)
    # plt.title("Gradient norm")
    # plt.plot(logger.get_variable_history(2), label="||g||")
    # plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse

import nibabel as nib

import theano.tensor as T

from learn2track.rnn import RNN
from learn2track.lstm import LSTM, LSTM_regression

from learn2track.loss import L2DistanceWithBinaryCrossEntropy, L2DistanceForSequences
from learn2track.view import RegressionError
from learn2track.batch_scheduler import BundlesBatchScheduler, SequenceBatchScheduler
from learn2track.utils import Timer, load_bundles

from smartlearner import Trainer, tasks, Dataset
from smartlearner import tasks
from smartlearner import stopping_criteria
from smartlearner import views
from smartlearner.optimizers import SGD, AdaGrad
from smartlearner.direction_modifiers import ConstantLearningRate

NB_POINTS = 100


def buildArgsParser():
    DESCRIPTION = "Script to train an LSTM to do tractography."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('name', type=str, help="name of the experiment.")
    p.add_argument('bundles_path', type=str, help='folder containing training data (.npz files).')
    p.add_argument('--learn_stopping', action="store_true", help='Whether or not to learn to stop tracking.')

    return p


def learn_direction_and_stopping(args):
    with Timer("Loading dataset"):
        trainset, validset, testset = load_bundles(args.bundles_path)

        # TODO: do this when generating the data (in the create_dataset script)
        # Normalize (inplace) the target directions
        for bundle in trainset.bundles:
            for target in bundle.targets:
                target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        for target in validset.targets:
            target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        # for target in testset.targets:
        #     target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        batch_size = 100
        batch_scheduler = BundlesBatchScheduler(trainset, batch_size, nb_updates_per_epoch=50)

    with Timer("Creating model"):
        hidden_size = 100
        #model = RNN(trainset.input_shape[-1], hidden_size, trainset.target_shape[-1])
        model = LSTM(trainset.input_shape[-1], hidden_size, trainset.target_shape[-1])
        model.initialize()  # By default, uniform initialization.

        save_path = pjoin('experiments', args.name)

    with Timer("Building optimizer"):
        #loss = L2DistanceForSequence(model, trainset)
        #loss = L2DistanceWithBinaryCrossEntropy(model, trainset)
        loss = L2DistanceWithBinaryCrossEntropy(model, trainset)
        optimizer = AdaGrad(loss=loss, lr=0.01)

    with Timer("Building trainer"):
        trainer = Trainer(optimizer, batch_scheduler)

        def save_model(*args):
            model.save(save_path)

        # Train for 100 epochs
        trainer.append_task(stopping_criteria.MaxEpochStopping(100))
        # Add early stopping too
        error = views.LossView(loss=L2DistanceWithBinaryCrossEntropy(model, validset),
                               batch_scheduler=SequenceBatchScheduler(validset, batch_size=512))
        trainer.append_task(stopping_criteria.EarlyStopping(error.mean, lookahead=10, callback=save_model))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())
        trainer.append_task(tasks.PrintAverageTrainingLoss(loss))

        # Print some variables
        trainer.append_task(tasks.PrintVariable("Avg. Objective: {0}\tCross: {1}",
                                                T.mean(loss.mean_sqr_error), T.mean(loss.cross_entropy),
                                                ))

        # Print mean/stderror of loss.
        trainer.append_task(tasks.Print("Validset - Error: {0:.4f} ± {1:.4f}", error.mean, error.stderror))
        trainer._build_theano_graph()

    with Timer("Training"):
        trainer.train()

    with Timer("Saving model"):
        model.save(save_path)


def learn_direction(args):
    with Timer("Loading dataset"):
        trainset, validset, testset = load_bundles(args.bundles_path)

        # TODO: do this when generating the data (in the create_dataset script)
        # Normalize (inplace) the target directions
        for bundle in trainset.bundles:
            for target in bundle.targets:
                target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        for target in validset.targets:
            target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        # for target in testset.targets:
        #     target /= np.sqrt(np.sum(target**2, axis=1, keepdims=True))

        batch_size = 100
        batch_scheduler = BundlesBatchScheduler(trainset, batch_size, nb_updates_per_epoch=50)

    with Timer("Creating model"):
        hidden_size = 500
        model = LSTM_regression(trainset.input_shape[-1], hidden_size, trainset.target_shape[-1])
        model.initialize()  # By default, uniform initialization.

        save_path = pjoin('experiments', args.name)

    with Timer("Building optimizer"):
        loss = L2DistanceForSequences(model, trainset)
        optimizer = AdaGrad(loss=loss, lr=0.01)

    with Timer("Building trainer"):
        trainer = Trainer(optimizer, batch_scheduler)

        def save_model(obj, status):
            print("\n*** Best epoch: {0}".format(obj.best_epoch))
            model.save(save_path)

        # Train for 100 epochs
        trainer.append_task(stopping_criteria.MaxEpochStopping(100))
        # Add early stopping too
        error = views.LossView(loss=L2DistanceForSequences(model, validset),
                               batch_scheduler=SequenceBatchScheduler(validset, batch_size=512))
        trainer.append_task(stopping_criteria.EarlyStopping(error.mean, lookahead=10, callback=save_model))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())
        trainer.append_task(tasks.PrintAverageTrainingLoss(loss))

        # Print mean/stderror of loss.
        trainer.append_task(tasks.Print("Validset - Error: {0:.4f} ± {1:.4f}", error.mean, error.stderror))
        trainer._build_theano_graph()

    with Timer("Training"):
        trainer.train()

    with Timer("Saving model"):
        model.save(save_path)


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    if args.learn_stopping:
        learn_direction_and_stopping(args)
    else:
        learn_direction(args)


if __name__ == "__main__":
    main()

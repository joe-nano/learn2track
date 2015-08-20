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
from learn2track.lstm import LSTM

from learn2track.loss import L2DistanceForSequence
from learn2track.view import RegressionError
from learn2track.dataset import BundlesBatchScheduler
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

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    with Timer("Loading dataset"):
        trainset, validset, testset = load_bundles(args.bundles_path)

    with Timer("Creating model"):
        hidden_size = 100
        #model = RNN(trainset.input_shape[-1], hidden_size, trainset.target_shape[-1])
        model = LSTM(trainset.input_shape[-1], hidden_size, trainset.target_shape[-1])
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        loss = L2DistanceForSequence(model, trainset)
        optimizer = SGD(loss=loss)
        optimizer.append_direction_modifier(ConstantLearningRate(0.001))

    with Timer("Building trainer"):
        batch_scheduler = BundlesBatchScheduler(trainset, 100, nb_updates_per_epoch=50)
        trainer = Trainer(optimizer, batch_scheduler)

        # Train for 100 epochs
        trainer.append_task(stopping_criteria.MaxEpochStopping(10))
        # Add early stopping too
        error = RegressionError(model.use, validset)
        trainer.append_task(stopping_criteria.EarlyStopping(error.mean, lookahead=10))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of reconstruction errors.
        trainer.append_task(tasks.Print("Validset - Error: {0:.4f} ± {1:.4f}", error.mean, error.stderror))

    with Timer("Training"):
        trainer.train()

    with Timer("Saving model"):
        save_path = pjoin('experiments', args.name)
        model.save(save_path)


if __name__ == "__main__":
    main()

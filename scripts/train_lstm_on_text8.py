#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from os.path import join as pjoin

import shutil
import argparse
import datetime
import theano.tensor as T

import pickle
import numpy as np

from smartlearner import Trainer
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria
import smartlearner.utils as smartutils

from learn2track import utils
from learn2track.utils import load_text8, Timer
from learn2track.lstm import LSTM_Softmax, LSTM_SoftmaxWithFeaturesExtraction
from learn2track.factories import ACTIVATION_FUNCTIONS
from learn2track.factories import WEIGHTS_INITIALIZERS, weigths_initializer_factory
from learn2track.factories import optimizer_factory
from learn2track.losses import SequenceNegativeLogLikelihood
from learn2track.batch_schedulers import SequenceBatchSchedulerText8


DATASETS = ["text8"]
MODELS = ['lstm', 'lstm_extraction']


def build_train_lstm_argparser(subparser):
    DESCRIPTION = "Train a LSTM."

    p = subparser.add_parser("lstm", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (convnade)
    model = p.add_argument_group("LSTM arguments")

    model.add_argument('--hidden-sizes', type=int, nargs='+', default=500,
                       help="Size of the hidden layers. Default: 500")

    model.add_argument('--hidden-activation', type=str, choices=ACTIVATION_FUNCTIONS, default=ACTIVATION_FUNCTIONS[0],
                       help="Activation functions: {}".format(ACTIVATION_FUNCTIONS),)
    model.add_argument('--weights-initialization', type=str, default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS,
                       help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)))
    model.add_argument('--initialization-seed', type=int, default=1234,
                       help='seed used to generate random numbers. Default=1234')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def build_train_lstm_extraction_argparser(subparser):
    DESCRIPTION = "Train a LSTM that has a features extraction as the first layer."

    p = subparser.add_parser("lstm_extraction", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, choices=DATASETS, metavar="DATASET",
                   help='dataset to train on [{0}].'.format(', '.join(DATASETS))),

    # Model options (convnade)
    model = p.add_argument_group("LSTM arguments")

    model.add_argument('--features-size', type=int, default=250,
                       help="Size of the features space (i.e. the first layer). Default: 250")

    model.add_argument('--hidden-sizes', type=int, nargs='+', default=500,
                       help="Size of the hidden layers. Default: 500")

    model.add_argument('--hidden-activation', type=str, choices=ACTIVATION_FUNCTIONS, default=ACTIVATION_FUNCTIONS[0],
                       help="Activation functions: {}".format(ACTIVATION_FUNCTIONS),)
    model.add_argument('--weights-initialization', type=str, default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS,
                       help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)))
    model.add_argument('--initialization-seed', type=int, default=1234,
                       help='seed used to generate random numbers. Default=1234')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def buildArgsParser():
    DESCRIPTION = ("Script to train a LSTM model on a dataset"
                   " (Text8) using Theano.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    duration = p.add_argument_group("Training duration options")
    duration.add_argument('--max-epoch', type=int, metavar='N', help='if specified, train for a maximum of N epochs.')
    duration.add_argument('--lookahead', type=int, metavar='K', default=10,
                          help='use early stopping with a lookahead of K. Default: 10')
    duration.add_argument('--lookahead-eps', type=float, default=1e-3,
                          help='in early stopping, an improvement is whenever the objective improve of at least `eps`. Default: 1e-3',)

    # Training options
    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int,
                          help='size of the batch to use when training the model. Default: 100.', default=100)
    training.add_argument('--sequence-length', type=int,
                          help='size of every training sequence. Default: 10.', default=10)

    # Optimizer options
    optimizer = p.add_argument_group("Optimizer (required)")
    optimizer = optimizer.add_mutually_exclusive_group(required=True)
    optimizer.add_argument('--SGD', metavar="LR", type=str, help='use SGD with constant learning rate for training.')
    optimizer.add_argument('--AdaGrad', metavar="LR [EPS=1e-6]", type=str, help='use AdaGrad for training.')
    optimizer.add_argument('--Adam', action="store_true", help='use Adam for training.')

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('--name', type=str,
                         help='name of the experiment. Default: name is generated from arguments.')

    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')

    subparser = p.add_subparsers(title="Models", dest="model")
    subparser.required = True   # force 'required' testing
    build_train_lstm_argparser(subparser)
    build_train_lstm_extraction_argparser(subparser)

    return p


def maybe_create_experiment_folder(args):
    # Extract experiments hyperparameters
    hyperparams = dict(vars(args))

    # Remove hyperparams that should not be part of the hash
    del hyperparams['max_epoch']
    del hyperparams['force']
    del hyperparams['name']

    # Get/generate experiment name
    experiment_name = args.name
    if experiment_name is None:
        experiment_name = utils.generate_uid_from_string(repr(hyperparams))

    # Create experiment folder
    experiment_path = pjoin(".", "experiments", experiment_name)
    resuming = False
    if os.path.isdir(experiment_path) and not args.force:
        resuming = True
        print("### Resuming experiment ({0}). ###\n".format(experiment_name))
        # Check if provided hyperparams match those in the experiment folder
        hyperparams_loaded = smartutils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
        if hyperparams != hyperparams_loaded:
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams[k]) for k in sorted(hyperparams.keys())]) + "\n}")
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams_loaded[k]) for k in sorted(hyperparams_loaded.keys())]) + "\n}")
            print("The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting.")
            exit(1)
    else:
        if os.path.isdir(experiment_path):
            shutil.rmtree(experiment_path)

        os.makedirs(experiment_path)
        smartutils.save_dict_to_json_file(pjoin(experiment_path, "hyperparams.json"), hyperparams)

    return experiment_path, hyperparams, resuming


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    experiment_path, hyperparams, resuming = maybe_create_experiment_folder(args)
    if resuming:
        print ("Resuming:", experiment_path)
    else:
        print ("Creating:", experiment_path)

    with Timer("Loading dataset"):
        trainset, validset = load_text8()

    with Timer("Building model"):
        if args.model == "lstm":
            model = LSTM_Softmax(trainset.vocabulary_size, args.hidden_sizes, trainset.vocabulary_size)
        elif args.model == "lstm_extraction":
            model = LSTM_SoftmaxWithFeaturesExtraction(trainset.vocabulary_size, args.features_size, args.hidden_sizes, trainset.vocabulary_size)

        model.initialize(weigths_initializer_factory(args.weights_initialization,
                                                     seed=args.initialization_seed))

        batch_scheduler = SequenceBatchSchedulerText8(trainset, batch_size=args.batch_size, sequence_length=args.sequence_length)

    with Timer("Building optimizer"):
        loss = SequenceNegativeLogLikelihood(model, trainset)
        optimizer = optimizer_factory(hyperparams, loss)

    with Timer("Building trainer"):
        trainer = Trainer(optimizer, batch_scheduler)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Log training error
        loss_monitor = views.MonitorVariable(loss.loss)
        avg_loss = tasks.AveragePerEpoch(loss_monitor)
        trainer.append_task(avg_loss)

        # Print average training loss.
        trainer.append_task(tasks.Print("Avg. training loss:     : {}", avg_loss))

        # Print NLL mean/stderror.

        nll = views.LossView(loss=SequenceNegativeLogLikelihood(model, validset),
                             batch_scheduler=SequenceBatchSchedulerText8(validset, batch_size=len(validset),
                                                                         sequence_length=1, nb_updates_per_epoch=1))

        trainer.append_task(tasks.Print("Validset - NLL          : {0:.2f} ± {1:.2f}", nll.mean, nll.stderror))

        # Save training progression
        def save_model(*args):
            trainer.save(experiment_path)

        trainer.append_task(stopping_criteria.EarlyStopping(nll.mean, lookahead=args.lookahead, eps=args.lookahead_eps, callback=save_model))

        if args.max_epoch is not None:
            trainer.append_task(stopping_criteria.MaxEpochStopping(args.max_epoch))

        trainer.build_theano_graph()

    if resuming:
        with Timer("Loading"):
            trainer.load(experiment_path)

    with Timer("Training"):
        trainer.train()

    trainer.save(experiment_path)
    model.save(experiment_path)

if __name__ == '__main__':
    main()

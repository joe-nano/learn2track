#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
from os.path import join as pjoin

from smartlearner import utils as smartutils

from learn2track import utils
from learn2track.utils import Timer, load_text8, id2char, char2id


def build_argparser():
    DESCRIPTION = "Sample from a LSTM model trained on Text8."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--nb-sentences', type=int, default=10,
                   help="Number of sentences to generate. Default: 10")
    p.add_argument('--sentence-length', type=int, default=80,
                   help="Number of characters in a sentence. Default: 80")
    p.add_argument('--seed', type=int, default=1234,
                   help="Seed used to choose the first character of each sentence. Default: 1234")

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

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
        trainset, validset = load_text8()

    with Timer("Loading model"):
        if hyperparams["model"] == "lstm":
            from learn2track.lstm import LSTM_Softmax
            model_class = LSTM_Softmax
        elif hyperparams["model"] == "lstm_extraction":
            from learn2track.lstm import LSTM_SoftmaxWithFeaturesExtraction
            model_class = LSTM_SoftmaxWithFeaturesExtraction

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path))  # Create new instance
        model.load(pjoin(experiment_path))  # Restore state.
        print(str(model))

    # Generate sentences.
    rng = np.random.RandomState(args.seed)
    text = [""] * args.nb_sentences

    input = np.zeros((args.nb_sentences, trainset.vocabulary_size), dtype=np.float32)
    input[np.arange(args.nb_sentences), rng.randint(0, trainset.vocabulary_size, size=(args.nb_sentences,))] = 1.

    for i in range(args.sentence_length):
        for j in range(args.nb_sentences):
            text[j] += id2char(np.argmax(input[j]))

        input = model.seq_next(input)
        indices = np.argmax(input, axis=1)
        input[:, :] = 0.
        input[np.arange(args.nb_sentences), indices] = 1.

    print("\n".join(text))

if __name__ == '__main__':
    main()

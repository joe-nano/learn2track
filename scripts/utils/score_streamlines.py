#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# # Hack so you don't have to put the library containing this script in the PYTHONPATH.
# sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse
import itertools
import theano
import time
import nibabel as nib
from nibabel.streamlines import ArraySequence, Tractogram

from smartlearner import views
from smartlearner.status import Status
from smartlearner import utils as smartutils

from learn2track.utils import Timer, log_variables
from learn2track.factories import model_factory
from learn2track.factories import loss_factory, batch_scheduler_factory

from learn2track import datasets
from learn2track import utils
from learn2track import batch_schedulers
from learn2track.neurotools import VolumeManager


def build_parser():
    DESCRIPTION = ("Score tractogram according to a model loss.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', help='name/path of the experiment.')
    p.add_argument('signal', help='Diffusion signal (.nii|.nii.gz).')
    p.add_argument('tractogram', help='tractogram to score.')
    p.add_argument('--bvals', help='File containing diffusion gradient lengths (Default: guess it from `signal`).')
    p.add_argument('--bvecs', help='File containing diffusion gradient directions (Default: guess it from `signal`).')
    p.add_argument('--out', default="tractogram.trk", help='output filename (TRK). Default: %(default)s')

    p.add_argument('--batch_size', type=int, default=200, help='size of the batch.')
    p.add_argument('--prune', type=float, help='prune streamlines having a loss higher (or lower if --NLL is used) than the specified threshold.')

    loss_type = p.add_mutually_exclusive_group(required=False)
    loss_type.add_argument('--expected-value', action='store_const', dest='loss_type', const='expected_value',
                           help='Use timestep expected value L2 error to color streamlines')
    loss_type.add_argument('--maximum-component', action='store_const', dest='loss_type', const='maximum_component',
                           help='Use timestep maximum distribution component L2 error to color streamlines')
    loss_type.add_argument('--NLL', action='store_const', dest='loss_type', const='NLL',
                           help='Use NLL to color streamlines')

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def main():
    parser = build_parser()
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

    # Use this for hyperparams added in a new version, but nonexistent from older versions
    retrocompatibility_defaults = {'feed_previous_direction': False,
                                   'predict_offset': False,
                                   'normalize': False,
                                   'keep_step_size': False,
                                   'sort_streamlines': False}
    for new_hyperparams, default_value in retrocompatibility_defaults.items():
        if new_hyperparams not in hyperparams:
            hyperparams[new_hyperparams] = default_value

    with Timer("Loading signal data and tractogram", newline=True):
        volume_manager = VolumeManager()
        dataset = datasets.load_tractography_dataset_from_dwi_and_tractogram(args.signal, args.tractogram, volume_manager,
                                                                             use_sh_coeffs=hyperparams['use_sh_coeffs'],
                                                                             bvals=args.bvals, bvecs=args.bvecs)
        print("Dataset size:", len(dataset))

    with Timer("Loading model"):
        model = None
        if hyperparams['model'] == 'gru_regression':
            from learn2track.models import GRU_Regression
            model = GRU_Regression.create(experiment_path, volume_manager=volume_manager)
        elif hyperparams['model'] == 'gru_mixture':
            from learn2track.models import GRU_Mixture
            model = GRU_Mixture.create(experiment_path, volume_manager=volume_manager)
        elif hyperparams['model'] == 'gru_multistep':
            from learn2track.models import GRU_Multistep_Gaussian
            model = GRU_Multistep_Gaussian.create(experiment_path, volume_manager=volume_manager)
            model.k = 1
            model.m = 1
        elif hyperparams['model'] == 'ffnn_regression':
            from learn2track.models import FFNN_Regression
            model = FFNN_Regression.create(experiment_path, volume_manager=volume_manager)
        else:
            raise NameError("Unknown model: {}".format(hyperparams['model']))

    with Timer("Building evaluation function"):
        # Override K for gru_multistep
        if 'k' in hyperparams:
            hyperparams['k'] = 1

        batch_scheduler = batch_scheduler_factory(hyperparams, dataset,
                                                  use_data_augment=False,  # Otherwise it doubles the number of losses :-/
                                                  train_mode=False,
                                                  batch_size_override=args.batch_size)
        loss_type = args.loss_type
        if loss_type == "NLL":
            loss_type = None

        loss = loss_factory(hyperparams, model, dataset, loss_type=loss_type)
        l2_error = views.LossView(loss=loss, batch_scheduler=batch_scheduler)

    with Timer("Scoring...", newline=True):
        dummy_status = Status()  # Forces recomputing results
        losses = l2_error.losses.view(dummy_status)
        mean = float(l2_error.mean.view(dummy_status))
        stderror = float(l2_error.stderror.view(dummy_status))

        if args.loss_type == "NLL":
            losses = np.exp(-losses)
            mean = np.exp(-mean)
            stderror = np.exp(-stderror)

        print("Loss: {:.4f} ± {:.4f}".format(mean, stderror))
        print("Min: {:.4f}".format(losses.min()))
        print("Max: {:.4f}".format(losses.max()))
        print("Percentiles: {}".format(np.percentile(losses, [0, 25, 50, 75, 100])))

    with Timer("Saving streamlines"):
        tractogram = Tractogram(dataset.streamlines, affine_to_rasmm=dataset.subjects[0].signal.affine)
        tractogram.data_per_streamline['loss'] = losses
        nib.streamlines.save(tractogram, args.out)

    if args.prune is not None:
        with Timer("Saving pruned streamlines"):
            if args.loss_type == "NLL":
                tractogram = tractogram[losses > args.prune]
            else:
                tractogram = tractogram[losses <= args.prune]

            out_filename = args.out[:-4] + "_p{}".format(args.prune) + args.out[-4:]
            nib.streamlines.save(tractogram, out_filename)


if __name__ == "__main__":
    main()

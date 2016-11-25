#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import os

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from itertools import chain

from matplotlib import cm
from matplotlib import colors as mplcolors

from learn2track.factories import loss_factory, batch_scheduler_factory
from learn2track.neurotools import VolumeManager

import numpy as np
from os.path import join as pjoin
import argparse
import nibabel as nib
from nibabel.streamlines import ArraySequence

from smartlearner import utils as smartutils

from learn2track import datasets
from learn2track.utils import Timer, log_variables


def build_evaluation_argparser(subparser):
    description = "Evaluate and color streamlines using a probabilistic model's pdf"

    p = subparser.add_parser("evaluation", description=description, help=description)

    # Options
    metric = p.add_argument_group("Metric evaluated for probabilistic coloring")
    metric = metric.add_mutually_exclusive_group(required=True)

    metric.add_argument('--sequence', action='store_const', dest='metric', const='sequence', help='Use sequence probability to color streamlines')
    metric.add_argument('--timestep', action='store_const', dest='metric', const='timestep', help='Use timestep probability to color streamlines')
    metric.add_argument('--timestep-cumulative-average', action='store_const', dest='metric', const='cumul_avg',
                        help='Use timestep probability cumulative average to color streamlines')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='generate a new tractogram even if it exists.')
    general.add_argument('--batch_size', type=int, default=100, help='size of the batch.')
    general.add_argument('--out', default='tractogram.trk', help='output file')


def build_generation_argparser(subparser):
    description = "Color the GT streamlines using a base color, then generate predictions and highlight using L2 reconstruction error"

    p = subparser.add_parser("prediction", description=description, help=description)

    # Options
    prediction = p.add_argument_group("Model prediction method")
    prediction = prediction.add_mutually_exclusive_group(required=True)

    prediction.add_argument('--expected-value', action='store_const', dest='prediction', const='expected_value',
                            help='Use timestep expected value L2 error to color streamlines')
    prediction.add_argument('--maximum-component', action='store_const', dest='prediction', const='maximum_component',
                            help='Use timestep maximum distribution component L2 error to color streamlines')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='generate a new tractogram even if it exists.')
    general.add_argument('--batch_size', type=int, default=100, help='size of the batch.')
    general.add_argument('--out', default='tractogram.trk', help='output file')


def build_args_parser():
    description = "Script to generate and visualize model predictions against the GT"
    p = argparse.ArgumentParser(description=description)

    # General options
    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('--streamlines', type=str, required=True, help='folder containing GT streamlines data (.npz files).')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='generate a new tractogram even if it exists.')
    general.add_argument('--batch_size', type=int, default=100, help='size of the batch.')
    general.add_argument('--out', default='tractogram.trk', help='output file')

    subparser = p.add_subparsers(title="Method to use (evaluation|prediction)", dest="method")
    subparser.required = True
    build_evaluation_argparser(subparser)
    build_generation_argparser(subparser)

    return p


def evaluation_tractogram(hyperparams, model, dataset, batch_size_override, metric):
    loss = loss_factory(hyperparams, model, dataset, loss_type=None)
    batch_scheduler = batch_scheduler_factory(hyperparams, dataset, train_mode=False, batch_size_override=batch_size_override, use_data_augment=False)

    loss.losses  # Hack to generate update dict in loss :(

    timestep_losses, seq_losses, inputs, targets, masks = log_variables(batch_scheduler,
                                                                        model,
                                                                        loss.loss_per_time_step,
                                                                        loss.loss_per_seq,
                                                                        dataset.symb_inputs * 1,
                                                                        dataset.symb_targets * 1,
                                                                        dataset.symb_mask * 1)

    timesteps_loss = ArraySequence([l[:int(m.sum())] for l, m in zip(chain(*timestep_losses), chain(*masks))])
    seq_loss = np.array(list(chain(*seq_losses)))
    timesteps_inputs = ArraySequence([i[:int(m.sum())] for i, m in zip(chain(*inputs), chain(*masks))])
    # Use np.squeeze in case gru_multistep is used to remove the empty k=1 dimension
    timesteps_targets = ArraySequence([np.squeeze(t[:int(m.sum())]) for t, m in zip(chain(*targets), chain(*masks))])

    if metric == 'sequence':
        # Color is based on sequence loss
        values = seq_loss
    elif metric == 'timestep' or metric == 'cumul_avg':
        # Color is based on timestep loss
        values = np.concatenate(timesteps_loss)
    else:
        raise ValueError("Unrecognized metric: {}".format(metric))

    cmap = cm.get_cmap('bwr')
    vmin = np.percentile(values, 5)
    vmax = np.percentile(values, 95)
    scalar_map = cm.ScalarMappable(norm=mplcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    streamlines = []
    colors = []

    for i, t, l, seq_l in zip(timesteps_inputs, timesteps_targets, timesteps_loss, seq_loss):
        pts = np.r_[i[:, :3], [i[-1, :3] + t[-1]]]

        color = np.zeros_like(pts)
        if metric == 'sequence':
            # Streamline color is based on sequence loss
            color[:, :] = scalar_map.to_rgba(seq_l, bytes=True)[:3]
        elif metric == 'timestep':
            # Streamline color is based on timestep loss
            # Identify first point with green
            color[0, :] = [0, 255, 0]
            color[1:, :] = scalar_map.to_rgba(l, bytes=True)[:, :3]
        elif metric == 'cumul_avg':
            # Streamline color is based on timestep loss

            # Compute cumulative average
            cumul_avg = np.cumsum(l) / np.arange(1, len(l) + 1)

            # Identify first point with green
            color[0, :] = [0, 255, 0]
            color[1:, :] = scalar_map.to_rgba(cumul_avg, bytes=True)[:, :3]
        else:
            raise ValueError("Unrecognized metric: {}".format(metric))

        streamlines.append(pts)
        colors.append(color)

    tractogram = nib.streamlines.Tractogram(streamlines, data_per_point={"colors": colors})
    return tractogram


def prediction_tractogram(hyperparams, model, dataset, batch_size_override, prediction_method):
    loss = loss_factory(hyperparams, model, dataset, loss_type=prediction_method)
    batch_scheduler = batch_scheduler_factory(hyperparams, dataset, train_mode=False, batch_size_override=batch_size_override, use_data_augment=False)

    loss.losses  # Hack to generate update dict in loss :(

    predictions = loss.samples

    predict, timestep_losses, inputs, targets, masks = log_variables(batch_scheduler,
                                                                     model,
                                                                     predictions,
                                                                     loss.loss_per_time_step,
                                                                     dataset.symb_inputs * 1,
                                                                     dataset.symb_targets * 1,
                                                                     dataset.symb_mask * 1)

    # Debug : Print norm stats
    # print("Dataset: {}; # of streamlines: {}".format(dataset.name, len(dataset)))
    # all_predictions = list(chain(*predict))
    # prediction_norms = [(lambda x: np.mean(np.linalg.norm(x, axis=1)))(x) for x in all_predictions]
    # print("Prediction norm --- Mean:{}; Max:{}; Min:{}".format(np.mean(prediction_norms), np.max(prediction_norms), np.min(prediction_norms)))
    # all_targets = list(chain(*targets))
    # target_norms = [(lambda x: np.mean(np.linalg.norm(x, axis=1)))(x) for x in all_targets]
    # print("Target norm --- Mean:{}; Max:{}; Min:{}".format(np.mean(target_norms), np.max(target_norms), np.min(target_norms)))

    timesteps_prediction = ArraySequence([p[:int(m.sum())] for p, m in zip(chain(*predict), chain(*masks))])
    timesteps_loss = ArraySequence([l[:int(m.sum())] for l, m in zip(chain(*timestep_losses), chain(*masks))])
    timesteps_inputs = ArraySequence([i[:int(m.sum())] for i, m in zip(chain(*inputs), chain(*masks))])
    # Use np.squeeze in case gru_multistep is used to remove the empty k=1 dimension
    timesteps_targets = ArraySequence([np.squeeze(t[:int(m.sum())]) for t, m in zip(chain(*targets), chain(*masks))])

    # Color is based on timestep loss
    cmap = cm.get_cmap('bwr')
    values = np.concatenate(timesteps_loss)
    vmin = np.percentile(values, 5)
    vmax = np.percentile(values, 95)
    scalar_map = cm.ScalarMappable(norm=mplcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    streamlines = []
    colors = []

    for i, t, p, l in zip(timesteps_inputs, timesteps_targets, timesteps_prediction, timesteps_loss):
        pts = np.r_[i[:, :3], [i[-1, :3] + t[-1]]]

        streamline = np.zeros(((len(pts) - 1) * 3 + 1, 3))
        streamline[::3] = pts
        streamline[1:-1:3] = pts[:-1] + p
        streamline[2:-1:3] = pts[:-1]
        streamlines.append(streamline)

        # Color input streamlines in a uniform color, then color predictions based on L2 error
        color = np.zeros_like(streamline)

        # Base color of streamlines is minimum value (best score)
        color[:] = scalar_map.to_rgba(vmin, bytes=True)[:3]
        color[1:-1:3, :] = scalar_map.to_rgba(l, bytes=True)[:, :3]
        colors.append(color)

    tractogram = nib.streamlines.Tractogram(streamlines, data_per_point={"colors": colors})
    return tractogram


def main():
    parser = build_args_parser()
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

    with Timer("Loading dataset", newline=True):
        volume_manager = VolumeManager()
        dataset = datasets.load_tractography_dataset([args.streamlines], volume_manager, name="dataset", use_sh_coeffs=hyperparams['use_sh_coeffs'])
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
        else:
            raise NameError("Unknown model: {}".format(hyperparams['model']))
        print(str(model))

    tractogram_file = pjoin(experiment_path, args.out)
    if not os.path.isfile(tractogram_file) or args.force:
        if args.method == 'prediction':
            tractogram = prediction_tractogram(hyperparams, model, dataset, args.batch_size, args.prediction)
        elif args.method == 'evaluation':
            tractogram = evaluation_tractogram(hyperparams, model, dataset, args.batch_size, args.metric)
        else:
            raise ValueError("Unrecognized method: {}".format(args.method))

        tractogram.affine_to_rasmm = dataset.subjects[0].signal.affine
        nib.streamlines.save(tractogram, tractogram_file)
    else:
        print("Tractogram already exists. (use --force to generate it again)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# # Hack so you don't have to put the library containing this script in the PYTHONPATH.
# sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from os.path import join as pjoin
import argparse
import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes

from smartlearner import views
from smartlearner.status import Status
from smartlearner import utils as smartutils

from learn2track.utils import Timer
from learn2track.factories import loss_factory, batch_scheduler_factory

from learn2track import datasets
from learn2track.neurotools import VolumeManager

try:
    from learn2track import vizu
    vizu_available = True
except:
    vizu_available = False


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
    p.add_argument('--keep-top', type=float,
                   help='top percent of streamlines to keep. Streamlines are ranked according to their NLL.')
    p.add_argument('--step-size', type=float,
                   help='If specified, all streamlines will have this step size (in mm).')

    loss_type = p.add_mutually_exclusive_group(required=False)
    loss_type.add_argument('--expected-value', action='store_const', dest='loss_type', const='expected_value',
                           help='Use timestep expected value L2 error to color streamlines')
    loss_type.add_argument('--maximum-component', action='store_const', dest='loss_type', const='maximum_component',
                           help='Use timestep maximum distribution component L2 error to color streamlines')
    loss_type.add_argument('--nll-mean', action='store_const', dest='loss_type', const='nll_mean',
                           help='Use NLL averaged over the time steps to color streamlines')
    loss_type.add_argument('--nll-sum', action='store_const', dest='loss_type', const='nll_sum',
                           help='Use NLL summed over the time steps to color streamlines')

    if vizu_available:
        p.add_argument('--vizu', action='store_true', help='check that streamlines fit on top of the diffusion signal.')

    p.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    if args.keep_top < 0:
        parser.error("--keep-top must be between in [0, 1].")

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
                                                                             bvals=args.bvals, bvecs=args.bvecs,
                                                                             step_size=args.step_size)
        print("Dataset size:", len(dataset))

        if vizu_available and args.vizu:
            vizu.check_dataset_integrity(dataset, subset=0.2)

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
        loss = loss_factory(hyperparams, model, dataset, loss_type=args.loss_type)
        l2_error = views.LossView(loss=loss, batch_scheduler=batch_scheduler)

    with Timer("Scoring...", newline=True):
        dummy_status = Status()  # Forces recomputing results
        losses = l2_error.losses.view(dummy_status)
        mean = float(l2_error.mean.view(dummy_status))
        stderror = float(l2_error.stderror.view(dummy_status))

        print("Loss: {:.4f} ± {:.4f}".format(mean, stderror))
        print("Min: {:.4f}".format(losses.min()))
        print("Max: {:.4f}".format(losses.max()))
        print("Percentiles: {}".format(np.percentile(losses, [0, 25, 50, 75, 100])))

    with Timer("Saving streamlines"):
        nii = dataset.subjects[0].signal
        tractogram = nib.streamlines.Tractogram(dataset.streamlines,
                                                affine_to_rasmm=nii.affine)
        tractogram.data_per_streamline['loss'] = losses

        header = {}
        header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
        header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
        header[Field.DIMENSIONS] = nii.shape[:3]
        header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

        nib.streamlines.save(tractogram.copy(), args.out, header=header)

    if args.keep_top is not None:
        with Timer("Saving top {}% streamlines".format(args.keep_top)):
            idx = np.argsort(losses)
            idx = idx[:int(args.keep_top * len(losses))]
            print("Keeping {}/{} streamlines".format(len(idx), len(losses)))
            tractogram = tractogram[idx]
            out_filename = args.out[:-4] + "_top{}".format(args.keep_top) + ".tck"
            nib.streamlines.save(tractogram, out_filename)


if __name__ == "__main__":
    main()

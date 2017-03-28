#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools

import dipy
import nibabel as nib
import numpy as np
import os
import theano
import theano.tensor as T
from os.path import join as pjoin
from smartlearner import utils as smartutils

from learn2track import neurotools
from learn2track.utils import Timer

floatX = theano.config.floatX


def build_argparser():
    DESCRIPTION = "Generate a white matter mask from diffusion information using a trained model."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--out', type=str,
                   help="name of the output mask (.tck|.trk). Default: auto generate a meaningful name")
    p.add_argument('--prefix', type=str,
                   help="prefix to use for the name of the output tractogram, only if it is auto generated.")

    p.add_argument('--batch-size', type=int, help="number of coordinates to process at the same time. "
                                                  "Default: the biggest possible, divide by half until it fits in memory")

    # Optional parameters
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    p.add_argument('-f', '--force', action='store_true', help='overwrite existing mask')

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

    with Timer("Loading DWIs"):
        # Load gradients table
        dwi_name = args.dwi
        if dwi_name.endswith(".gz"):
            dwi_name = dwi_name[:-3]
        if dwi_name.endswith(".nii"):
            dwi_name = dwi_name[:-4]
        bvals_filename = dwi_name + ".bvals"
        bvecs_filename = dwi_name + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        if hyperparams["use_sh_coeffs"]:
            # Use 45 spherical harmonic coefficients to represent the diffusion signal.
            weights = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)
        else:
            # Resample the diffusion signal to have 100 directions.
            weights = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

    with Timer("Loading model"):
        if hyperparams["model"] == "ffnn_classification":
            from learn2track.models import FFNN_Classification
            model_class = FFNN_Classification
        else:
            raise ValueError("Unknown model!")

        kwargs = {}
        volume_manager = neurotools.VolumeManager()
        volume_manager.register(weights)
        kwargs['volume_manager'] = volume_manager

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path), **kwargs)  # Create new instance and restore model.
        print(str(model))

    with Timer("Generating mask"):
        symb_input = T.matrix(name="input")
        model_symb_pred = model.get_output(symb_input)
        f = theano.function(inputs=[symb_input], outputs=[model_symb_pred])

        generated_mask = np.zeros(dwi.shape[:3]).astype(np.float32)

        # all_coords.shape = (n_coords, 3)
        all_coords = np.argwhere(generated_mask == 0)

        volume_ids = np.zeros((all_coords.shape[0], 1))
        all_coords_and_volume_ids = np.concatenate((all_coords, volume_ids), axis=1).astype(np.float32)

        batch_size = args.batch_size if args.batch_size else len(all_coords_and_volume_ids)
        probs = []
        while batch_size > 1:
            print("Trying to to process batches of size {} out of {}".format(batch_size, len(all_coords_and_volume_ids)))
            nb_batches = int(np.ceil(len(all_coords_and_volume_ids) / batch_size))
            try:
                for batch_count in range(nb_batches):
                    start = batch_count * batch_size
                    end = (batch_count + 1) * batch_size
                    probs.extend(f(all_coords_and_volume_ids[start:end])[-1])
                    print("Generated batch {} out of {}".format(batch_count+1, nb_batches))
                break
            except MemoryError:
                print("{} coordinates at the same time is too much!".format(batch_size))
                batch_size //= 2
            except RuntimeError:
                print("{} coordinates at the same time is too much!".format(batch_size))
                batch_size //= 2
        if not probs:
            raise RuntimeError("Could not generate predictions...")

        generated_mask[np.where(generated_mask == 0)] = np.array(probs) > 0.5

    with Timer("Saving generated mask"):
        filename = args.out
        if args.out is None:
            prefix = args.prefix
            if prefix is None:
                dwi_name = os.path.basename(args.dwi)
                if dwi_name.endswith(".nii.gz"):
                    dwi_name = dwi_name[:-7]
                else:  # .nii
                    dwi_name = dwi_name[:-4]

                prefix = os.path.basename(os.path.dirname(args.dwi)) + dwi_name
                prefix = prefix.replace(".", "_")

            filename = "{}.nii.gz".format(prefix)

        save_path = pjoin(experiment_path, filename)
        try:  # Create dirs, if needed.
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

        print("Saving to {}".format(save_path))
        mask = nib.Nifti1Image(generated_mask, dwi.affine)
        nib.save(mask, save_path)

if __name__ == "__main__":
    main()

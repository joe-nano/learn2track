from __future__ import print_function

import os
import sys
import numpy as np
import theano
from time import time
from os.path import join as pjoin
from scipy.ndimage import map_coordinates
from itertools import chain

from nibabel.streamlines import CompactList

from smartlearner import Dataset
from .dataset import SequenceDataset, BundlesDataset


floatX = theano.config.floatX


def load_bundles(bundles_path):
    dataset_name = "ISMRM15_Challenge"

    bundles = {'trainset': [], 'validset': [], 'testset': []}
    for f in os.listdir(bundles_path):
        if f.endswith("_trainset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            data = np.load(pjoin(bundles_path, f))
            dataset = SequenceDataset(data['inputs'], data['targets'], name=bundle_name)
            bundles["trainset"].append(dataset)
        elif f.endswith("_validset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            data = np.load(pjoin(bundles_path, f))
            dataset = SequenceDataset(data['inputs'], data['targets'], name=bundle_name)
            bundles["validset"].append(dataset)
        elif f.endswith("_testset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            data = np.load(pjoin(bundles_path, f))
            dataset = SequenceDataset(data['inputs'], data['targets'], name=bundle_name)
            bundles["testset"].append(dataset)

    trainset = BundlesDataset(bundles["trainset"], name=dataset_name+"_trainset")

    #validset_inputs = np.concatenate([b.inputs for b in bundles["validset"]])
    #validset_targets = np.concatenate([b.targets for b in bundles["validset"]])
    validset_inputs = list(chain(*[b.inputs for b in bundles["validset"]]))
    validset_targets = list(chain(*[b.targets for b in bundles["validset"]]))
    validset = SequenceDataset(validset_inputs, validset_targets, name=dataset_name+"_validset")
    #validset = BundlesDataset(bundles["validset"], name=dataset_name+"_validset")

    testset_inputs = np.concatenate([b.inputs for b in bundles["testset"]])
    testset_targets = np.concatenate([b.targets for b in bundles["testset"]])
    testset = SequenceDataset(testset_inputs, testset_targets, name=dataset_name+"_testset")
    #testset = BundlesDataset(bundles["testset"], name=dataset_name+"_testset")

    return trainset, validset, testset


def save_bundle(file, inputs, targets):
    """ Saves a bundle compatible with the learn2track framework.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    inputs : `nibabel.streamlines.CompactList` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.CompactList` object
        the direction leading from any 3D point to the next in every streamline.
    """
    np.savez(file,
             inputs_data=inputs._data,
             inputs_offsets=inputs._offsets,
             inputs_lengths=inputs._lengths,
             targets_data=targets._data,
             targets_offsets=targets._offsets,
             targets_lengths=targets._lengths
             )


def load_bundle(file):
    """ Loads a bundle compatible with the learn2track framework.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.

    Returns
    -------
    inputs : `nibabel.streamlines.CompactList` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.CompactList` object
        the direction leading from any 3D point to the next in every streamline.
    """
    data = np.load(file)

    inputs = CompactList()
    inputs._data = data["inputs_data"]
    inputs._offsets = data["inputs_offsets"]
    inputs._lengths = data["inputs_lengths"]

    targets = CompactList()
    targets._data = data["targets_data"]
    targets._offsets = data["targets_offsets"]
    targets._lengths = data["targets_lengths"]

    return inputs, targets

# def load_bundles(bundles_path):
#     dataset_name = "ISMRM15_Challenge"

#     bundles = {'trainset': [], 'validset': [], 'testset': []}
#     for f in os.listdir(bundles_path):
#         if f.endswith("_trainset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             data = np.load(pjoin(bundles_path, f))
#             dataset = Dataset(data['inputs'].astype(floatX), data['targets'].astype(floatX), name=bundle_name, keep_on_cpu=True)
#             bundles["trainset"].append(dataset)
#         elif f.endswith("_validset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             data = np.load(pjoin(bundles_path, f))
#             dataset = Dataset(data['inputs'].astype(floatX), data['targets'].astype(floatX), name=bundle_name, keep_on_cpu=True)
#             bundles["validset"].append(dataset)
#         elif f.endswith("_testset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             data = np.load(pjoin(bundles_path, f))
#             dataset = Dataset(data['inputs'].astype(floatX), data['targets'].astype(floatX), name=bundle_name, keep_on_cpu=True)
#             bundles["testset"].append(dataset)

#     trainset = BundlesDataset(bundles["trainset"], name=dataset_name+"_trainset")

#     validset_inputs = np.concatenate([b.inputs.get_value() for b in bundles["validset"]])
#     validset_targets = np.concatenate([b.targets.get_value() for b in bundles["validset"]])
#     validset = Dataset(validset_inputs, validset_targets, name=dataset_name+"_validset")
#     #validset = BundlesDataset(bundles["validset"], name=dataset_name+"_validset")

#     testset_inputs = np.concatenate([b.inputs.get_value() for b in bundles["testset"]])
#     testset_targets = np.concatenate([b.targets.get_value() for b in bundles["testset"]])
#     testset = Dataset(testset_inputs, testset_targets, name=dataset_name+"_testset")
#     #testset = BundlesDataset(bundles["testset"], name=dataset_name+"_testset")

#     return trainset, validset, testset


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


def map_coordinates_3d_4d(input_array, indices, affine=None):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

    Notes
    -----
    At some point this will be merged in Dipy. See PR #587.
    """
    if affine is not None:
        inv_affine = np.linalg.inv(affine)
        indices = (np.dot(indices, inv_affine[:3, :3]) + inv_affine[:3, 3])

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def normalize_dwi(dwi, bvals):
    """
    Parameters:
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion weighted images (4D).
    bvals : list of int
        B-values used with each direction.
    """
    # Indices of bvals sorted
    sorted_bvals_idx = np.argsort(bvals)
    nb_b0s = int(np.sum(bvals == 0))
    dwi_weights = dwi.get_data().astype("float32")

    # Get the first b0.
    b0 = dwi_weights[..., [sorted_bvals_idx[0]]]
    # Keep only b-value greater than 0
    weights = dwi_weights[..., sorted_bvals_idx[nb_b0s:]]
    # Make sure in every voxels weights are lower than ones from the b0. (should not happen)
    #weights_normed = np.minimum(weights, b0)
    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.isnan(weights_normed)] = 0.

    return weights_normed

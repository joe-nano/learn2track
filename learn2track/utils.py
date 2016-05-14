from __future__ import print_function

import os
import sys
import numpy as np
import theano
import theano.tensor as T
import itertools
import string
import hashlib
from time import time
from os.path import join as pjoin
from scipy.ndimage import map_coordinates
from itertools import chain

import nibabel as nib
from nibabel.streamlines import ArraySequence

from smartlearner import Dataset
from .dataset import ReconstructionDataset, MaskedSequenceDataset, SequenceDataset, BundlesDataset, StreamlinesDataset


DATASETS_ENV = "DATASETS"

floatX = theano.config.floatX


class StreamlinesData(object):
    def __init__(self, bundle_names):
        self.streamlines = nib.streamlines.ArraySequence()
        self.bundle_ids = np.zeros((0,), dtype=np.int16)
        self.bundle_names = bundle_names

    def add(self, streamlines, bundle_ids):
        self.streamlines.extend(streamlines)
        size = len(self.bundle_ids)
        new_size = size + len(bundle_ids)
        self.bundle_ids.resize((new_size,))
        self.bundle_ids[size:new_size] = bundle_ids

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        streamlines_data = cls(data['bundle_names'])
        streamlines_data.streamlines._data = data['coords']
        streamlines_data.streamlines._offsets = data['offsets']
        streamlines_data.streamlines._lengths = data['lengths']
        streamlines_data.bundle_ids = data['bundle_ids']
        return streamlines_data

    def save(self, filename):
        np.savez(filename,
                 coords=self.streamlines._data.astype(np.float32),
                 offsets=self.streamlines._offsets,
                 lengths=self.streamlines._lengths.astype(np.int16),
                 bundle_ids=self.bundle_ids,
                 bundle_names=self.bundle_names)


def load_streamlines_dataset(dwi_filename, streamlines_filename, name="ISMRM15_Challenge", use_sh_coeffs=False):
    import nibabel as nib
    from dipy.io.gradients import read_bvals_bvecs

    with Timer("Loading DWIs"):
        # Load gradients table
        bvals_filename = dwi_filename.split('.')[0] + ".bvals"
        bvecs_filename = dwi_filename.split('.')[0] + ".bvecs"
        bvals, bvecs = read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(dwi_filename)
        if use_sh_coeffs:
            volume = get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)  # Use 45 spherical harmonic coefficients
        else:
            volume = resample_dwi(dwi, bvals, bvecs).astype(np.float32)  # Resample to 100 directions

    with Timer("Loading streamlines"):
        basename = streamlines_filename[:-len('.npz')]
        if basename.endswith("_trainset"):
            basename = basename[:-len("_trainset")]

        trainset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_trainset.npz"), name=name+"_trainset")
        validset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_validset.npz"), name=name+"_validset")
        testset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_testset.npz"), name=name+"_testset")

    return trainset, validset, testset


def load_ismrm2015_challenge(bundles_path, classification=False):
    dataset_name = "ISMRM15_Challenge"

    bundles = {'trainset': [], 'validset': [], 'testset': []}
    for f in os.listdir(bundles_path):
        if f.endswith("_trainset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["trainset"].append(dataset)
        elif f.endswith("_validset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["validset"].append(dataset)
        elif f.endswith("_testset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["testset"].append(dataset)

    trainset = BundlesDataset(bundles["trainset"], name=dataset_name+"_trainset")

    validset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["validset"]]))
    validset_targets = ArraySequence(chain(*[b.targets for b in bundles["validset"]]))
    validset = MaskedSequenceDataset(validset_inputs, validset_targets, name=dataset_name+"_validset")

    testset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["testset"]]))
    testset_targets = ArraySequence(chain(*[b.targets for b in bundles["testset"]]))
    testset = MaskedSequenceDataset(testset_inputs, testset_targets, name=dataset_name+"_testset")

    if classification:
        # Transform targets (directions) into class id.
        from dipy.data import get_sphere
        sphere = get_sphere("repulsion724")  # All possible directions (normed)
        sphere.vertices = sphere.vertices.astype(theano.config.floatX)

        # Target is the id of the closest direction on the sphere `sphere` determined using cosine similarity.
        # We do this for each point of the streamline and also each point of the reversed streamline.
        directions = sphere.vertices.astype(theano.config.floatX)
        for bundle in trainset.bundles:
            bundle.targets._data = np.c_[np.argmax(np.dot(bundle.targets._data, directions.T), axis=-1)[:, None],
                                         np.argmax(np.dot(-bundle.targets._data, directions.T), axis=-1)[:, None]].astype(theano.config.floatX)

        validset.targets._data = np.c_[np.argmax(np.dot(validset.targets._data, directions.T), axis=-1)[:, None],
                                       np.argmax(np.dot(-validset.targets._data, directions.T), axis=-1)[:, None]].astype(theano.config.floatX)

    return trainset, validset, testset


def load_ismrm2015_challenge_contiguous(bundles_path, classification=False, batch_size=1024*10):
    dataset_name = "ISMRM15_Challenge"

    bundles = {'trainset': [], 'validset': [], 'testset': []}
    for f in os.listdir(bundles_path):
        if f.endswith("_trainset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["trainset"].append(dataset)
        elif f.endswith("_validset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["validset"].append(dataset)
        elif f.endswith("_testset.npz"):
            bundle_name = f.split("/")[-1][:-len(".npz")]
            inputs, targets = load_bundle(pjoin(bundles_path, f))
            dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
            bundles["testset"].append(dataset)

    trainset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["trainset"]]))
    trainset_targets = ArraySequence(chain(*[b.targets for b in bundles["trainset"]]))

    # Shuffle streamlines
    rng = np.random.RandomState(42)
    indices = np.arange(len(trainset_inputs))
    rng.shuffle(indices)
    trainset_inputs = trainset_inputs[indices]
    trainset_targets = trainset_targets[indices]

    trainset = MaskedSequenceDataset(trainset_inputs, trainset_targets, name=dataset_name+"_trainset")

    validset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["validset"]]))
    validset_targets = ArraySequence(chain(*[b.targets for b in bundles["validset"]]))
    validset = MaskedSequenceDataset(validset_inputs, validset_targets, name=dataset_name+"_validset")

    testset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["testset"]]))
    testset_targets = ArraySequence(chain(*[b.targets for b in bundles["testset"]]))
    testset = MaskedSequenceDataset(testset_inputs, testset_targets, name=dataset_name+"_testset")

    if classification:
        # Transform targets (directions) into class id.
        from dipy.data import get_sphere
        sphere = get_sphere("repulsion724")  # All possible directions (normed)
        sphere.vertices = sphere.vertices.astype(theano.config.floatX)

        # Target is the id of the closest direction on the sphere `sphere` determined using cosine similarity.
        # We do this for each point of the streamline and also each point of the reversed streamline.
        directions = sphere.vertices.astype(theano.config.floatX)
        for start in range(0, len(trainset.targets._data), batch_size):
            end = start + batch_size
            trainset.targets._data[start:end, 0] = np.argmax(np.dot(trainset.targets._data[start:end], directions.T), axis=-1)
            trainset.targets._data[start:end, 1] = np.argmax(np.dot(-trainset.targets._data[start:end], directions.T), axis=-1)

        for start in range(0, len(validset.targets._data), batch_size):
            end = start + batch_size
            validset.targets._data[start:end, 0] = np.argmax(np.dot(validset.targets._data[start:end], directions.T), axis=-1)
            validset.targets._data[start:end, 1] = np.argmax(np.dot(-validset.targets._data[start:end], directions.T), axis=-1)

        # Remove 3rd dimension.
        trainset.targets._data = trainset.targets._data[:, :2]
        validset.targets._data = validset.targets._data[:, :2]

    return trainset, validset, testset


def save_bundle(file, inputs, targets, indices=[]):
    """ Saves a bundle compatible with the learn2track framework.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    inputs : `nibabel.streamlines.ArraySequence` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.ArraySequence` object
        the direction leading from any 3D point to the next in every streamline.
    """
    np.savez(file,
             inputs_data=inputs._data,
             inputs_offsets=inputs._offsets,
             inputs_lengths=inputs._lengths,
             targets_data=targets._data,
             targets_offsets=targets._offsets,
             targets_lengths=targets._lengths,
             indices=indices
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
    inputs : `nibabel.streamlines.ArraySequence` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.ArraySequence` object
        the direction leading from any 3D point to the next in every streamline.
    """
    data = np.load(file)

    inputs = ArraySequence()
    inputs._data = data["inputs_data"]
    inputs._offsets = data["inputs_offsets"]
    inputs._lengths = data["inputs_lengths"]

    targets = ArraySequence()
    targets._data = data["targets_data"]
    targets._offsets = data["targets_offsets"]
    targets._lengths = data["targets_lengths"]

    return inputs, targets


vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


def load_text8():
    # http://mattmahoney.net/dc/textdata
    dataset_name = "Text8"

    datasets_repo = os.environ.get(DATASETS_ENV, './datasets')
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        filename = os.path.join(repo, 'text8.zip')
        if not os.path.isdir(repo) or not os.path.isfile(filename):
            os.mkdir(repo)

            import urllib.request
            filename, _ = urllib.request.urlretrieve('http://mattmahoney.net/dc/text8.zip', filename)

        import zipfile
        with zipfile.ZipFile(filename) as f:
            text = f.read(f.namelist()[0])
            text = np.array(list(map(char2id, map(chr, text))), dtype=np.int8)

        valid_size = 1000
        validset, trainset = text[:valid_size], text[valid_size:]
        np.savez(dataset_npy,
                 trainset_inputs=trainset,
                 validset_inputs=validset)

    data = np.load(dataset_npy)
    trainset = SequenceDataset(data['trainset_inputs'], data['trainset_inputs'], name="trainset")
    validset = SequenceDataset(data['validset_inputs'], data['trainset_inputs'], name="validset")
    trainset.vocabulary_size = vocabulary_size
    validset.vocabulary_size = vocabulary_size

    return trainset, validset


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt, newline=False):
        self.txt = txt
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.2f} sec.".format(time()-self.start))


def generate_uid_from_string(value):
    """ Creates unique identifier from a string. """
    return hashlib.sha256(value.encode()).hexdigest()


def map_coordinates_3d_4d(input_array, indices, affine=None, order=1):
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
        return map_coordinates(input_array, indices.T, order=order)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=order)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def eval_volume_at_3d_coordinates(volume, coords):
    """ Evaluates the volume data at the given coordinates using trilinear interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.

    Returns
    -------
    output : 2D array
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode="nearest")

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i],
                                         coords.T, order=1, mode="nearest")
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def normalize_dwi(dwi, bvals):
    """ Normalize dwi by the first b0.

    Parameters:
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion weighted images (4D).
    bvals : list of int
        B-values used with each direction. With this we can
        know which "directions" correspond to B0 images.

    Returns
    -------
    ndarray
        Diffusion weights normalized by the first B0.
    """
    # Indices of bvals sorted
    sorted_bvals_idx = np.argsort(bvals)
    nb_b0s = int(np.sum(bvals == 0))
    dwi_weights = dwi.get_data().astype("float32")

    # Get the first b0.
    b0 = dwi_weights[..., [sorted_bvals_idx[0]]]
    # Keep only b-value greater than 0
    weights = dwi_weights[..., sorted_bvals_idx[nb_b0s:]]
    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    weights = np.minimum(weights, b0)
    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed


def get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=8, smooth=0.006):
    """ Compute coefficients of the spherical harmonics basis.

    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.

    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    from dipy.core.sphere import HemiSphere
    from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

    # Start by normalizing DWI with the B0.
    weights = normalize_dwi(dwi, bvals)

    idx = np.where(bvals >= 1e-4)[0]  # Discard b-value == 0
    bvals = np.asarray(bvals)[idx]
    bvecs = np.asarray(bvecs)[idx]
    # bvecs *= np.array([1, -1, 1], dtype=np.float32)  # Debugging HACK

    # Assuming all directions are on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(weights, invB.T)
    return data_sh


def resample_dwi(dwi, bvals, bvecs, directions=None, sh_order=8, smooth=0.006):
    """ Resamples a diffusion signal according to a set of directions using spherical harmonics.

    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    directions : `dipy.core.sphere.Sphere` object, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be used.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.

    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    from dipy.data import get_sphere
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sph_harm_lookup

    data_sh = get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=sh_order, smooth=smooth)

    sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
    if directions is not None:
        sphere = Sphere(xyz=bvecs[1:])

    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)
    return data_resampled


# def resample_dwi_and_normalize(dwi, bvals, bvecs, directions=None, sh_order=8, smooth=0.006):
#     """ Resamples a diffusion signal according to a set of directions using spherical harmonics.

#     Parameters
#     -----------
#     dwi : `nibabel.NiftiImage` object
#         Diffusion signal as weighted images (4D).
#     bvals : ndarray shape (N,)
#         B-values used with each direction.
#     bvecs : ndarray shape (N, 3)
#         Directions of the diffusion signal. Directions are
#         assumed to be only on the hemisphere.
#     directions : `dipy.core.sphere.Sphere` object, optional
#         Directions the diffusion signal will be resampled to. Directions are
#         assumed to be on the whole sphere, not the hemisphere like bvecs.
#         If omitted, 100 directions evenly distributed on the sphere will be used.
#     sh_order : int, optional
#         SH order. Default: 4
#     smooth : float, optional
#         Lambda-regularization in the SH fit. Default: 0.006.

#     Returns
#     -------
#     ndarray
#         Diffusion weights resampled according to `sphere`.
#     """
#     from dipy.data import get_sphere
#     from dipy.core.sphere import Sphere, HemiSphere
#     from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

#     # Indices of bvals sorted
#     sorted_bvals_idx = np.argsort(bvals)
#     nb_b0s = int(np.sum(bvals == 0))
#     dwi_weights = dwi.get_data().astype("float32")

#     # Keep only b-value greater than 0
#     weights = dwi_weights[..., sorted_bvals_idx[nb_b0s:]]

#     idx = np.where(bvals >= 1e-4)[0]  # Discard b-value == 0
#     bvals = np.asarray(bvals)[idx]
#     bvecs = np.asarray(bvecs)[idx]
#     bvecs *= np.array([1, -1, 1], dtype=np.float32)

#     # Assuming all directions are on the hemisphere.
#     raw_sphere = HemiSphere(xyz=bvecs)

#     # Fit SH to signal
#     sph_harm_basis = sph_harm_lookup.get('mrtrix')
#     Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
#     L = -n * (n + 1)
#     invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
#     data_sh = np.dot(weights, invB.T)

#     sphere = get_sphere('repulsion100')
#     # sphere = get_sphere('repulsion724')
#     if directions is not None:
#         sphere = Sphere(xyz=bvecs[1:])

#     Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
#     weights_resampled = np.dot(data_sh, Ba.T)

#     # Remove mean and divided by standar deviation for non-zero value
#     mean = weights_resampled.mean(axis=-1)
#     weights_resampled_and_normalized = normalize_dwi(dwi, bvals)

#     return weights_resampled_and_normalized


def find_closest(sphere, xyz, normed=True):
    """
    Find the index of the vertex in the Sphere closest to the input vector

    Parameters
    ----------
    xyz : ndarray shape (N, 3)
        Input vector(s)
    normed : {True, False}, optional
        Normalized input vector(s).

    Return
    ------
    idx : ndarray shape (N,)
        The index/indices into the Sphere.vertices array that gives the closest
        vertex (in angle).
    """
    if normed:
        xyz = xyz / np.sqrt(np.sum(xyz**2, axis=1, dtype=float, keepdims=True)).astype(np.float32)

    cos_sim = np.abs(np.dot(sphere.vertices, xyz.T))
    return np.argmax(cos_sim, axis=0)


def logsumexp(x, axis=None, keepdims=False):
    max_value = T.max(x, axis=axis, keepdims=True)
    res = max_value + T.log(T.sum(T.exp(x-max_value), axis=axis, keepdims=True))
    if not keepdims:
        if axis is None:
            return T.squeeze(res)

        slices = [slice(None, None, None)]*res.ndim
        slices[axis] = 0  # Axis being merged
        return res[tuple(slices)]

    return res


def softmax(x, axis=None):
    return T.exp(x - logsumexp(x, axis=axis, keepdims=True))


def log_variables(batch_scheduler, *symb_vars):
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

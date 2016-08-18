import hashlib
import numpy as np
from collections import OrderedDict

from scipy.ndimage import map_coordinates

import nibabel as nib
from nibabel.streamlines import ArraySequence

from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv


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


class TractographyData(object):
    def __init__(self, signal, gradients):
        """
        Parameters
        ----------
        signal : :class:`nibabel.Nifti1Image` object
            Diffusion signal used to generate the streamlines.
        gradients : :class:`dipy.core.gradients.GradientTable` object
            Diffusion gradient information for the `signal`.
        """
        self.streamlines = nib.streamlines.ArraySequence()
        self.bundle_ids = np.zeros((0,), dtype=np.int16)
        self.name2id = OrderedDict()
        self.signal = signal
        self.gradients = gradients
        self.volume = signal.get_data()

    @property
    def bundle_names(self):
        return list(self.name2id.keys())

    def add(self, streamlines, bundle_name):
        """ Adds a bundle of streamlines to this container.

        Parameters
        ----------
        streamlines : `ArraySequence` object or list of 3D arrays
            Streamlines to be added.
        bundle_name : str
            Name of the bundle the streamlines belong to.
        """
        # Get bundle ID, create one if it's new bundle.
        if bundle_name not in self.name2id:
            self.name2id[bundle_name] = len(self.name2id)

        bundle_id = self.name2id[bundle_name]

        # Append streamlines
        self.streamlines.extend(streamlines)
        size = len(self.bundle_ids)
        new_size = size + len(streamlines)
        self.bundle_ids.resize((new_size,))
        self.bundle_ids[size:new_size] = bundle_id

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        streamlines_data = cls(data['signal'].item(), data['gradients'].item())
        streamlines_data.streamlines._data = data['coords']
        streamlines_data.streamlines._offsets = data['offsets']
        streamlines_data.streamlines._lengths = data['lengths']
        streamlines_data.bundle_ids = data['bundle_ids']
        streamlines_data.name2id = OrderedDict(data['name2id'])
        return streamlines_data

    def save(self, filename):
        np.savez(filename,
                 signal=self.signal,
                 gradients=self.gradients,
                 coords=self.streamlines._data.astype(np.float32),
                 offsets=self.streamlines._offsets,
                 lengths=self.streamlines._lengths.astype(np.int16),
                 bundle_ids=self.bundle_ids,
                 name2id=list(self.name2id.items()))


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
    data_sh = get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=sh_order, smooth=smooth)

    sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
    if directions is not None:
        sphere = Sphere(xyz=bvecs[1:])

    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)
    return data_resampled

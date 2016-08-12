from __future__ import division

import numpy as np
from os.path import join as pjoin

import theano
import theano.tensor as T
import nibabel as nib

from smartlearner import Dataset
from learn2track.utils import Timer
from learn2track import neurotools
from learn2track.neurotools import TractographyData

floatX = theano.config.floatX


class SequenceDataset(Dataset):
    """ Dataset interface.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.
    symb_targets : `theano.tensor.TensorType` object or None
        Symbolic variables representing the targets.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, targets=None, name="dataset"):
        """
        Parameters
        ----------
        inputs : list of ndarray
            Training examples (can be variable length sequences).
        targets : ndarray (optional)
            Target for each training example (can be variable length sequences).
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        self.name = name
        self.inputs = inputs
        self.targets = targets

        self.symb_inputs = T.TensorVariable(type=T.TensorType("floatX", [False]*(inputs[0].ndim+1)),
                                            name=self.name+'_symb_inputs')
        self.symb_inputs.tag.test_value = inputs[0][None, ...]  # For debugging Theano graphs.

        self.symb_targets = None
        if self.has_targets:
            self.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*(targets[0].ndim+1)),
                                                 name=self.name+'_symb_targets')
            self.symb_targets.tag.test_value = targets[0][None, ...]  # For debugging Theano graphs.

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets = value
        else:
            self._targets = None

    @property
    def has_targets(self):
        return self.targets is not None

    @property
    def input_shape(self):
        return self.inputs[0].shape

    @property
    def target_shape(self):
        if self.has_targets:
            return self.targets[0].shape

        return None

    @property
    def input_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.input_shape[-1]`.
        return self.input_shape[-1]

    @property
    def target_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.target_shape[-1]`.
        if self.has_targets:
            return self.target_shape[-1]

        return None

    def __len__(self):
        return len(self.inputs)


class MaskedSequenceDataset(SequenceDataset):
    """ Dataset interface.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.
    symb_targets : `theano.tensor.TensorType` object or None
        Symbolic variables representing the targets.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, targets=None, name="dataset"):
        """
        Parameters
        ----------
        inputs : list of ndarray
            Training examples (can be variable length sequences).
        targets : ndarray (optional)
            Target for each training example (can be variable length sequences).
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        super().__init__(inputs, targets, name)
        self.symb_mask = T.TensorVariable(type=T.TensorType("floatX", [False]*inputs[0].ndim),
                                          name=self.name+'_symb_mask')
        self.symb_mask.tag.test_value = (inputs[0][:, 0] > 0.5).astype(floatX)[None, ...]  # For debugging Theano graphs.


class TractographyDataset(MaskedSequenceDataset):
    def __init__(self, subjects, name="dataset"):
        """
        Parameters
        ----------
        subjects: list of TractogramData
        """
        self.subjects = subjects
        self.volumes = [subject.volume for subject in self.subjects]

        # Combine all tractograms in one.
        self.streamlines = nib.streamlines.ArraySequence()
        for i, subject in enumerate(self.subjects):
            self.streamlines.extend(subject.streamlines)

        super().__init__(self.streamlines, targets=None, name=name)

        # Build int2indices
        self.streamline_id_to_volume_id = np.nan * np.ones((len(self.streamlines),))

        start = 0
        for i, subject in enumerate(self.subjects):
            end = start + len(subject.streamlines)
            self.streamline_id_to_volume_id[start:end] = i
            start = end

        assert not np.isnan(self.streamline_id_to_volume_id.sum())
        self.streamline_id_to_volume_id = self.streamline_id_to_volume_id.astype(floatX)

        # self.streamlines = streamlines_data.streamlines
        # self.bundle_ids = streamlines_data.bundle_ids
        # self.bundle_names = streamlines_data.bundle_names
        # self.bundle_counts = np.bincount(self.bundle_ids)
        # self.bundle_indices = [np.where(self.bundle_ids == i)[0] for i in range(len(self.bundle_names))]
        # super().__init__(self.streamlines, targets=None, name=name)

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        return self.streamlines[idx], self.streamline_id_to_volume_id[idx]


def load_tractography_dataset(subject_files, name="HCP", use_sh_coeffs=False):
    subjects = []
    with Timer("  Loading subject(s)", newline=True):
        for subject_file in sorted(subject_files):
            print("    {}".format(subject_file))
            tracto_data = TractographyData.load(subject_file)

            dwi = tracto_data.signal
            bvals = tracto_data.gradients.bvals
            bvecs = tracto_data.gradients.bvecs
            if use_sh_coeffs:
                # Use 45 spherical harmonic coefficients to represent the diffusion signal.
                volume = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)
            else:
                # Resample the diffusion signal to have 100 directions.
                volume = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

            tracto_data.signal.uncache()  # Free some memory as we don't need the original signal.
            tracto_data.volume = volume
            subjects.append(tracto_data)

    return TractographyDataset(subjects, name)


# # Deprecated

# def load_streamlines_dataset(dwi_filename, streamlines_filename, name="ISMRM15_Challenge", use_sh_coeffs=False):
#     import nibabel as nib
#     from dipy.io.gradients import read_bvals_bvecs

#     with Timer("Loading DWI"):
#         # Load gradients table
#         bvals_filename = dwi_filename.split('.')[0] + ".bvals"
#         bvecs_filename = dwi_filename.split('.')[0] + ".bvecs"
#         bvals, bvecs = read_bvals_bvecs(bvals_filename, bvecs_filename)

#         dwi = nib.load(dwi_filename)
#         if use_sh_coeffs:
#             volume = get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)  # Use 45 spherical harmonic coefficients
#         else:
#             volume = resample_dwi(dwi, bvals, bvecs).astype(np.float32)  # Resample to 100 directions

#     with Timer("Loading streamlines"):
#         basename = streamlines_filename[:-len('.npz')]
#         if basename.endswith("_trainset"):
#             basename = basename[:-len("_trainset")]

#         trainset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_trainset.npz"), name=name+"_trainset")
#         validset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_validset.npz"), name=name+"_validset")
#         testset = StreamlinesDataset(volume, StreamlinesData.load(basename + "_testset.npz"), name=name+"_testset")

#     return trainset, validset, testset


# def load_ismrm2015_challenge(bundles_path, classification=False):
#     dataset_name = "ISMRM15_Challenge"

#     bundles = {'trainset': [], 'validset': [], 'testset': []}
#     for f in os.listdir(bundles_path):
#         if f.endswith("_trainset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["trainset"].append(dataset)
#         elif f.endswith("_validset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["validset"].append(dataset)
#         elif f.endswith("_testset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["testset"].append(dataset)

#     trainset = BundlesDataset(bundles["trainset"], name=dataset_name+"_trainset")

#     validset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["validset"]]))
#     validset_targets = ArraySequence(chain(*[b.targets for b in bundles["validset"]]))
#     validset = MaskedSequenceDataset(validset_inputs, validset_targets, name=dataset_name+"_validset")

#     testset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["testset"]]))
#     testset_targets = ArraySequence(chain(*[b.targets for b in bundles["testset"]]))
#     testset = MaskedSequenceDataset(testset_inputs, testset_targets, name=dataset_name+"_testset")

#     if classification:
#         # Transform targets (directions) into class id.
#         from dipy.data import get_sphere
#         sphere = get_sphere("repulsion724")  # All possible directions (normed)
#         sphere.vertices = sphere.vertices.astype(theano.config.floatX)

#         # Target is the id of the closest direction on the sphere `sphere` determined using cosine similarity.
#         # We do this for each point of the streamline and also each point of the reversed streamline.
#         directions = sphere.vertices.astype(theano.config.floatX)
#         for bundle in trainset.bundles:
#             bundle.targets._data = np.c_[np.argmax(np.dot(bundle.targets._data, directions.T), axis=-1)[:, None],
#                                          np.argmax(np.dot(-bundle.targets._data, directions.T), axis=-1)[:, None]].astype(theano.config.floatX)

#         validset.targets._data = np.c_[np.argmax(np.dot(validset.targets._data, directions.T), axis=-1)[:, None],
#                                        np.argmax(np.dot(-validset.targets._data, directions.T), axis=-1)[:, None]].astype(theano.config.floatX)

#     return trainset, validset, testset


# def load_ismrm2015_challenge_contiguous(bundles_path, classification=False, batch_size=1024*10):
#     dataset_name = "ISMRM15_Challenge"

#     bundles = {'trainset': [], 'validset': [], 'testset': []}
#     for f in os.listdir(bundles_path):
#         if f.endswith("_trainset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["trainset"].append(dataset)
#         elif f.endswith("_validset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["validset"].append(dataset)
#         elif f.endswith("_testset.npz"):
#             bundle_name = f.split("/")[-1][:-len(".npz")]
#             inputs, targets = load_bundle(pjoin(bundles_path, f))
#             dataset = MaskedSequenceDataset(inputs, targets, name=bundle_name)
#             bundles["testset"].append(dataset)

#     trainset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["trainset"]]))
#     trainset_targets = ArraySequence(chain(*[b.targets for b in bundles["trainset"]]))

#     # Shuffle streamlines
#     rng = np.random.RandomState(42)
#     indices = np.arange(len(trainset_inputs))
#     rng.shuffle(indices)
#     trainset_inputs = trainset_inputs[indices]
#     trainset_targets = trainset_targets[indices]

#     trainset = MaskedSequenceDataset(trainset_inputs, trainset_targets, name=dataset_name+"_trainset")

#     validset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["validset"]]))
#     validset_targets = ArraySequence(chain(*[b.targets for b in bundles["validset"]]))
#     validset = MaskedSequenceDataset(validset_inputs, validset_targets, name=dataset_name+"_validset")

#     testset_inputs = ArraySequence(chain(*[b.inputs for b in bundles["testset"]]))
#     testset_targets = ArraySequence(chain(*[b.targets for b in bundles["testset"]]))
#     testset = MaskedSequenceDataset(testset_inputs, testset_targets, name=dataset_name+"_testset")

#     if classification:
#         # Transform targets (directions) into class id.
#         from dipy.data import get_sphere
#         sphere = get_sphere("repulsion724")  # All possible directions (normed)
#         sphere.vertices = sphere.vertices.astype(theano.config.floatX)

#         # Target is the id of the closest direction on the sphere `sphere` determined using cosine similarity.
#         # We do this for each point of the streamline and also each point of the reversed streamline.
#         directions = sphere.vertices.astype(theano.config.floatX)
#         for start in range(0, len(trainset.targets._data), batch_size):
#             end = start + batch_size
#             trainset.targets._data[start:end, 0] = np.argmax(np.dot(trainset.targets._data[start:end], directions.T), axis=-1)
#             trainset.targets._data[start:end, 1] = np.argmax(np.dot(-trainset.targets._data[start:end], directions.T), axis=-1)

#         for start in range(0, len(validset.targets._data), batch_size):
#             end = start + batch_size
#             validset.targets._data[start:end, 0] = np.argmax(np.dot(validset.targets._data[start:end], directions.T), axis=-1)
#             validset.targets._data[start:end, 1] = np.argmax(np.dot(-validset.targets._data[start:end], directions.T), axis=-1)

#         # Remove 3rd dimension.
#         trainset.targets._data = trainset.targets._data[:, :2]
#         validset.targets._data = validset.targets._data[:, :2]

#     return trainset, validset, testset

from __future__ import division

import numpy as np

import theano
import theano.tensor as T
import nibabel as nib

from smartlearner import Dataset

floatX = theano.config.floatX


class ReconstructionDataset(Dataset):
    """ ReconstructionDataset interface.

    Behaves like a normal `Dataset` object but the targets are the inputs.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        inputs : ndarray
            Training examples
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        super().__init__(inputs, name=name, keep_on_cpu=keep_on_cpu)
        self._targets_shared = self._inputs_shared
        self.symb_targets = self.symb_inputs.copy()


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


class StreamlinesDataset(MaskedSequenceDataset):
    def __init__(self, volume, streamlines_data, name="dataset"):
        self.volume = volume
        self.streamlines = streamlines_data.streamlines
        self.bundle_ids = streamlines_data.bundle_ids
        self.bundle_names = streamlines_data.bundle_names
        self.bundle_counts = np.bincount(self.bundle_ids)
        self.bundle_indices = [np.where(self.bundle_ids == i)[0] for i in range(len(self.bundle_names))]

        super().__init__(self.streamlines, targets=None, name=name)


class TractogramsDataset(MaskedSequenceDataset):
    def __init__(self, subjects, name="dataset"):
        """
        Parameters
        ----------
        subjects: list of  (3d array, `StreamlinesData`) pairs.
        """
        self.subjects = subjects
        self.volumes, self.tractograms = list(zip(*self.subjects))

        # Combine all tractograms in one.
        self.streamlines = nib.streamlines.ArraySequence()
        for i, tractogram in enumerate(self.tractograms):
            self.streamlines.extend(tractogram.streamlines)

        super().__init__(self.streamlines, targets=None, name=name)

        # Build int2indices
        self.streamline_id_to_volume_id = np.nan * np.ones((len(self.streamlines),))

        start = 0
        for i, tractogram in enumerate(self.tractograms):
            end = start + len(tractogram.streamlines)
            self.streamline_id_to_volume_id[start:end] = i
            start = end

        assert not np.isnan(self.streamline_id_to_volume_id.sum())

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


class BundlesDataset(MaskedSequenceDataset):
    def __init__(self, bundles, name=""):
        """
        Parameters
        ----------
        bundles : list of `smartlearner.interfaces.dataset.Dataset` objects
        """
        super().__init__(bundles[0].inputs, bundles[0].targets, name=name)

        self.bundles = bundles
        self.bundles_size = list(map(len, self.bundles))

    def __len__(self):
        return sum(self.bundles_size)

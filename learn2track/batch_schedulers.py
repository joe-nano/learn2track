import theano
import numpy as np
import pickle
import theano.tensor as T
from os.path import join as pjoin
import itertools

from smartlearner.interfaces import BatchScheduler
from smartlearner.utils import sharedX

from learn2track import utils

floatX = theano.config.floatX


class StreamlinesBatchScheduler(BatchScheduler):
    """ Batch scheduler for streamlines dataset. """
    def __init__(self, dataset, batch_size, patch_shape=None, noisy_streamlines_sigma=None, nb_updates_per_epoch=None, seed=1234, include_last_point=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.patch_shape = patch_shape
        self.use_neighborhood_patch = self.patch_shape is not None
        if self.use_neighborhood_patch:
            self._patch_offset_idx = np.array(list(itertools.product(*[range(-(self.patch_shape//2), (self.patch_shape//2)+1)]*3)))

        self.include_last_point = include_last_point

        self.use_augment_by_flipping = True

        self._nb_updates_per_epoch = nb_updates_per_epoch
        self.use_sample_from_bundle = self._nb_updates_per_epoch is not None

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed+1)

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_mask = sharedX(np.ndarray((0, 0)))

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_mask.tag.test_value = batch_mask

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*(batch_targets.ndim)),
                                                         name=self.dataset.name+'_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        if self.use_neighborhood_patch:
            return self.dataset.volume.shape[-1] * int(self.patch_shape**3)

        return self.dataset.volume.shape[-1]  # Number of diffusion directions

    @property
    def target_size(self):
        return 3  # Direction to follow.

    @property
    def nb_updates_per_epoch(self):
        if self._nb_updates_per_epoch is None:
            return int(np.ceil(len(self.dataset) / self.batch_size))

        return self._nb_updates_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        # Compute the number of streamlines from each bundle that should be present in a batch.
        nb_bundles = int(np.sum(self.dataset.bundle_counts > 0))  # Keep only bundles that are present.
        nb_streamlines_from_each_bundle = (value//nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1

        self._nb_streamlines_from_each_bundle = np.zeros(len(self.dataset.bundle_names), dtype=int)
        self._nb_streamlines_from_each_bundle[self.dataset.bundle_counts > 0] = nb_streamlines_from_each_bundle
        assert sum(self._nb_streamlines_from_each_bundle) == self.batch_size

        # Pre-allocated memory for indices (speedup)
        self._indices = np.zeros((self._batch_size,), dtype=int)

    def _augment_data(self, inputs):
        pass

    def _add_noise_to_streamlines(self, streamlines):
        if not self.use_noisy_streamlines:
            return streamlines

        # Add gaussian noise, N(0, self.sigma).
        noisy_streamlines = streamlines.copy()
        shape = noisy_streamlines._data.shape
        noisy_streamlines._data += self.noisy_streamlines_sigma * self.rng_noise.randn(*shape)
        return noisy_streamlines

    def _add_neighborhood_patch(self, inputs, streamlines_pts):
        pts = (streamlines_pts[:, None] + self._patch_offset_idx).reshape((-1, 3))
        inputs = utils.eval_volume_at_3d_coordinates(self.dataset.volume, pts).reshape((len(streamlines_pts), -1))
        return inputs

    def _prepare_batch(self, indices):
        orig_streamlines = self.dataset.streamlines[indices].copy()
        streamlines = self._add_noise_to_streamlines(orig_streamlines)

        inputs = utils.eval_volume_at_3d_coordinates(self.dataset.volume, streamlines._data)
        if self.use_neighborhood_patch:
            inputs = self._add_neighborhood_patch(inputs, streamlines._data)

        targets = streamlines._data[1:] - streamlines._data[:-1]
        targets = targets / np.sqrt(np.sum(targets**2, axis=1, keepdims=True))

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        if self.include_last_point:  # only for the input

            max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
            batch_masks = np.zeros((batch_size, max_streamline_length-1), dtype=floatX)
            batch_inputs = np.zeros((batch_size, max_streamline_length, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, max_streamline_length-1, 3), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                batch_masks[i, :length-1] = 1
                batch_inputs[i, :length] = inputs[offset:offset+length]  # [0, 1, 2, 3, 4] => [0, 1, 2, 3, 4]
                batch_targets[i, :length-1] = targets[offset:offset+length-1]  # [1-0, 2-1, 3-2, 4-3] => [1-0, 2-1, 3-2, 4-3]

                if self.use_augment_by_flipping:
                    batch_masks[i+len(streamlines), :length-1] = 1
                    batch_inputs[i+len(streamlines), :length] = inputs[offset:offset+length][::-1]  # [0, 1, 2, 3, 4] => [4, 3, 2, 1, 0]
                    batch_targets[i+len(streamlines), :length-1] = -targets[offset:offset+length-1][::-1]  # [1-0, 2-1, 3-2, 4-3] => [4-3, 3-2, 2-1, 1-0]

        else:
            max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
            batch_masks = np.zeros((batch_size, max_streamline_length-1), dtype=floatX)
            batch_inputs = np.zeros((batch_size, max_streamline_length-1, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, max_streamline_length-1, 3), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                batch_masks[i, :length-1] = 1
                batch_inputs[i, :length-1] = inputs[offset:offset+length-1]  # [0, 1, 2, 3, 4] => [0, 1, 2, 3]
                batch_targets[i, :length-1] = targets[offset:offset+length-1]  # [1-0, 2-1, 3-2, 4-3] => [1-0, 2-1, 3-2, 4-3]

                if self.use_augment_by_flipping:
                    batch_masks[i+len(streamlines), :length-1] = 1
                    batch_inputs[i+len(streamlines), :length-1] = inputs[offset+1:offset+length][::-1]  # [0, 1, 2, 3, 4] => [4, 3, 2, 1]
                    batch_targets[i+len(streamlines), :length-1] = -targets[offset:offset+length-1][::-1]  # [1-0, 2-1, 3-2, 4-3] => [4-3, 3-2, 2-1, 1-0]

        return batch_inputs, batch_targets, batch_masks

    def _next_batch(self, batch_count):
        if not self.use_sample_from_bundle:
            # Simply take the next slice.
            start = batch_count * self.batch_size
            end = (batch_count + 1) * self.batch_size
            return self._prepare_batch(slice(start, end))

        # Batch is a stratified sample of streamlines from the different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)


            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def save(self, savedir):
        state = {"version": 1,
                 "batch_size": self.batch_size,
                 "nb_updates_per_epoch": self._nb_updates_per_epoch,
                 "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
                 "use_augment_by_flipping": self.use_augment_by_flipping,
                 "seed": self.seed,
                 "rng": pickle.dumps(self.rng),
                 "rng_noise": pickle.dumps(self.rng_noise),
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self._nb_updates_per_epoch = state["nb_updates_per_epoch"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])


class MultistepSequenceBatchScheduler(BatchScheduler):
    """ Multistep batch scheduler for streamlines dataset. """

    def __init__(self, dataset, batch_size, k, noisy_streamlines_sigma=None, nb_updates_per_epoch=None, seed=1234, include_last_point=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.k = k

        self.include_last_point = include_last_point

        self.use_augment_by_flipping = True

        self._nb_updates_per_epoch = nb_updates_per_epoch
        self.use_sample_from_bundle = self._nb_updates_per_epoch is not None

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed + 1)

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0, 0)))
        self._shared_batch_mask = sharedX(np.ndarray((0, 0)))

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_mask.tag.test_value = batch_mask

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_targets.ndim),
                                                         name=self.dataset.name + '_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        return self.dataset.volume.shape[-1]  # Number of diffusion directions

    @property
    def target_size(self):
        return 3  # Unnormalized direction vector

    @property
    def nb_updates_per_epoch(self):
        if self._nb_updates_per_epoch is None:
            return int(np.ceil(len(self.dataset) / self.batch_size))

        return self._nb_updates_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        # Compute the number of streamlines from each bundle that should be present in a batch.
        nb_bundles = int(np.sum(self.dataset.bundle_counts > 0))  # Keep only bundles that are present.
        nb_streamlines_from_each_bundle = (value // nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1

        self._nb_streamlines_from_each_bundle = np.zeros(len(self.dataset.bundle_names), dtype=int)
        self._nb_streamlines_from_each_bundle[self.dataset.bundle_counts > 0] = nb_streamlines_from_each_bundle
        assert sum(self._nb_streamlines_from_each_bundle) == self.batch_size

        # Pre-allocated memory for indices (speedup)
        self._indices = np.zeros((self._batch_size,), dtype=int)

    def _add_noise_to_streamlines(self, streamlines):
        if not self.use_noisy_streamlines:
            return streamlines

        # Add gaussian noise, N(0, self.sigma).
        noisy_streamlines = streamlines.copy()
        shape = noisy_streamlines._data.shape
        noisy_streamlines._data += self.noisy_streamlines_sigma * self.rng_noise.randn(*shape)
        return noisy_streamlines

    def _prepare_batch(self, indices):
        orig_streamlines = self.dataset.streamlines[indices].copy()
        streamlines = self._add_noise_to_streamlines(orig_streamlines)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        if self.include_last_point:  # only for the input
            raise NotImplementedError()

        else:
            max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
            batch_masks = np.zeros((batch_size, max_streamline_length - self.k), dtype=floatX)
            batch_inputs = np.zeros((batch_size, max_streamline_length - self.k, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, max_streamline_length - self.k, self.k, self.target_size), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                n = length - self.k
                batch_masks[i, :n] = 1
                batch_inputs[i, :n] = inputs[offset:offset + n]
                batch_targets[i, :n] = self._window_stack(targets[offset:offset + length - 1, None], self.k)

                if self.use_augment_by_flipping:
                    batch_masks[i + len(streamlines), :n] = 1
                    batch_inputs[i + len(streamlines), :n] = inputs[offset + self.k:offset + length][::-1]
                    batch_targets[i + len(streamlines), :n] = self._window_stack(-targets[offset:offset + length - 1, None][::-1], self.k)

        return batch_inputs, batch_targets, batch_masks

    @staticmethod
    def _window_stack(x, width):
        return np.hstack(x[i:1 + i - width or None] for i in range(width))

    def _next_batch(self, batch_count):
        if not self.use_sample_from_bundle:
            # Simply take the next slice.
            start = batch_count * self.batch_size
            end = (batch_count + 1) * self.batch_size
            return self._prepare_batch(slice(start, end))

        # Batch is a stratified sample of streamlines from the different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)

    @property
    def givens(self):
        return {
            self.dataset.symb_inputs: self._shared_batch_inputs,
            self.dataset.symb_targets: self._shared_batch_targets,
            self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def save(self, savedir):
        state = {
            "version": 1,
            "batch_size": self.batch_size,
            "k": self.k,
            "nb_updates_per_epoch": self._nb_updates_per_epoch,
            "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
            "use_augment_by_flipping": self.use_augment_by_flipping,
            "seed": self.seed,
            "rng": pickle.dumps(self.rng),
            "rng_noise": pickle.dumps(self.rng_noise)}

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.k = state["k"]
        self._nb_updates_per_epoch = state["nb_updates_per_epoch"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])


# OLD batch schedulers

class SequenceBatchScheduler(BatchScheduler):
    """ Batch of padded examples.
    """
    def __init__(self, dataset, batch_size, append_previous_direction=False, seed=1234):
        """
        Parameters
        ----------
        dataset : `SequenceDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        seed : int (optional)
            Seed of the random numbers generator used to sample different examples
            for each batch.
        """
        super().__init__(dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.append_previous_direction = append_previous_direction
        self._input_size = self.dataset.input_shape[1] + 3*self.append_previous_direction

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_targets.tag.test_value = batch_targets
        self.dataset.symb_mask.tag.test_value = batch_mask

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, 1, self._input_size)))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, 1, self.dataset.target_shape[1])))
        self._shared_batch_mask = sharedX(np.zeros((self.batch_size, 1)))

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size))

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def _next_batch(self, batch_count):
        # Compute max sequence length
        start = batch_count * self.batch_size
        end = (batch_count + 1) * self.batch_size
        inputs = self.dataset.inputs[start:end]
        targets = self.dataset.targets[start:end]
        max_sequence_length = max(map(len, inputs))

        # Pad sequences so that they have all the same length.
        current_batch_size = len(inputs)
        batch_mask = np.zeros((2*current_batch_size, max_sequence_length-1), dtype=floatX)
        batch_inputs = np.zeros((2*current_batch_size, max_sequence_length-1, self._input_size), dtype=floatX)
        batch_targets = np.zeros((2*current_batch_size, max_sequence_length-1, targets[0].shape[1]), dtype=floatX)

        for i, (x, y) in enumerate(zip(inputs, targets)):
            # No direction to predict for the last point, so we omit it.
            batch_mask[i, :len(x)-1] = 1
            batch_targets[i, :len(y)] = y
            if self.append_previous_direction:
                batch_inputs[i, :len(x)-1, :-3] = x[:-1]
                batch_inputs[i, 1:len(x)-1, -3:] = y[:-1]  # Direction of the previous timestep.
                # batch_inputs[i, 0, -3:] = (0, 0, 0)  # No previous direction for the first timestep.
                batch_inputs[i, 0, -3:] = y[0]
            else:
                batch_inputs[i, :len(x)-1] = x[:-1]

            # Flip version
            batch_mask[i+current_batch_size, :len(x)-1] = 1
            batch_targets[i+current_batch_size, :len(y)] = -y[::-1]
            if self.append_previous_direction:
                batch_inputs[i+current_batch_size, :len(x)-1, :-3] = x[::-1][:-1]
                batch_inputs[i+current_batch_size, 1:len(x)-1, -3:] = -y[::-1][:-1]  # Direction of the previous timestep.
                # batch_inputs[i+current_batch_size, 0, -3:] = (0, 0, 0)  # No previous direction for the first timestep.
                batch_inputs[i+current_batch_size, 0, -3:] = -y[::-1][0]
            else:
                batch_inputs[i+current_batch_size, :len(x)-1] = x[::-1][:-1]

        return batch_inputs, batch_targets, batch_mask

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    def save(self, savedir):
        state = {"version": 2,
                 "batch_size": self.batch_size,
                 "append_previous_direction": self.append_previous_direction,
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]


class BundlesBatchScheduler(BatchScheduler):
    """ Batch of examples are sampled proportionally from each bundle.
    """
    def __init__(self, bundles_dataset, batch_size, seed=1234):
        """
        Parameters
        ----------
        bundles_dataset : `BundlesDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        seed : int (optional)
            Seed of the random numbers generator used to sample different examples
            for each batch.
        """
        super(BundlesBatchScheduler, self).__init__(bundles_dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.shared_batch_count = theano.shared(np.array(0, dtype='i4'))

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # Test value
        self.shared_batch_count.tag.test_value = self.shared_batch_count.get_value()

        batch_inputs, batch_targets, batch_mask = self._next_batch()
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_targets.tag.test_value = batch_targets
        self.dataset.symb_mask.tag.test_value = batch_mask

        self.max_sequence_length = max([max(map(len, b.inputs))for b in self.dataset.bundles])

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, self.max_sequence_length, self.dataset.input_shape[1])))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, self.max_sequence_length, self.dataset.target_shape[1])))
        self._shared_batch_mask = sharedX(np.zeros((self.batch_size, self.max_sequence_length)))
        self._batch_inputs = np.empty_like(self._shared_batch_inputs.get_value())
        self._batch_targets = np.empty_like(self._shared_batch_targets.get_value())
        self._batch_mask = np.empty_like(self._shared_batch_mask.get_value())

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size/10.))

        # Compute the number of streamlines from each bundle that should be
        # present in a batch.
        nb_bundles = len(self.dataset.bundles)
        self.nb_streamlines_from_each_bundle = (value//nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        self.nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1
        assert sum(self.nb_streamlines_from_each_bundle) == self.batch_size

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def _next_batch(self):
        # Sample examples for next batch
        inputs = []
        targets = []
        for bundle, nb_streamlines in zip(self.dataset.bundles, self.nb_streamlines_from_each_bundle):
            indices = self.rng.permutation(len(bundle))[:nb_streamlines]
            inputs.extend(bundle.inputs[indices])
            targets.extend(bundle.targets[indices])

        # Compute max sequence length
        max_sequence_length = max(map(len, inputs))

        # Pad sequences so that they have all the same length.
        batch_mask = np.zeros((self.batch_size, max_sequence_length), dtype=floatX)
        batch_inputs = np.zeros((self.batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
        batch_targets = np.zeros((self.batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)
        flips = self.rng.randint(0, 2, size=len(inputs))

        for i, (x, y, flip) in enumerate(zip(inputs, targets, flips)):
            batch_mask[i, :len(x)-1] = 1
            if flip:
                batch_inputs[i, :len(x)-1] = x[::-1][:-1]  # No direction to predict for the last point.
                batch_targets[i, :len(y)] = -y[::-1]
            else:
                batch_inputs[i, :len(x)-1] = x[:-1]  # No direction to predict for the last point.
                batch_targets[i, :len(y)] = y

        return batch_inputs, batch_targets, batch_mask

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch()
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1

    def save(self, savedir):
        state = {"version": 1,
                 "seed": self.seed,
                 "batch_size": self.batch_size,
                 "nb_updates_per_epoch": self.nb_updates_per_epoch,
                 "shared_batch_count": self.shared_batch_count.get_value(),
                 "rng": pickle.dumps(self.rng)
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.shared_batch_count.set_value(state["shared_batch_count"])
        self.rng = pickle.loads(state["rng"])
        self.nb_updates_per_epoch = state["nb_updates_per_epoch"]


class SequenceBatchSchedulerWithClassTarget(BatchScheduler):
    """ Batch of padded examples.
    """
    def __init__(self, dataset, batch_size, seed=1234):
        """
        Parameters
        ----------
        dataset : `SequenceDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        seed : int (optional)
            Seed of the random numbers generator used to sample different examples
            for each batch.
        """
        super().__init__(dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_targets.tag.test_value = batch_targets
        self.dataset.symb_mask.tag.test_value = batch_mask

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, 1, self.dataset.input_shape[1])))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, 1, 1)))
        self._shared_batch_mask = sharedX(np.zeros((self.batch_size, 1)))

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size))

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def _next_batch(self, batch_count):
        # Compute max sequence length
        start = batch_count * self.batch_size
        end = (batch_count + 1) * self.batch_size
        inputs = self.dataset.inputs[start:end]
        targets = self.dataset.targets[start:end]
        max_sequence_length = max(map(len, inputs))

        # Pad sequences so that they have all the same length.
        current_batch_size = len(inputs)
        batch_mask = np.zeros((2*current_batch_size, max_sequence_length), dtype=floatX)
        batch_inputs = np.zeros((2*current_batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
        # batch_targets = np.zeros((2*current_batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)
        batch_targets = np.zeros((2*current_batch_size, max_sequence_length, 1), dtype=floatX)

        for i, (x, y) in enumerate(zip(inputs, targets)):
            # No direction to predict for the last point, so we omit it.
            batch_mask[i, :len(x)-1] = 1
            batch_inputs[i, :len(x)-1] = x[:-1]
            batch_targets[i, :len(y)] = y[:, [0]]

            # Flip version
            batch_mask[i+current_batch_size, :len(x)-1] = 1
            batch_inputs[i+current_batch_size, :len(x)-1] = x[::-1][:-1]
            batch_targets[i+current_batch_size, :len(y)] = y[::-1, [1]]

        return batch_inputs, batch_targets, batch_mask

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    def save(self, savedir):
        state = {"version": 1,
                 "batch_size": self.batch_size,
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]


class BundlesBatchSchedulerWithClassTarget(BatchScheduler):
    """ Batch of examples are sampled proportionally from each bundle.
    """
    def __init__(self, bundles_dataset, batch_size, seed=1234):
        """
        Parameters
        ----------
        bundles_dataset : `BundlesDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        seed : int (optional)
            Seed of the random numbers generator used to sample different examples
            for each batch.
        """
        super(BundlesBatchSchedulerWithClassTarget, self).__init__(bundles_dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.shared_batch_count = theano.shared(np.array(0, dtype='i4'))

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # Test value
        self.shared_batch_count.tag.test_value = self.shared_batch_count.get_value()

        batch_inputs, batch_targets, batch_mask = self._next_batch()
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_targets.tag.test_value = batch_targets
        self.dataset.symb_mask.tag.test_value = batch_mask

        self.max_sequence_length = max([max(map(len, b.inputs))for b in self.dataset.bundles])

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, self.max_sequence_length, self.dataset.input_shape[1])))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, self.max_sequence_length, 1)))
        self._shared_batch_mask = sharedX(np.zeros((self.batch_size, self.max_sequence_length)))
        self._batch_inputs = np.empty_like(self._shared_batch_inputs.get_value())
        self._batch_targets = np.empty_like(self._shared_batch_targets.get_value())
        self._batch_mask = np.empty_like(self._shared_batch_mask.get_value())

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))
        self.nb_updates_per_epoch = int(np.ceil(len(self.dataset)/self.batch_size/10.))

        # Compute the number of streamlines from each bundle that should be
        # present in a batch.
        nb_bundles = len(self.dataset.bundles)
        self.nb_streamlines_from_each_bundle = (value//nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        self.nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1
        assert sum(self.nb_streamlines_from_each_bundle) == self.batch_size

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def _next_batch(self):
        # Sample examples for next batch
        inputs = []
        targets = []
        for bundle, nb_streamlines in zip(self.dataset.bundles, self.nb_streamlines_from_each_bundle):
            indices = self.rng.permutation(len(bundle))[:nb_streamlines]
            inputs.extend(bundle.inputs[indices])
            targets.extend(bundle.targets[indices])

        # Compute max sequence length
        max_sequence_length = max(map(len, inputs))

        # Pad sequences so that they have all the same length.
        batch_mask = np.zeros((self.batch_size, max_sequence_length), dtype=floatX)
        batch_inputs = np.zeros((self.batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
        # batch_targets = np.zeros((self.batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)
        batch_targets = np.zeros((self.batch_size, max_sequence_length, 1), dtype=floatX)
        flips = self.rng.randint(0, 2, size=len(inputs))

        for i, (x, y, flip) in enumerate(zip(inputs, targets, flips)):
            batch_mask[i, :len(x)-1] = 1
            if flip:
                batch_inputs[i, :len(x)-1] = x[::-1][:-1]  # No direction to predict for the last point.
                batch_targets[i, :len(y)] = y[::-1, [1]]
            else:
                batch_inputs[i, :len(x)-1] = x[:-1]  # No direction to predict for the last point.
                batch_targets[i, :len(y)] = y[:, [0]]

        return batch_inputs, batch_targets, batch_mask

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch()
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1

    def save(self, savedir):
        state = {"version": 1,
                 "seed": self.seed,
                 "batch_size": self.batch_size,
                 "shared_batch_count": self.shared_batch_count.get_value(),
                 "rng": pickle.dumps(self.rng)
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.shared_batch_count.set_value(state["shared_batch_count"])
        self.rng = pickle.loads(state["rng"])


class ProportionalBundlesBatchScheduler(BatchScheduler):
    """ Batch of examples are sampled proportionally from each bundle.
    """
    def __init__(self, bundles_dataset, batch_size, nb_updates_per_epoch, seed=1234):
        """
        Parameters
        ----------
        bundles_dataset : `BundlesDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        nb_updates_per_epoch : int
            Number of updates to do each epoch.
        seed : int (optional)
            Seed of the random numbers generator used to sample different examples
            for each batch.
        """
        super(ProportionalBundlesBatchScheduler, self).__init__(bundles_dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.shared_batch_count = theano.shared(np.array(0, dtype='i4'))
        self.nb_updates_per_epoch = nb_updates_per_epoch

        self.shared_batch_count.tag.test_value = self.shared_batch_count.get_value()
        self.dataset.symb_inputs.tag.test_value = np.tile(self.dataset.symb_inputs.tag.test_value, (self.batch_size, 1, 1))
        self.dataset.symb_targets.tag.test_value = np.tile(self.dataset.symb_targets.tag.test_value, (self.batch_size, 1, 1))
        self.dataset.symb_mask.tag.test_value = np.tile(self.dataset.symb_mask.tag.test_value, (self.batch_size, 1))

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.max_sequence_length = 377

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, self.max_sequence_length, self.dataset.input_shape[1])))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, self.max_sequence_length, self.dataset.target_shape[1])))
        self._shared_batch_mask = sharedX(np.zeros((self.batch_size, self.max_sequence_length)))
        #self.batch_inputs = np.empty_like(self._shared_batch_inputs.get_value())
        #self.batch_targets = np.empty_like(self._shared_batch_targets.get_value())
        #self.batch_targets = np.empty_like(self._shared_batch_targets.get_value())

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))

        # Compute the number of streamlines from each bundle that should be
        # present in a batch.
        self.nb_streamlines_from_each_bundle = []
        nb_streamlines_total = len(self.dataset)
        for bundle_size in self.dataset.bundles_size:
            nb_streamlines = int(np.round(self.batch_size * (bundle_size / nb_streamlines_total)))
            # Make sure we got at least one streamline from each bundle.
            nb_streamlines = max(nb_streamlines, 1)
            self.nb_streamlines_from_each_bundle.append(nb_streamlines)

        # Make sure the splits sum to `batch_size`.
        self.nb_streamlines_from_each_bundle[-1] += self.batch_size - sum(self.nb_streamlines_from_each_bundle)
        assert sum(self.nb_streamlines_from_each_bundle) == self.batch_size

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            # Sample examples for next batch
            inputs = []
            targets = []
            for bundle, nb_streamlines in zip(self.dataset.bundles, self.nb_streamlines_from_each_bundle):
                indices = self.rng.permutation(len(bundle))[:nb_streamlines]
                inputs.extend(bundle.inputs[indices])
                targets.extend(bundle.targets[indices])

            # Compute max sequence length
            max_sequence_length = max(map(len, inputs))
            #max_sequence_length = self.max_sequence_length

            # Pad sequences so that they have all the same length.
            mask = np.zeros((self.batch_size, max_sequence_length), dtype=floatX)
            batch_inputs = np.zeros((self.batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
            batch_targets = np.zeros((self.batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)

            for i, (x, y) in enumerate(zip(inputs, targets)):
                mask[i, :len(x)] = 1
                batch_inputs[i, :len(x)] = x
                batch_targets[i, :len(y)] = y

            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(mask)

            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1


class SequenceBatchSchedulerText8(BatchScheduler):
    """ BatchScheduler specially design for the Text8 dataset. """

    def __init__(self, dataset, batch_size, sequence_length=10, nb_updates_per_epoch=100):
        """
        Parameters
        ----------
        dataset : `SequenceDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        """
        super().__init__(dataset)
        self._shared_batch_size = theano.shared(np.array(0, dtype='i4'))
        self.batch_size = batch_size
        self.sequence_length = sequence_length + 1
        self.nb_updates_per_epoch = nb_updates_per_epoch
        segment_length = len(self.dataset) // batch_size
        self._cursors = np.array([offset * segment_length for offset in range(batch_size)])

        # Redefine symbolic inputs
        self.dataset.symb_inputs = T.TensorVariable(type=T.TensorType("floatX", [False]*3),
                                                    name=self.dataset.name+'_symb_inputs')
        self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*3),
                                                     name=self.dataset.name+'_symb_targets')

        # Keep only `batch_size` examples as test values.
        self.dataset.symb_inputs.tag.test_value = self._next_batch()[:, :-1, :]
        self.dataset.symb_targets.tag.test_value = self._next_batch()[:, 1:, :]

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, sequence_length, self.dataset.vocabulary_size)))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size, sequence_length, self.dataset.vocabulary_size)))

    def _next_batch(self):
        # Make sure there are self.sequence_length characters available ahead of us.
        self._cursors[self._cursors + self.sequence_length > len(self.dataset)] %= len(self.dataset) - self.sequence_length

        batch = np.zeros(shape=(self.batch_size, self.sequence_length, self.dataset.vocabulary_size), dtype=theano.config.floatX)
        for i in range(self.sequence_length):
            batch[range(self.batch_size), i, self.dataset.inputs[self._cursors].astype(int)] = 1.0
            self._cursors += 1

        # self._cursors -= 1  # Overlap

        return batch

    @property
    def updates(self):
        return {}  # No updates

    @property
    def batch_size(self):
        return self._shared_batch_size.get_value()

    @batch_size.setter
    def batch_size(self, value):
        self._shared_batch_size.set_value(np.array(value, dtype='i4'))

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            inputs = self._next_batch()
            self._shared_batch_inputs.set_value(inputs[:, :-1, :])
            self._shared_batch_targets.set_value(inputs[:, 1:, :])
            yield batch_count + 1

    def save(self, savedir):
        state = {"version": 1,
                 "sequence_length": self.sequence_length,
                 "batch_size": self.batch_size,
                 "cursors": self._cursors
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self._cursors = state["cursors"]

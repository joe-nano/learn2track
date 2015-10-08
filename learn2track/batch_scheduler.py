import theano
import numpy as np

from smartlearner.interfaces import BatchScheduler
from smartlearner.utils import sharedX

floatX = theano.config.floatX


class SequenceBatchScheduler(BatchScheduler):
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

        # Keep only `batch_size` examples as test values.
        self.dataset.symb_inputs.tag.test_value = np.tile(self.dataset.symb_inputs.tag.test_value, (self.batch_size, 1, 1))
        if self.dataset.has_targets:
            self.dataset.symb_targets.tag.test_value = np.tile(self.dataset.symb_targets.tag.test_value, (self.batch_size, 1, 1))
        self.dataset.symb_mask.tag.test_value = np.tile(self.dataset.symb_mask.tag.test_value, (self.batch_size, 1))

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size, 1, self.dataset.input_shape[1])))
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

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            # Compute max sequence length
            start = batch_count * self.batch_size
            end = (batch_count + 1) * self.batch_size
            inputs = self.dataset.inputs[start:end]
            targets = self.dataset.targets[start:end]
            max_sequence_length = max(map(len, inputs))

            # Pad sequences so that they have all the same length.
            current_batch_size = len(inputs)
            mask = np.zeros((current_batch_size, max_sequence_length), dtype=floatX)
            batch_inputs = np.zeros((current_batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
            batch_targets = np.zeros((current_batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)

            for i, (x, y) in enumerate(zip(inputs, targets)):
                mask[i, :len(x)] = 1
                batch_inputs[i, :len(x)] = x
                batch_targets[i, :len(y)] = y

            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(mask)

            yield batch_count + 1


class BundlesBatchScheduler(BatchScheduler):
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
        super(BundlesBatchScheduler, self).__init__(bundles_dataset)
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

            # Pad sequences so that they have all the same length.
            batch_mask = np.zeros((self.batch_size, max_sequence_length), dtype=floatX)
            batch_inputs = np.zeros((self.batch_size, max_sequence_length) + inputs[0].shape[1:], dtype=floatX)
            batch_targets = np.zeros((self.batch_size, max_sequence_length) + targets[0].shape[1:], dtype=floatX)

            # Clear previous batch values
            # self._batch_mask[:, :] = 0
            # self._batch_inputs[:, :] = 0
            # self._batch_targets[:, :] = 0

            for i, (x, y) in enumerate(zip(inputs, targets)):
                # self._batch_mask[i, :len(x)] = 1
                # self._batch_inputs[i, :len(x)] = x
                # self._batch_targets[i, :len(y)] = y
                batch_mask[i, :len(x)] = 1
                batch_inputs[i, :len(x)] = x
                batch_targets[i, :len(y)] = y

            # self._shared_batch_inputs.set_value(self._batch_inputs)
            # self._shared_batch_targets.set_value(self._batch_targets)
            # self._shared_batch_mask.set_value(self._batch_mask)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1


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

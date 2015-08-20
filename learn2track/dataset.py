from __future__ import division

import numpy as np

import theano
import theano.tensor as T

from smartlearner import Dataset
from smartlearner.batch_schedulers import MiniBatchScheduler
from smartlearner.utils import sharedX


class BundlesDataset(Dataset):
    def __init__(self, bundles, name=""):
        """
        Parameters
        ----------
        bundles : list of `smartlearner.interfaces.dataset.Dataset` objects
        """
        super().__init__(bundles[0].inputs.get_value(), bundles[0].targets.get_value(), name)

        self.bundles = bundles
        self.bundles_size = list(map(len, self.bundles))

        # self.symb_inputs = self.bundles[0].symb_inputs.copy()
        # self.symb_inputs.name = self.name + '_inputs'
        # self.symb_targets = self.bundles[0].symb_targets.copy()
        # self.symb_targets.name = self.name + '_targets'

    @property
    def has_targets(self):
        return True

    @property
    def input_shape(self):
        return self.bundles[0].input_shape

    @property
    def target_shape(self):
        return self.bundles[0].target_shape

    def __len__(self):
        return sum(self.bundles_size)


class BundlesBatchScheduler(MiniBatchScheduler):
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
        super(BundlesBatchScheduler, self).__init__(bundles_dataset, batch_size)
        self.batch_size = batch_size
        self.nb_updates_per_epoch = nb_updates_per_epoch
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self._shared_batch_inputs = sharedX(np.zeros((self.batch_size,) + self.dataset.input_shape))
        self._shared_batch_targets = sharedX(np.zeros((self.batch_size,) + self.dataset.target_shape))
        self.batch_inputs = np.empty_like(self._shared_batch_inputs.get_value())
        self.batch_targets = np.empty_like(self._shared_batch_targets.get_value())

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
        if self.dataset.has_targets:
            return {self.dataset.symb_inputs: self._shared_batch_inputs,
                    self.dataset.symb_targets: self._shared_batch_targets}
        else:
            return {self.dataset.symb_inputs: self._shared_batch_inputs}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            # Sample examples for next batch
            start = 0
            for bundle, nb_streamlines in zip(self.dataset.bundles, self.nb_streamlines_from_each_bundle):
                end = start + nb_streamlines
                indices = self.rng.permutation(len(bundle))[:nb_streamlines]
                self.batch_inputs[start:end] = bundle.inputs.get_value()[indices]
                self.batch_targets[start:end] = bundle.targets.get_value()[indices]
                start = end

            self._shared_batch_inputs.set_value(self.batch_inputs)
            self._shared_batch_targets.set_value(self.batch_targets)

            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1

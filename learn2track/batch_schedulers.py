import theano
import numpy as np
import pickle
import theano.tensor as T
from os.path import join as pjoin

from dipy.tracking.streamline import set_number_of_points

from smartlearner.interfaces import BatchScheduler
from smartlearner.utils import sharedX

floatX = theano.config.floatX


class TractographyBatchScheduler(BatchScheduler):
    """ Batch scheduler for streamlines coming from multiple subjects. """

    def __init__(self, dataset, batch_size, noisy_streamlines_sigma=None, seed=1234, use_data_augment=True, normalize_target=False,
                 shuffle_streamlines=True, resample_streamlines=True, feed_previous_direction=False, sort_streamlines_by_length=False):
        """
        Parameters
        ----------
        dataset : :class:`TractographyDataset`
            Dataset from which to get the examples.
        batch_size : int
            Nb. of examples per batch.
        seed : int, optional
            Seed for the random generator when shuffling streamlines or adding noise to the streamlines.
        use_data_augment : bool
            If true, perform data augmentation by flipping streamlines.
        normalize_target : bool
            If true, targets will have a norm of one (usually used by the GruRegression model).
        shuffle_streamlines : bool
            Shuffle streamlines in the dataset between each epoch.
        resample_streamlines : bool
            Streamlines in a same batch will all have the same number of points.
            Should be always set to True for now (until the method _process_batch supports it).
        feed_previous_direction : bool
            Should the previous direction be appended to the input when making a prediction?
        sort_streamlines_by_length : bool
            Streamlines will be approximatively regrouped according to their length.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_augment_by_flipping = use_data_augment
        self.normalize_target = normalize_target

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed+1)
        self.shuffle_streamlines = shuffle_streamlines
        self.resample_streamlines = resample_streamlines
        self.sort_streamlines_by_length = sort_streamlines_by_length
        self.feed_previous_direction = feed_previous_direction
        
        # Sort streamlines according to their length by default.
        # This should speed up validation.
        self.indices = np.argsort(self.dataset.streamlines._lengths)

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
        # Number of diffusion directions or Spherical Harmonics (SH) coefficients
        return self.dataset.volumes[0].shape[-1]

    @property
    def target_size(self):
        return 3  # Direction to follow.

    @property
    def nb_updates_per_epoch(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

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

    def _prepare_batch(self, indices):
        orig_streamlines, volume_ids = self.dataset[indices]
        streamlines = self._add_noise_to_streamlines(orig_streamlines.copy())

        streamlines._lengths = streamlines._lengths.astype("int64")
        if self.resample_streamlines:
            # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamlines = set_number_of_points(streamlines, nb_points=streamline_length)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions
        if self.normalize_target:
            targets = targets / np.sqrt(np.sum(targets**2, axis=1, keepdims=True))  # Normalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
        batch_masks = np.zeros((batch_size, max_streamline_length-1), dtype=floatX)
        batch_inputs = np.zeros((batch_size, max_streamline_length-1, inputs.shape[1]), dtype=floatX)
        batch_targets = np.zeros((batch_size, max_streamline_length-1, 3), dtype=floatX)
        # batch_volume_ids = np.zeros((batch_size, max_streamline_length-1, 1), dtype=floatX)

        for i, (offset, length, volume_id) in enumerate(zip(streamlines._offsets, streamlines._lengths, volume_ids)):
            batch_masks[i, :length-1] = 1
            batch_inputs[i, :length-1] = inputs[offset:offset+length-1]  # [0, 1, 2, 3, 4] => [0, 1, 2, 3]
            batch_targets[i, :length-1] = targets[offset:offset+length-1]  # [1-0, 2-1, 3-2, 4-3] => [1-0, 2-1, 3-2, 4-3]
            # batch_volume_ids[i, :length-1] = volume_id

            if self.use_augment_by_flipping:
                batch_masks[i+len(streamlines), :length-1] = 1
                batch_inputs[i+len(streamlines), :length-1] = inputs[offset+1:offset+length][::-1]  # [0, 1, 2, 3, 4] => [4, 3, 2, 1]
                batch_targets[i+len(streamlines), :length-1] = -targets[offset:offset+length-1][::-1]  # [1-0, 2-1, 3-2, 4-3] => [4-3, 3-2, 2-1, 1-0]
                # batch_volume_ids[i+len(streamlines), :length-1] = volume_id

        batch_volume_ids = np.tile(volume_ids[:, None, None], (1 + self.use_augment_by_flipping, max_streamline_length-1, 1))
        batch_inputs = np.concatenate([batch_inputs, batch_volume_ids], axis=2)  # Streamlines coords + dwi ID

        if self.feed_previous_direction:
            previous_directions = np.concatenate([np.zeros((batch_size, 1, 3), dtype=floatX), batch_targets[:, :-1]], axis=1)
            previous_directions = previous_directions / np.sqrt(np.sum(previous_directions ** 2, axis=2, keepdims=True) + 1e-6)  # Normalized directions
            batch_inputs = np.concatenate([batch_inputs, previous_directions], axis=2)  # Streamlines coords + dwi ID + previous direction

        return batch_inputs, batch_targets, batch_masks

    def _next_batch(self, batch_count):
        # Simply take the next slice.
        start = batch_count * self.batch_size
        end = (batch_count + 1) * self.batch_size
        return self._prepare_batch(self.indices[slice(start, end)])

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        if self.shuffle_streamlines:
            self.rng.shuffle(self.indices)

        if self.sort_streamlines_by_length:
            lengths = self.dataset.streamlines._lengths
            step = len(lengths) // 100
            intervals = range(step, len(lengths), step)
            self.indices = np.argpartition(lengths, intervals)

        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def get_state(self):
        state = {"version": 4,
                 "batch_size": self.batch_size,
                 "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
                 "use_augment_by_flipping": self.use_augment_by_flipping,
                 "normalize_target": self.normalize_target,
                 "shuffle_streamlines": self.shuffle_streamlines,
                 "resample_streamlines": self.resample_streamlines,
                 "feed_previous_direction": self.feed_previous_direction,
                 "seed": self.seed,
                 "rng": pickle.dumps(self.rng),
                 "rng_noise": pickle.dumps(self.rng_noise),
                 "indices": self.indices,
                 "sort_streamlines_by_length": self.sort_streamlines_by_length
                 }
        return state

    def set_state(self, state):
        self.batch_size = state["batch_size"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])
        self.indices = state["indices"]
        if state["version"] < 2:
            self.normalize_target = True
        else:
            self.normalize_target = state["normalize_target"]

        if state["version"] < 3:
            self.shuffle_streamlines = state["shuffle_streamlines"]
            self.resample_streamlines = state["resample_streamlines"]

        if state["version"] < 4:
            self.feed_previous_direction = False
        else:
            self.feed_previous_direction = state["feed_previous_direction"]

        if state["version"] < 5:
            self.sort_streamlines_by_length = False
        else:
            self.sort_streamlines_by_length = state["sort_streamlines_by_length"]

    def save(self, savedir):
        state = self.get_state()
        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.set_state(state)


class TractographyBatchSchedulerWithProportionalSamplingFromBundles(BatchScheduler):
    """ TODO

    Right now, there are only a few snippets of code coming from
    https://github.com/MarcCote/learn2track/blob/aa87673e76062eb0254339fe9f30500037ec35fc/learn2track/batch_schedulers.py#L16
    but that would certainly need to be adapted to handle multiple subjects.

    Some things to verify/do:
     - make sure the i-th bundle from subject correspond to the i-th in another.
       It should be the case if the datasets where generated with the script
       `process_streamlines.py`.
     - adapt the dataset so it is easier to sample streamlines from a given bundle
       using its ID (and that across all subjects).

    """

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

    def _next_batch(self, batch_count):
        # Batch is a stratified sample of streamlines w.r.t. different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)


class TractographyBatchSchedulerWithProportionalSamplingFromSubjects(TractographyBatchScheduler):
    """ Batch scheduler for streamlines dataset, where streamlines are sampled proportionally from each subject
    """

    @property
    def indices_per_subject(self):
        if not hasattr(self, '_indices_per_subject'):
            subject_offsets = self.dataset.streamlines_per_sujet_offsets + [len(self.dataset)]
            self._indices_per_subject = [np.arange(subject_offsets[i], subject_offsets[i + 1]) for i in range(len(self.dataset.subjects))]
        return self._indices_per_subject

    @property
    def batch_size_per_subject(self):
        if not hasattr(self, '_batch_size_per_subject'):
            subject_relative_sizes = np.array(self.dataset.nb_streamlines_per_sujet) / sum(self.dataset.nb_streamlines_per_sujet)

            # Largest remainder method to make sure `np.sum(self._batch_size_per_subject) == self.batch_size`
            weighted_partition = subject_relative_sizes * self.batch_size
            remainder, discrete_partition = np.modf(weighted_partition)
            missing = int(self.batch_size - np.sum(discrete_partition))
            if missing > 0:
                sorted_remainder = np.argsort(remainder)
                discrete_partition[sorted_remainder[-missing:]] += 1
            self._batch_size_per_subject = discrete_partition.astype(int)

            assert np.sum(self._batch_size_per_subject) == self.batch_size, "batch_size_per_subject does not sum to batch_size: {}".format(
                self._batch_size_per_subject)
            assert np.all(self._batch_size_per_subject > 1), "Not all subjects will be present in a batch: {}".format(self._batch_size_per_subject)

        return self._batch_size_per_subject

    def _next_batch(self, batch_count):
        start_per_subject = batch_count * self.batch_size_per_subject
        end_per_subject = (batch_count + 1) * self.batch_size_per_subject
        batch_indices = []
        for i, (start, end) in enumerate(zip(start_per_subject, end_per_subject)):
            batch_indices.extend(self.indices_per_subject[i][slice(start, end)])

            if end > len(self.indices_per_subject[i]):
                # Not enough streamlines for subject i; sample from already seen streamlines
                needed = end - max(start, len(self.indices_per_subject[i]))
                sampled_ids = self.rng.randint(0, len(self.indices_per_subject[i]), needed)
                batch_indices.extend(self.indices_per_subject[i][sampled_ids])

        return self._prepare_batch(batch_indices)

    def __iter__(self):
        if self.shuffle_streamlines:
            for subject_indices in self.indices_per_subject:
                self.rng.shuffle(subject_indices)

        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1


class MultistepSequenceBatchScheduler(TractographyBatchSchedulerWithProportionalSamplingFromSubjects):
    """ Multistep batch scheduler for streamlines dataset.
    """

    def __init__(self, dataset, batch_size, k, noisy_streamlines_sigma=None, seed=1234, use_data_augment=True, normalize_target=False,
                 shuffle_streamlines=True, resample_streamlines=True, feed_previous_direction=False):
        """
        Parameters
        ----------
        dataset : :class:`TractographyDataset`
            Dataset from which to get the examples.
        batch_size : int
            Nb. of examples per batch.
        k : int
            Nb. of steps to predict.
        seed : int, optional
            Seed for the random generator when shuffling streamlines or adding noise to the streamlines.
        use_data_augment : bool
            If true, perform data augmentation by flipping streamlines.
        normalize_target : bool
            If true, targets will have a norm of one (usually used by the GruRegression model).
        shuffle_streamlines : bool
            Shuffle streamlines in the dataset between each epoch.
        resample_streamlines : bool
            Streamlines in a same batch will all have the same number of points.
            Should be always set to True for now (until the method _process_batch supports it).
        feed_previous_direction : bool
            Should the previous direction be appended to the input when making a prediction?
        """
        self.k = k
        super().__init__(dataset=dataset, batch_size=batch_size, noisy_streamlines_sigma=noisy_streamlines_sigma, seed=seed,
                         use_data_augment=use_data_augment, normalize_target=normalize_target, shuffle_streamlines=shuffle_streamlines,
                         resample_streamlines=resample_streamlines, feed_previous_direction=feed_previous_direction)

    @property
    def target_size(self):
        return 3  # Direction: X, Y, Z

    def _prepare_batch(self, indices):
        orig_streamlines, volume_ids = self.dataset[indices]
        streamlines = self._add_noise_to_streamlines(orig_streamlines.copy())

        streamlines._lengths = streamlines._lengths.astype("int64")
        if self.resample_streamlines:
            # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamlines = set_number_of_points(streamlines, nb_points=streamline_length)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions
        if self.normalize_target:
            targets = targets / np.sqrt(np.sum(targets**2, axis=1, keepdims=True))  # Normalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
        batch_masks = np.zeros((batch_size, max_streamline_length - self.k), dtype=floatX)
        batch_inputs = np.zeros((batch_size, max_streamline_length - self.k, inputs.shape[1]), dtype=floatX)
        batch_targets = np.zeros((batch_size, max_streamline_length - 1, self.target_size), dtype=floatX)

        for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
            n = length - self.k
            batch_masks[i, :n] = 1
            batch_inputs[i, :n] = inputs[offset:offset + n]
            batch_targets[i, :length - 1] = targets[offset:offset + length - 1]

            if self.use_augment_by_flipping:
                batch_masks[i + len(streamlines), :n] = 1
                batch_inputs[i + len(streamlines), :n] = inputs[offset + self.k:offset + length][::-1]
                batch_targets[i + len(streamlines), :length - 1] = -targets[offset:offset + length - 1][::-1]

        batch_volume_ids = np.tile(volume_ids[:, None, None], (1 + self.use_augment_by_flipping, max_streamline_length - self.k, 1))
        batch_inputs = np.concatenate([batch_inputs, batch_volume_ids], axis=2)  # Streamlines coords + dwi ID

        if self.feed_previous_direction:
            previous_directions = np.concatenate([np.zeros((batch_size, 1, 3), dtype=floatX), batch_targets[:, :-self.k]], axis=1)
            previous_directions = previous_directions / np.sqrt(np.sum(previous_directions ** 2, axis=2, keepdims=True))  # Normalized directions
            batch_inputs = np.concatenate([batch_inputs, previous_directions], axis=2)  # Streamlines coords + dwi ID + previous direction

        return batch_inputs, batch_targets, batch_masks

    def get_state(self):
        state = super().get_state()
        state["k"] = self.k
        return state

    def set_state(self, state):
        super().set_state(state)
        self.k = state["k"]


class MultistepSequenceBatchSchedulerWithoutMask(BatchScheduler):
    """ Multistep batch scheduler for streamlines dataset.

    The only difference with MultistepSequenceBatchScheduler is that we
    assume (forces) all streamline to have the same number of points
    (same seq_length) for a given batch. To do so, we resample the streamlines
    so they have the same number of points as the shortest (arbitrary choice)
    streamline.
    """

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

        # No need for a mask since streamlines are going to be resampled.
        self.dataset.symb_mask = None

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0, 0)))

        # Test value
        batch_inputs, batch_targets = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_targets.ndim),
                                                         name=self.dataset.name + '_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        return self.dataset.volume.shape[-1]  # Number of diffusion directions or coefficients of spherical harmonics

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

        # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
        streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
        streamlines._lengths = streamlines._lengths.astype("int64")
        streamlines = set_number_of_points(streamlines, nb_points=streamline_length)
        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        if self.include_last_point:  # only for the input
            raise NotImplementedError()

        else:
            batch_inputs = np.zeros((batch_size, streamline_length - self.k, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, streamline_length - self.k, self.k, self.target_size), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                n = length - self.k
                batch_inputs[i, :n] = inputs[offset:offset + n]
                batch_targets[i, :n] = self._window_stack(targets[offset:offset + length - 1, None], self.k)

                if self.use_augment_by_flipping:
                    batch_inputs[i + len(streamlines), :n] = inputs[offset + self.k:offset + length][::-1]
                    batch_targets[i + len(streamlines), :n] = self._window_stack(-targets[offset:offset + length - 1, None][::-1], self.k)

        return batch_inputs, batch_targets

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
            self.dataset.symb_targets: self._shared_batch_targets}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)

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


class SingleInputTractographyBatchScheduler(BatchScheduler):
    """
    Batch scheduler for streamlines coming from multiple subjects, where each data point must be fed as a single input to the model
    (not as sequences). Each batch will still select a number of streamlines, but will then split them as single coordinates to be given
    to the model.
    """

    def __init__(self, dataset, batch_size, noisy_streamlines_sigma=None, seed=1234, use_data_augment=True, normalize_target=False, shuffle_streamlines=True,
                 resample_streamlines=True, feed_previous_direction=False):
        """
        Parameters
        ----------
        dataset : :class:`TractographyDataset`
            Dataset from which to get the examples.
        batch_size : int
            Nb. of examples per batch.
        seed : int, optional
            Seed for the random generator when shuffling streamlines or adding noise to the streamlines.
        use_data_augment : bool
            If true, perform data augmentation by flipping streamlines.
        normalize_target : bool
            If true, targets will have a norm of one (usually used by the GruRegression model).
        shuffle_streamlines : bool
            Shuffle streamlines in the dataset between each epoch.
        resample_streamlines : bool
            Streamlines in a same batch will all have the same number of points.
            Should be always set to True for now (until the method _process_batch supports it).
        feed_previous_direction : bool
            Should the previous direction be appended to the input when making a prediction?
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.normalize_target = normalize_target

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        # Parameter use_data_augment cannot be used in the case of a FFNN model (or any other non-recurrent model,
        # because the targets are flipped but the inputs stay the same)
        self.use_augment_by_flipping = False
        if use_data_augment:
            print("WARNING: {} cannot use parameter use_data_augment and will ignore it.".format(type(self).__name__))

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed + 1)
        self.shuffle_streamlines = shuffle_streamlines
        self.resample_streamlines = resample_streamlines
        self.indices = np.arange(len(self.dataset))

        self.feed_previous_direction = feed_previous_direction

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0)))

        # Test value
        batch_inputs, batch_targets = self._next_batch(0)

        # Redefine symbolic variables for single input model
        self.dataset.symb_inputs = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_inputs.ndim),
                                                    name=self.dataset.name + '_symb_inputs')
        self.dataset.symb_inputs.tag.test_value = batch_inputs

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_targets.ndim),
                                                         name=self.dataset.name + '_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        # Number of diffusion directions or Spherical Harmonics (SH) coefficients
        return self.dataset.volumes[0].shape[-1]

    @property
    def target_size(self):
        return 3  # Direction to follow.

    @property
    def nb_updates_per_epoch(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

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

    def _prepare_batch(self, indices):
        orig_streamlines, volume_ids = self.dataset[indices]
        streamlines = self._add_noise_to_streamlines(orig_streamlines.copy())

        streamlines._lengths = streamlines._lengths.astype("int64")
        if self.resample_streamlines:
            # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamlines = set_number_of_points(streamlines, nb_points=streamline_length)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions
        if self.normalize_target:
            targets = targets / np.sqrt(np.sum(targets ** 2, axis=1, keepdims=True))  # Normalized directions

        actual_batch_size = sum(map(lambda x: len(x)-1, streamlines))
        if self.use_augment_by_flipping:
            half_batch_size = actual_batch_size
            actual_batch_size *= 2

        inputs_shape = inputs.shape[1]
        if self.feed_previous_direction:
            inputs_shape *= 2

        batch_inputs = np.zeros((actual_batch_size, inputs_shape), dtype=floatX)
        batch_targets = np.zeros((actual_batch_size, 3), dtype=floatX)
        batch_array_index = 0

        for i, (offset, length, volume_id) in enumerate(zip(streamlines._offsets, streamlines._lengths, volume_ids)):

            start = batch_array_index
            end = batch_array_index + (length - 1)
            batch_array_index += length - 1

            batch_inputs[start:end, :3] = inputs[offset:offset + length - 1]  # [0, 1, 2, 3, 4] => [0, 1, 2, 3]
            batch_targets[start:end] = targets[offset:offset + length - 1]  # [1-0, 2-1, 3-2, 4-3] => [1-0, 2-1, 3-2, 4-3]

            if self.feed_previous_direction:
                batch_inputs[start, 3:] = np.zeros((1, 3))
                previous_directions = batch_targets[start:end - 1]
                batch_inputs[start + 1:end, 3:] = previous_directions / np.sqrt(np.sum(previous_directions ** 2, axis=1, keepdims=True))  # Normalized directions

            if self.use_augment_by_flipping:
                flipped_start = start + half_batch_size
                flipped_end = end + half_batch_size
                batch_inputs[flipped_start:flipped_end] = inputs[offset + 1:offset + length][::-1]  # [0, 1, 2, 3, 4] => [4, 3, 2, 1]
                batch_targets[flipped_start:flipped_end] = -targets[offset:offset + length - 1][::-1]  # [1-0, 2-1, 3-2, 4-3] => [4-3, 3-2, 2-1, 1-0]

                if self.feed_previous_direction:
                    batch_inputs[flipped_start, 3:] = np.zeros((1, 3))
                    previous_directions = batch_targets[flipped_start:flipped_end - 1]
                    batch_inputs[flipped_start + 1:flipped_end, 3:] = previous_directions / np.sqrt(np.sum(previous_directions ** 2, axis=1, keepdims=True))  # Normalized directions

        batch_volume_ids = np.repeat(volume_ids, list(map(lambda x: len(x)-1, streamlines)))
        if self.use_augment_by_flipping:
            batch_volume_ids = np.tile(batch_volume_ids, [2])

        # Add dwi ID.
        if self.feed_previous_direction:
            batch_inputs = np.concatenate([batch_inputs[:, :3], batch_volume_ids[:, None], batch_inputs[:, 3:]], axis=1)  # Streamlines coords + dwi ID + previous direction
        else:
            batch_inputs = np.concatenate([batch_inputs, batch_volume_ids[:, None]], axis=1)  # Streamlines coords + dwi ID

        return batch_inputs, batch_targets

    def _next_batch(self, batch_count):
        # Simply take the next slice.
        start = batch_count * self.batch_size
        end = (batch_count + 1) * self.batch_size
        return self._prepare_batch(self.indices[slice(start, end)])

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets}

    def __iter__(self):
        if self.shuffle_streamlines:
            self.rng.shuffle(self.indices)

        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def get_state(self):
        state = {"version": 2,
                 "batch_size": self.batch_size,
                 "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
                 "use_augment_by_flipping": self.use_augment_by_flipping,
                 "normalize_target": self.normalize_target,
                 "shuffle_streamlines": self.shuffle_streamlines,
                 "resample_streamlines": self.resample_streamlines,
                 "feed_previous_direction": self.feed_previous_direction,
                 "seed": self.seed,
                 "rng": pickle.dumps(self.rng),
                 "rng_noise": pickle.dumps(self.rng_noise),
                 "indices": self.indices,
                 }
        return state

    def set_state(self, state):
        self.batch_size = state["batch_size"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])
        self.indices = state["indices"]
        self.normalize_target = state["normalize_target"]
        self.shuffle_streamlines = state["shuffle_streamlines"]
        self.resample_streamlines = state["resample_streamlines"]
        if state["version"] < 2:
            self.feed_previous_direction = False
        else:
            self.feed_previous_direction = state["feed_previous_direction"]

    def save(self, savedir):
        state = self.get_state()
        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.set_state(state)

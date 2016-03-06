import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import pickle
import shutil
import numpy as np
from os.path import join as pjoin
import argparse

import theano
import nibabel as nib

from time import sleep

import theano.tensor as T

from learn2track import utils
from learn2track.utils import Timer

from learn2track.dataset import StreamlinesDataset
from learn2track.batch_schedulers import StreamlinesBatchScheduler


def gen_streamlines_dataset(volume_shape, nb_streamlines):
    volume_shape = np.asarray(volume_shape)
    rng = np.random.RandomState(42)

    min_points, max_points = 10, 20
    streamlines = [rng.rand(rng.randint(min_points, max_points), 3) * volume_shape[:3]
                   for _ in range(nb_streamlines)]
    streamlines = nib.streamlines.ArraySequence(streamlines)

    volume = np.ones(volume_shape, dtype=np.float32)
    # volume = np.exp(-(volume_shape - volume_shape/2.)**2)
    # volume /= volume.max()

    nb_bundles = int(np.ceil(0.1 * nb_streamlines))
    bundle_ids = rng.randint(0, nb_bundles, size=(nb_streamlines,))
    bundle_names = ["Bundle #{}".format(i) for i in range(nb_bundles)]

    streamlines_data = utils.StreamlinesData(bundle_names)
    streamlines_data.streamlines = streamlines
    streamlines_data.bundle_ids = bundle_ids
    return StreamlinesDataset(volume, streamlines_data)


def test_streamlines_batch_scheduler():
    dataset = gen_streamlines_dataset(volume_shape=(10, 20, 30, 5), nb_streamlines=10)
    batch_size = 3
    batch_scheduler = StreamlinesBatchScheduler(dataset, batch_size=batch_size,
                                                noisy_streamlines_sigma=None,
                                                nb_updates_per_epoch=10)

    for batch_id in batch_scheduler:
        print("Batch", batch_id)

        print("Mask:", batch_scheduler._shared_batch_mask.get_value().sum(1))
        print("Inputs:", batch_scheduler._shared_batch_inputs.get_value().shape)
        print("Targets:", batch_scheduler._shared_batch_targets.get_value().shape)


if __name__ == "__main__":
    test_streamlines_batch_scheduler()

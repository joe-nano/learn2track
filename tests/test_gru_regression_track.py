import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
from types import SimpleNamespace

sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from scripts.track import make_is_outside_mask, make_is_too_long, make_is_too_curvy, make_is_stopping, STOPPING_MASK, STOPPING_LENGTH, STOPPING_CURVATURE, \
    batch_track, make_is_unlikely, STOPPING_LIKELIHOOD

import theano

from learn2track import neurotools, factories
from learn2track.utils import Timer
from tests.utils import make_dummy_dwi

import numpy as np


def test_gru_regression_track():
    hidden_sizes = 50

    with Timer("Creating dummy volume", newline=True):
        volume_manager = neurotools.VolumeManager()
        dwi, gradients = make_dummy_dwi(nb_gradients=30, volume_shape=(10, 10, 10), seed=1234)
        volume = neurotools.resample_dwi(dwi, gradients.bvals, gradients.bvecs).astype(np.float32)

        volume_manager.register(volume)

    with Timer("Creating model"):
        hyperparams = {'model': 'gru_regression',
                       'SGD': "1e-2",
                       'hidden_sizes': hidden_sizes,
                       'learn_to_stop': False,
                       'normalize': False,
                       'activation': 'tanh',
                       'feed_previous_direction': False,
                       'predict_offset': False,
                       'use_layer_normalization': False,
                       'drop_prob': 0.,
                       'use_zoneout': False,
                       'skip_connections': False,
                       'neighborhood_radius': None,
                       'nb_seeds_per_voxel': 2,
                       'step_size': 0.5,
                       'batch_size': 200,
                       'seed': 1234}
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=3,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    rng = np.random.RandomState(1234)
    mask = np.ones(volume.shape[:3])
    seeding_mask = np.random.randint(2, size=mask.shape)
    seeds = []
    indices = np.array(np.where(seeding_mask)).T
    for idx in indices:
        seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(hyperparams['nb_seeds_per_voxel'], 3))
        seeds.extend(seeds_in_voxel)
    seeds = np.array(seeds, dtype=theano.config.floatX)

    is_outside_mask = make_is_outside_mask(mask, np.eye(4), threshold=0.5)
    is_too_long = make_is_too_long(150)
    is_too_curvy = make_is_too_curvy(np.rad2deg(30))
    is_unlikely = make_is_unlikely(0.5)
    is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                    STOPPING_LENGTH: is_too_long,
                                    STOPPING_CURVATURE: is_too_curvy,
                                    STOPPING_LIKELIHOOD: is_unlikely})
    is_stopping.max_nb_points = 150

    args = SimpleNamespace()
    args.track_like_peter = False
    args.pft_nb_retry = 0
    args.pft_nb_backtrack_steps = 0
    args.use_max_component = False
    args.flip_x = False
    args.flip_y = False
    args.flip_z = False
    args.verbose = True

    tractogram = batch_track(model, volume, seeds,
                             step_size=hyperparams['step_size'],
                             is_stopping=is_stopping,
                             batch_size=hyperparams['batch_size'],
                             args=args)

    return True


def test_gru_regression_track_neighborhood():
    hidden_sizes = 50

    with Timer("Creating dummy volume", newline=True):
        volume_manager = neurotools.VolumeManager()
        dwi, gradients = make_dummy_dwi(nb_gradients=30, volume_shape=(10, 10, 10), seed=1234)
        volume = neurotools.resample_dwi(dwi, gradients.bvals, gradients.bvecs).astype(np.float32)

        volume_manager.register(volume)

    with Timer("Creating model"):
        hyperparams = {'model': 'gru_regression',
                       'SGD': "1e-2",
                       'hidden_sizes': hidden_sizes,
                       'learn_to_stop': False,
                       'normalize': False,
                       'activation': 'tanh',
                       'feed_previous_direction': False,
                       'predict_offset': False,
                       'use_layer_normalization': False,
                       'drop_prob': 0.,
                       'use_zoneout': False,
                       'skip_connections': False,
                       'neighborhood_radius': 0.5,
                       'nb_seeds_per_voxel': 2,
                       'step_size': 0.5,
                       'batch_size': 200,
                       'seed': 1234}
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=3,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    rng = np.random.RandomState(1234)
    mask = np.ones(volume.shape[:3])
    seeding_mask = np.random.randint(2, size=mask.shape)
    seeds = []
    indices = np.array(np.where(seeding_mask)).T
    for idx in indices:
        seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(hyperparams['nb_seeds_per_voxel'], 3))
        seeds.extend(seeds_in_voxel)
    seeds = np.array(seeds, dtype=theano.config.floatX)

    is_outside_mask = make_is_outside_mask(mask, np.eye(4), threshold=0.5)
    is_too_long = make_is_too_long(150)
    is_too_curvy = make_is_too_curvy(np.rad2deg(30))
    is_unlikely = make_is_unlikely(0.5)
    is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                    STOPPING_LENGTH: is_too_long,
                                    STOPPING_CURVATURE: is_too_curvy,
                                    STOPPING_LIKELIHOOD: is_unlikely})
    is_stopping.max_nb_points = 150

    args = SimpleNamespace()
    args.track_like_peter = False
    args.pft_nb_retry = 0
    args.pft_nb_backtrack_steps = 0
    args.use_max_component = False
    args.flip_x = False
    args.flip_y = False
    args.flip_z = False
    args.verbose = True

    tractogram = batch_track(model, volume, seeds,
                             step_size=hyperparams['step_size'],
                             is_stopping=is_stopping,
                             batch_size=hyperparams['batch_size'],
                             args=args)

    return True


def test_gru_regression_track_stopping():
    hidden_sizes = 50

    with Timer("Creating dummy volume", newline=True):
        volume_manager = neurotools.VolumeManager()
        dwi, gradients = make_dummy_dwi(nb_gradients=30, volume_shape=(10, 10, 10), seed=1234)
        volume = neurotools.resample_dwi(dwi, gradients.bvals, gradients.bvecs).astype(np.float32)

        volume_manager.register(volume)

    with Timer("Creating model"):
        hyperparams = {'model': 'gru_regression',
                       'SGD': "1e-2",
                       'hidden_sizes': hidden_sizes,
                       'learn_to_stop': True,
                       'normalize': False,
                       'activation': 'tanh',
                       'feed_previous_direction': False,
                       'predict_offset': False,
                       'use_layer_normalization': False,
                       'drop_prob': 0.,
                       'use_zoneout': False,
                       'skip_connections': False,
                       'neighborhood_radius': None,
                       'nb_seeds_per_voxel': 2,
                       'step_size': 0.5,
                       'batch_size': 200,
                       'seed': 1234}
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=3,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    rng = np.random.RandomState(1234)
    mask = np.ones(volume.shape[:3])
    seeding_mask = np.random.randint(2, size=mask.shape)
    seeds = []
    indices = np.array(np.where(seeding_mask)).T
    for idx in indices:
        seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(hyperparams['nb_seeds_per_voxel'], 3))
        seeds.extend(seeds_in_voxel)
    seeds = np.array(seeds, dtype=theano.config.floatX)

    is_outside_mask = make_is_outside_mask(mask, np.eye(4), threshold=0.5)
    is_too_long = make_is_too_long(150)
    is_too_curvy = make_is_too_curvy(np.rad2deg(30))
    is_unlikely = make_is_unlikely(0.5)
    is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                    STOPPING_LENGTH: is_too_long,
                                    STOPPING_CURVATURE: is_too_curvy,
                                    STOPPING_LIKELIHOOD: is_unlikely})
    is_stopping.max_nb_points = 150

    args = SimpleNamespace()
    args.track_like_peter = False
    args.pft_nb_retry = 0
    args.pft_nb_backtrack_steps = 0
    args.use_max_component = False
    args.flip_x = False
    args.flip_y = False
    args.flip_z = False
    args.verbose = True

    tractogram = batch_track(model, volume, seeds,
                             step_size=hyperparams['step_size'],
                             is_stopping=is_stopping,
                             batch_size=hyperparams['batch_size'],
                             args=args)

    return True


if __name__ == "__main__":
    test_gru_regression_track()
    test_gru_regression_track_neighborhood()
    test_gru_regression_track_stopping()

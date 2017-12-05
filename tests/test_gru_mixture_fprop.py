import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import theano

from learn2track import batch_schedulers, neurotools, factories
from learn2track.utils import Timer
from tests.utils import make_dummy_dataset


def test_gru_mixture_fprop():
    hyperparams = {'model': 'gru_mixture',
                   'SGD': "1e-2",
                   'hidden_sizes': 50,
                   'batch_size': 16,
                   'learn_to_stop': False,
                   'normalize': True,
                   'activation': 'tanh',
                   'feed_previous_direction': False,
                   'predict_offset': False,
                   'use_layer_normalization': False,
                   'drop_prob': 0.,
                   'use_zoneout': False,
                   'skip_connections': False,
                   'neighborhood_radius': None,
                   'noisy_streamlines_sigma': None,
                   'keep_step_size': True,
                   'sort_streamlines': False,
                   'n_gaussians': 2,
                   'seed': 1234}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, dataset=trainset)
        print ("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print(volume_manager.data_dimension, hyperparams['hidden_sizes'], batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))


    # Test fprop with missing streamlines from one subject in a batch
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)


    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                loss.loss,
                                updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)


    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)

    return True


def test_gru_mixture_fprop_neighborhood():
    hyperparams = {'model': 'gru_mixture',
                   'SGD': "1e-2",
                   'hidden_sizes': 50,
                   'batch_size': 16,
                   'learn_to_stop': False,
                   'normalize': True,
                   'activation': 'tanh',
                   'feed_previous_direction': False,
                   'predict_offset': False,
                   'use_layer_normalization': False,
                   'drop_prob': 0.,
                   'use_zoneout': False,
                   'skip_connections': False,
                   'seed': 1234,
                   'noisy_streamlines_sigma': None,
                   'keep_step_size': True,
                   'sort_streamlines': False,
                   'n_gaussians': 2,
                   'neighborhood_radius': 0.5}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, dataset=trainset)
        print ("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print (volume_manager.data_dimension, hyperparams['hidden_sizes'], batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

        print("Input size: {}".format(model.model_input_size))

    # Test fprop with missing streamlines from one subject in a batch
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)

    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                loss.loss,
                                updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)

    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)

    return True


def test_gru_mixture_fprop_stopping():
    hyperparams = {'model': 'gru_mixture',
                   'SGD': "1e-2",
                   'hidden_sizes': 50,
                   'batch_size': 16,
                   'learn_to_stop': True,
                   'normalize': True,
                   'activation': 'tanh',
                   'feed_previous_direction': False,
                   'predict_offset': False,
                   'use_layer_normalization': False,
                   'drop_prob': 0.,
                   'use_zoneout': False,
                   'skip_connections': False,
                   'seed': 1234,
                   'noisy_streamlines_sigma': None,
                   'keep_step_size': True,
                   'sort_streamlines': False,
                   'n_gaussians': 2,
                   'neighborhood_radius': None}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, dataset=trainset)
        print("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print(volume_manager.data_dimension, hyperparams['hidden_sizes'], batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams,
                                        input_size=volume_manager.data_dimension,
                                        output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

        print("Input size: {}".format(model.input_size))

    # Test fprop with missing streamlines from one subject in a batch
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)

    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                loss.loss,
                                updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)

    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask],
                                list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)

    return True

if __name__ == "__main__":
    test_gru_mixture_fprop()
    test_gru_mixture_fprop_neighborhood()
    test_gru_mixture_fprop_stopping()

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import theano

from learn2track import neurotools, factories
from learn2track.utils import Timer
from tests.utils import make_dummy_dataset


def test_gru_multistep_fprop_k1_single_subject():
    hidden_sizes = 50

    hyperparams = {
        'model': 'gru_multistep',
        'k': 1,
        'm': 1,
        'batch_size': 16,
        'SGD': "1e-2",
        'hidden_sizes': hidden_sizes,
        'learn_to_stop': False,
        'normalize': False,
        'noisy_streamlines_sigma': None,
        'shuffle_streamlines': True,
        'seed': 1234}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager, nb_subjects=1)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, trainset, train_mode=True)
        print("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print(volume_manager.data_dimension, hidden_sizes, batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams, input_size=volume_manager.data_dimension, output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    # Test fprop
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)

    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], loss.loss, updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)

    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)


def test_gru_multistep_fprop_k3():
    hidden_sizes = 50

    hyperparams = {
        'model': 'gru_multistep',
        'k': 3,
        'm': 3,
        'batch_size': 16,
        'SGD': "1e-2",
        'hidden_sizes': hidden_sizes,
        'learn_to_stop': False,
        'normalize': False,
        'noisy_streamlines_sigma': None,
        'shuffle_streamlines': True,
        'seed': 1234}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, trainset, train_mode=True)
        print("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print(volume_manager.data_dimension, hidden_sizes, batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams, input_size=volume_manager.data_dimension, output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    # Test fprop
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)

    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], loss.loss, updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)

    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)


def test_gru_multistep_fprop_k3_multidataset():
    hidden_sizes = 50

    hyperparams = {
        'model': 'gru_multistep',
        'k': 3,
        'm': 3,
        'batch_size': 16,
        'SGD': "1e-2",
        'hidden_sizes': hidden_sizes,
        'learn_to_stop': False,
        'normalize': False,
        'noisy_streamlines_sigma': None,
        'shuffle_streamlines': True,
        'seed': 1234}

    with Timer("Creating dataset", newline=True):
        volume_manager = neurotools.VolumeManager()
        trainset = make_dummy_dataset(volume_manager)
        validset = make_dummy_dataset(volume_manager)
        print("Dataset sizes:", len(trainset))

        batch_scheduler = factories.batch_scheduler_factory(hyperparams, trainset, train_mode=True)
        print("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print(volume_manager.data_dimension, hidden_sizes, batch_scheduler.target_size)

    with Timer("Creating model"):
        model = factories.model_factory(hyperparams, input_size=volume_manager.data_dimension, output_size=batch_scheduler.target_size,
                                        volume_manager=volume_manager)
        model.initialize(factories.weigths_initializer_factory("orthogonal", seed=1234))

    # Test fprop
    output = model.get_output(trainset.symb_inputs)
    fct = theano.function([trainset.symb_inputs], output, updates=model.graph_updates)

    batch_inputs, batch_targets, batch_mask = batch_scheduler._next_batch(2)
    out = fct(batch_inputs)

    with Timer("Building optimizer"):
        loss = factories.loss_factory(hyperparams, model, trainset)
        optimizer = factories.optimizer_factory(hyperparams, loss)

    fct_loss = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], loss.loss, updates=model.graph_updates)

    loss_value = fct_loss(batch_inputs, batch_targets, batch_mask)
    print("Loss:", loss_value)

    fct_optim = theano.function([trainset.symb_inputs, trainset.symb_targets, trainset.symb_mask], list(optimizer.directions.values()),
                                updates=model.graph_updates)

    dirs = fct_optim(batch_inputs, batch_targets, batch_mask)


if __name__ == "__main__":
    test_gru_multistep_fprop_k1_single_subject()
    print("*** Test passed: K1 (single subject) ***")

    test_gru_multistep_fprop_k3()
    print("*** Test passed: K3 ***")

    test_gru_multistep_fprop_k3_multidataset()
    print("*** Test passed: K3 (multidataset) ***")

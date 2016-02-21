#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this  script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import theano
import theano.tensor as T
import numpy as np

import unittest
import tempfile
from nose.tools import assert_true
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal
from numpy.testing import measure

import smartlearner.initializers as initer
from smartlearner import Trainer, Dataset, Model
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria

import smartlearner.initializers as initer
from smartlearner.utils import sharedX
from smartlearner.optimizers import Adam
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import MiniBatchScheduler, FullBatchScheduler
# from smartlearner.losses.distribution_losses import CrossEntropy

# from convnade.utils import Timer, cartesian
# from convnade.datasets import load_binarized_mnist

# from convnade import DeepConvNADE, DeepConvNADEBuilder
# from convnade import generate_blueprints
# #from convnade.tasks import DeepNadeOrderingTask
# from convnade.batch_schedulers import MiniBatchSchedulerWithAutoregressiveMask
# from convnade.losses import BinaryCrossEntropyEstimateWithAutoRegressiveMask

import smartlearner.initializers as initer

from learn2track.utils import load_text8, Timer
from learn2track.lstm import LayerLSTM, LayerLSTMFast
from learn2track.lstm import LSTM, LSTMFast, LSTMFaster
from learn2track.losses import SequenceNegativeLogLikelihood
from learn2track.batch_schedulers import SequenceBatchSchedulerText8


DATA = {}


def setup():
    global DATA
    batch_size = 64
    seq_len = 30
    hidden_size = 500
    features_size = 100
    rng = np.random.RandomState(42)
    DATA['rng'] = rng
    DATA['batch_size'] = batch_size
    DATA['seq_len'] = seq_len
    DATA['hidden_size'] = hidden_size
    DATA['features_size'] = features_size
    DATA['batch'] = rng.randn(batch_size, seq_len, features_size).astype(theano.config.floatX)
    DATA['mask'] = (np.arange(seq_len) <= rng.randint(10, seq_len, size=(batch_size,))[:, None]).astype(theano.config.floatX)
    DATA['batch_one_step'] = DATA['batch'][:, 0, :]
    DATA['state_h'] = rng.randn(batch_size, hidden_size)
    DATA['state_m'] = rng.randn(batch_size, hidden_size)


class TestLayerLSTM(unittest.TestCase):

    def test_fprop(self):
        activation = "tanh"
        seed = 1234
        repeat = 1000

        layer = LayerLSTM(input_size=DATA['features_size'],
                          hidden_size=DATA['hidden_size'],
                          activation=activation)

        layer.initialize(initer.UniformInitializer(seed))

        # input = T.tensor3('input')
        input = T.matrix('input')
        input.tag.test_value = DATA['batch_one_step']
        last_h = sharedX(DATA['state_h'])
        last_m = sharedX(DATA['state_m'])

        fprop = theano.function([input], layer.fprop_faster(input, last_h, last_m))
        fprop_time = measure("h, m = fprop(DATA['batch_one_step'])", repeat)
        print("fprop time: {:.2f} sec.", fprop_time)
        h, m = fprop(DATA['batch_one_step'])

    def test_fprop_faster(self):
        activation = "tanh"
        seed = 1234
        repeat = 1000

        layer = LayerLSTM(input_size=DATA['features_size'],
                          hidden_size=DATA['hidden_size'],
                          activation=activation)

        layer.initialize(initer.UniformInitializer(seed))

        layer_fast = LayerLSTMFast(input_size=DATA['features_size'],
                                   hidden_size=DATA['hidden_size'],
                                   activation=activation)

        # Wi, Wo, Wf, Wm
        layer_fast.W.set_value(np.concatenate([layer.Wi.get_value(), layer.Wo.get_value(), layer.Wf.get_value(), layer.Wm.get_value()], axis=1))
        layer_fast.U.set_value(np.concatenate([layer.Ui.get_value(), layer.Uo.get_value(), layer.Uf.get_value(), layer.Um.get_value()], axis=1))

        input = T.matrix('input')
        input.tag.test_value = DATA['batch_one_step']
        last_h = sharedX(DATA['state_h'])
        last_m = sharedX(DATA['state_m'])

        fprop = theano.function([input], layer.fprop(input, last_h, last_m))
        fprop_faster = theano.function([input], layer_fast.fprop(input, last_h, last_m))

        fprop_time = measure("h, m = fprop(DATA['batch_one_step'])", repeat)
        fprop_faster_time = measure("h, m = fprop_faster(DATA['batch_one_step'])", repeat)

        print("fprop time: {:.2f} sec.", fprop_time)
        print("fprop faster time: {:.2f} sec.", fprop_faster_time)
        print("Speedup: {:.2f}x".format(fprop_time/fprop_faster_time))

        for i in range(DATA['seq_len']):
            h1, m1 = fprop(DATA['batch'][:, i, :])
            h2, m2 = fprop_faster(DATA['batch'][:, i, :])
            assert_array_equal(h1, h2)
            assert_array_equal(m1, m2)


class TestLSTM(unittest.TestCase):

    def test_fprop_mask_vs_not_mask(self):
        activation = "tanh"
        seed = 1234
        repeat = 100

        lstm = LSTM(input_size=DATA['features_size'],
                    hidden_sizes=[DATA['hidden_size']],
                    )

        lstm.initialize(initer.UniformInitializer(seed))

        lstm2 = LSTMFast(input_size=DATA['features_size'],
                         hidden_sizes=[DATA['hidden_size']],
                         )
        lstm2.mask = sharedX(DATA['mask'])
        # Wi, Wo, Wf, Wm
        # Make sure the weights are the same.
        lstm2.layers_lstm[0].W.set_value(np.concatenate([lstm.layers_lstm[0].Wi.get_value(), lstm.layers_lstm[0].Wo.get_value(), lstm.layers_lstm[0].Wf.get_value(), lstm.layers_lstm[0].Wm.get_value()], axis=1))
        lstm2.layers_lstm[0].U.set_value(np.concatenate([lstm.layers_lstm[0].Ui.get_value(), lstm.layers_lstm[0].Uo.get_value(), lstm.layers_lstm[0].Uf.get_value(), lstm.layers_lstm[0].Um.get_value()], axis=1))

        input = T.tensor3('input')
        input.tag.test_value = DATA['batch']

        fprop = theano.function([input], lstm.get_output(input))
        fprop2 = theano.function([input], lstm2.get_output(input))
        # fprop_time = measure("out = fprop(DATA['batch'])", repeat)
        # print("fprop time: {:.2f} sec.", fprop_time)
        out = fprop(DATA['batch'])
        out2 = fprop2(DATA['batch'])

        assert_true(out.sum != out2.sum())
        assert_array_equal((out * DATA['mask'][:, :, None]),
                           (out2 * DATA['mask'][:, :, None]))

    def test_fprop_fast(self):
        seed = 1234
        repeat = 100

        lstm = LSTM(input_size=DATA['features_size'],
                    hidden_sizes=[DATA['hidden_size']],
                    )

        lstm.initialize(initer.UniformInitializer(seed))

        lstm2 = LSTMFast(input_size=DATA['features_size'],
                         hidden_sizes=[DATA['hidden_size']],
                         )
        # Wi, Wo, Wf, Wm
        # Make sure the weights are the same.
        lstm2.layers_lstm[0].W.set_value(np.concatenate([lstm.layers_lstm[0].Wi.get_value(), lstm.layers_lstm[0].Wo.get_value(), lstm.layers_lstm[0].Wf.get_value(), lstm.layers_lstm[0].Wm.get_value()], axis=1))
        lstm2.layers_lstm[0].U.set_value(np.concatenate([lstm.layers_lstm[0].Ui.get_value(), lstm.layers_lstm[0].Uo.get_value(), lstm.layers_lstm[0].Uf.get_value(), lstm.layers_lstm[0].Um.get_value()], axis=1))

        input = T.tensor3('input')
        input.tag.test_value = DATA['batch']

        fprop = theano.function([input], lstm.get_output(input))
        fprop2 = theano.function([input], lstm2.get_output(input))
        fprop_time = measure("out = fprop(DATA['batch'])", repeat)
        print("fprop time: {:.2f} sec.", fprop_time)
        fprop2_time = measure("out = fprop2(DATA['batch'])", repeat)
        print("fprop faster time: {:.2f} sec.", fprop2_time)
        print("Speedup: {:.2f}x".format(fprop_time/fprop2_time))

        out = fprop(DATA['batch'])
        out2 = fprop2(DATA['batch'])
        assert_array_equal(out, out2)

    def test_fprop_faster(self):
        seed = 1234
        repeat = 100

        lstm = LSTM(input_size=DATA['features_size'],
                    hidden_sizes=[DATA['hidden_size']],
                    )

        lstm.initialize(initer.UniformInitializer(seed))

        lstm2 = LSTMFaster(input_size=DATA['features_size'],
                           hidden_sizes=[DATA['hidden_size']],
                           )
        # Wi, Wo, Wf, Wm
        # Make sure the weights are the same.
        lstm2.layers_lstm[0].W.set_value(np.concatenate([lstm.layers_lstm[0].Wi.get_value(), lstm.layers_lstm[0].Wo.get_value(), lstm.layers_lstm[0].Wf.get_value(), lstm.layers_lstm[0].Wm.get_value()], axis=1))
        lstm2.layers_lstm[0].U.set_value(np.concatenate([lstm.layers_lstm[0].Ui.get_value(), lstm.layers_lstm[0].Uo.get_value(), lstm.layers_lstm[0].Uf.get_value(), lstm.layers_lstm[0].Um.get_value()], axis=1))

        input = T.tensor3('input')
        input.tag.test_value = DATA['batch']

        fprop = theano.function([input], lstm.get_output(input))
        fprop2 = theano.function([input], lstm2.get_output(input))
        fprop_time = measure("out = fprop(DATA['batch'])", repeat)
        print("fprop time: {:.2f} sec.", fprop_time)
        fprop2_time = measure("out = fprop2(DATA['batch'])", repeat)
        print("fprop faster time: {:.2f} sec.", fprop2_time)
        print("Speedup: {:.2f}x".format(fprop_time/fprop2_time))

        out = fprop(DATA['batch'])
        out2 = fprop2(DATA['batch'])
        assert_array_equal(out, out2)
        # assert_array_equal((out * DATA['mask'][:, :, None]),
        #                    (out2 * DATA['mask'][:, :, None]))


if __name__ == "__main__":
    setup()
    # TestLayerLSTM().test_fprop()
    # TestLayerLSTM().test_fprop_faster()
    # TestLSTM().test_fprop_mask_vs_not_mask()
    TestLSTM().test_fprop_fast()
    # TestLSTM().test_fprop_faster()

from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer
from smartlearner.interfaces import Loss

from learn2track.models.layers import LayerRegression, LayerDense
from learn2track.models import GRU
from learn2track.utils import l2distance


class GRU_RegressionAndBinaryClassification(GRU):
    """ A standard GRU model with both a regression output layer and
    binary classification output layer.

    The regression layer consists in fully connected layer (DenseLayer)
    whereas the binary classification layer consists in a fully connected
    layer with a sigmoid non-linearity to learn when to stop.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        output_size : int
            Number of units the regression layer should have.
        """
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.output_size)
        self.stopping_layer = LayerDense(self.hidden_sizes[-1] + input_size, 1, activation="sigmoid", name="stopping")

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)
        self.stopping_layer.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters + self.stopping_layer.parameters

    def _fprop(self, Xi, Xi_plus1, *args):
        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        regression_out = self.layer_regression.fprop(last_layer_h)
        stopping = self.stopping_layer.fprop(T.concatenate([last_layer_h, Xi_plus1], axis=1))

        return outputs + (regression_out, stopping)

    def get_output(self, X):
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [None, None],
                                       sequences=[{"input": T.transpose(X, axes=(1, 0, 2)),  # We want to scan over sequence elements, not the examples.
                                                   "taps": [0, 1]}],
                                       n_steps=X.shape[1]-1)

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.regression_out = T.transpose(results[-2], axes=(1, 0, 2))
        self.stopping = T.transpose(results[-1], axes=(1, 0, 2))

        return self.regression_out, self.stopping

    def use(self, X):
        directions = self.get_output(X)
        return directions


class L2DistancePlusBinaryCrossEntropyForSequences(Loss):
    """ Computes the L2 error of the regression part of the output and the
    binary cross-entropy for the classification part of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    This loss assumes the classification target at every time step is False
    except for the last time step where it is True.
    """
    def __init__(self, normalize_output=False):
        self.normalize_output = normalize_output

    def _get_updates(self):
        return {}  # There is no updates for L2DistancePlusBinaryCrossEntropyForSequences.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask
        regression_outputs, stopping = model_output

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=1e-8)

        # Regression part (next direction)
        # L2_errors_per_time_step.shape = (batch_size,)
        self.L2_errors_per_time_step = l2distance(regression_outputs, self.dataset.symb_targets)
        # avg_L2_error_per_seq.shape = (batch_size,)
        self.avg_L2_error_per_seq = T.sum(self.L2_errors_per_time_step*mask, axis=1) / T.sum(mask, axis=1)

        # Binary classification part (stopping criterion)
        lengths = T.sum(mask, axis=1)
        lengths_int = T.cast(lengths, dtype="int32")  # Mask values are floats.
        idx_examples = T.arange(mask.shape[0])
        # Create a mask that does not contain the last element of each sequence.
        smaller_mask = T.set_subtensor(mask[idx_examples, lengths_int-1], 0)

        # Compute cross-entropy for non-ending points.
        target = T.zeros(1)
        cross_entropy_not_ending = T.sum(T.nnet.binary_crossentropy(stopping, target)*smaller_mask[:, :, None], axis=[1, 2])

        # Compute cross-entropy for ending points.
        # We add a scaling factor because there is only one ending point per sequence whereas
        # there multiple non-ending points.
        target = T.ones(1)
        cross_entropy_ending = T.nnet.binary_crossentropy(stopping[idx_examples, lengths_int-1, 0], target) * (lengths-1)
        self.cross_entropy = (cross_entropy_not_ending + cross_entropy_ending) / lengths

        return self.avg_L2_error_per_seq + self.cross_entropy

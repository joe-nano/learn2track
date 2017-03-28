import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from learn2track.models import FFNN
from learn2track.models.layers import LayerDense
from smartlearner.interfaces.loss import Loss

from learn2track.utils import l2distance

floatX = theano.config.floatX


class FFNN_Classification(FFNN):
    """ A standard FFNN model with a classification (sigmoid) layer stacked on top of it.
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, use_layer_normalization=False, **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Use to evaluate the diffusion signal at specific coordinates.
        input_size : int
            Number of units each element X has.
        hidden_sizes : int, list of int
            Number of hidden units each FFNN layer should have.
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations
        """
        super().__init__(input_size, hidden_sizes, use_layer_normalization=use_layer_normalization)
        self.volume_manager = volume_manager
        self.output_size = 1  # Positive class probability

        self.layer_classification = LayerDense(self.hidden_sizes[-1], self.output_size, activation="sigmoid")

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_classification.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_classification.parameters

    def _fprop(self, Xi, *args):
        # Xi.shape : (batch_size, 4)
        # coords + dwi ID

        # coords : brain 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi[:, :4]

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = self.volume_manager.eval_at_coords(coords)

        layer_outputs = super()._fprop(data_at_coords)

        # Compute positive class probability
        classification_out = self.layer_classification.fprop(layer_outputs[-1])

        # Remove single-dimension from shape
        classification_out = classification_out[:, 0]

        return layer_outputs + (classification_out,)


class BinaryCrossEntropy(Loss):
    """ Computes the binary cross-entropy
    """
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def _get_updates(self):
        return {}  # No updates

    def _compute_losses(self, model_output):
        # model_output.shape : (batch_size,)

        # targets.shape : (batch_size,)  # 0/1
        targets = self.dataset.symb_targets

        return T.nnet.binary_crossentropy(model_output, targets)

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

    def __init__(self, volume_manager, input_size, hidden_sizes, **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Use to evaluate the diffusion signal at specific coordinates.
        input_size : int
            Number of units each element X has.
        hidden_sizes : int, list of int
            Number of hidden units each FFNN layer should have.
        """
        super().__init__(input_size, hidden_sizes)
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

        return layer_outputs + (classification_out,)

    def make_sequence_generator(self, subject_id=0, **_):
        """ Makes function that returns the class_probability for x_{t+1} for every
        sequence in the batch given x_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        """

        # Build the sequence generator as a theano function.
        symb_x_t = T.matrix(name="x_t")

        layer_outputs = self._fprop(symb_x_t)

        # predictions.shape : (batch_size, 1)
        predictions = layer_outputs[-1]

        f = theano.function(inputs=[symb_x_t], outputs=[predictions])

        def _gen(x_t, states, *args, **kwargs):
            """ Returns the positive class probability for x_{t+1} for every
                sequence in the batch given x_{t}.

            Parameters
            ----------
            x_t : ndarray with shape (batch_size, 3)
                coordinate (x, y, z).
            states : list of 2D array of shape (batch_size, hidden_size)
                Currrent states of the network.

            Returns
            -------
            class_probability : ndarray with shape (batch_size, 1)
                Positive class probability
            new_states : list of 2D array of shape (batch_size, hidden_size)
                Updated states of the network after seeing x_t.
            """
            # Append the DWI ID of each sequence after the 3D coordinates.
            subject_ids = np.array([subject_id] * len(x_t), dtype=floatX)[:, None]

            x_t = np.c_[x_t, subject_ids]

            results = f(x_t)
            class_probability = results[-1]

            # FFNN is not a recurrent network, return original states
            new_states = states

            return class_probability, new_states

        return _gen


class BinaryCrossEntropy(Loss):
    """ Computes the binary cross-entropy
    """
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def _get_updates(self):
        return {}  # No updates

    def _compute_losses(self, model_output):
        # model_output.shape : (batch_size, 1)

        # symb_targets.shape : (batch_size, 1)  # 0/1
        targets = self.dataset.symb_targets

        return T.squeeze(T.nnet.binary_crossentropy(model_output, targets))

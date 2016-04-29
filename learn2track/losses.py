import numpy as np
import theano
import theano.tensor as T

from smartlearner.interfaces import Loss
from learn2track.utils import softmax


class L2DistanceForSequences(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        regression_outputs = model_output
        # sum_sqr_error_per_item = T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2)
        # self.mean_sqr_error = T.sum(sum_sqr_error_per_item*mask, axis=1) / T.sum(mask, axis=1)
        self.L2_error_per_item = T.sqrt(T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2))
        self.mean_sqr_error = T.sum(self.L2_error_per_item*mask, axis=1) / T.sum(mask, axis=1)

        return self.mean_sqr_error


class L2DistanceWithBinaryCrossEntropy(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask
        regression_outputs, stopping = model_output

        # Regression part (next direction)
        self.L2_error_per_item = T.sqrt(T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2))
        self.mean_sqr_error = T.sum(self.L2_error_per_item*mask, axis=1) / T.sum(mask, axis=1)

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

        return self.mean_sqr_error + self.cross_entropy


class SequenceNegativeLogLikelihood(Loss):
    """
    Compute the Negative Log-Likelihood of sequence random variables.
    """
    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self._graph_updates = {}

    def _get_updates(self):
        return self._graph_updates

    def _compute_losses(self, model_output):

        def _nll(x, y):
            nll = -T.log(x)
            selected_nll = nll[T.arange(x.shape[0]), y]
            return selected_nll

        y_idx = T.cast(T.argmax(self.dataset.symb_targets, axis=2), dtype="int32")
        nlls, updates = theano.scan(fn=_nll,
                                    # outputs_info=[None],
                                    sequences=[T.transpose(model_output, axes=(1, 0, 2)),  # We want to scan over sequence elements, not the examples.
                                               T.transpose(y_idx, axes=(1, 0))])  # We want to scan over sequence elements, not the examples.

        self._graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        nlls = T.transpose(nlls, axes=(1, 0))
        return T.mean(nlls, axis=1)


class NegativeLogLikelihoodForSequences(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        nll = -T.log(model_output)  # Assume model has a softmax output, or equivalent
        indices = T.cast(self.dataset.symb_targets[:, :, 0], dtype="int32")  # Targets are floats.
        selected_nll = nll[T.arange(self.dataset.symb_targets.shape[0]), T.arange(self.dataset.symb_targets.shape[1]), indices]
        selected_nll_seq = T.mean(selected_nll*mask[:, :, None], axis=[1, 2])

        from ipdb import set_trace as dbg
        dbg()

        return selected_nll_seq


class NLLForSequenceOfDirections(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        target_directions = self.dataset.symb_targets
        # target_directions /= T.sqrt(T.sum(target_directions**2, axis=1, keepdims=True))

        from dipy.data import get_sphere
        sphere = get_sphere("repulsion100")  # All possible directions (normed)
        sphere.vertices = sphere.vertices.astype(theano.config.floatX)

        # Find the closest direction.
        cos_sim = T.dot(target_directions, sphere.vertices.T)
        targets = T.argmax(cos_sim, axis=-1)

        # Compute NLL
        model_directions = model_output  # Assume model outputs one 3D vector (normed) per sequence element.
        probs = softmax(T.dot(model_directions, sphere.vertices.T), axis=-1)
        nlls = -T.log(probs)

        indices = T.cast(targets, dtype="int32")  # Targets are floats.
        selected_nll = nlls[T.arange(self.dataset.symb_targets.shape[0])[:, None],
                            T.arange(self.dataset.symb_targets.shape[1])[None, :],
                            indices]
        selected_nll_seq = T.sum(selected_nll*mask, axis=1) / T.sum(mask, axis=1)

        return selected_nll_seq


class ErrorForSequenceOfDirections(Loss):
    """ Computes classfication error for learn2track.
    """
    def _get_updates(self):
        return {}  # There is no updates for ClassificationError.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        target_directions = self.dataset.symb_targets
        # target_directions /= T.sqrt(T.sum(target_directions**2, axis=1, keepdims=True))

        from dipy.data import get_sphere
        sphere = get_sphere("repulsion100")  # All possible directions (normed)
        sphere.vertices = sphere.vertices.astype(theano.config.floatX)

        # Find the closest direction.
        cos_sim = T.dot(target_directions, sphere.vertices.T)
        targets = T.argmax(cos_sim, axis=-1)

        # Compute probs
        model_directions = model_output  # Assume model outputs one 3D vector (normed) per sequence element.
        probs = softmax(T.dot(model_directions, sphere.vertices.T), axis=-1)

        predictions = T.argmax(probs, axis=2)
        errors = T.neq(predictions, targets)
        mean_errors_per_sequence = T.sum(errors*mask, axis=1) / T.sum(mask, axis=1)

        return mean_errors_per_sequence


class NLLForSequenceWithClassTarget(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        targets = self.dataset.symb_targets  # Class IDs.

        # Compute NLL
        probs = model_output  # Assume model outputs a vector of probabilities (i.e. softmax output layer).
        nlls = -T.log(probs)

        indices = T.cast(targets, dtype="int32")  # Targets are floats.
        selected_nll = nlls[T.arange(targets.shape[0])[:, None],
                            T.arange(targets.shape[1])[None, :],
                            indices[:, :, 0]]
        selected_nll_seq = T.sum(selected_nll*mask, axis=1) / T.sum(mask, axis=1)

        return selected_nll_seq


class ErrorForSequenceWithClassTarget(Loss):
    """ Computes classfication error for learn2track.
    """
    def _get_updates(self):
        return {}  # There is no updates for ClassificationError.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        targets = self.dataset.symb_targets

        # Compute probs
        probs = model_output  # Assume model outputs a vector of probabilities (i.e. softmax output layer).

        predictions = T.argmax(probs, axis=2)
        errors = T.neq(predictions, targets[:, :, 0])
        mean_errors_per_sequence = T.sum(errors*mask, axis=1) / T.sum(mask, axis=1)

        return mean_errors_per_sequence


class MultistepMultivariateGaussianLossForSequences(Loss):
    """ Compute multistep loss for a multivariate gaussian distribution using a monte-carlo estimate of the likelihood
    """

    def getstate(self):
        state = {"version": 1, "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _get_updates(self):
        return {}

    def _compute_losses(self, model_output):
        # model_output.shape : shape : (batch_size, seq_len, K, M, target_size, 2)
        # self.dataset.symb_targets.shape = (batch_size, seq_len, K, target_size)

        # mask.shape : (batch_size, seq_len)
        mask = self.dataset.symb_mask

        # mu.shape = (batch_size, seq_len, K, M, target_size)
        mu = model_output[:, :, :, :, :, 0]

        # sigma.shape = (batch_size, seq_len, K, M, target_size)
        sigma = model_output[:, :, :, :, :, 1]

        # targets.shape = (batch_size, seq_len, K, 1, target_size)
        targets = self.dataset.symb_targets[:, :, :, None, :]

        # Likelihood of multivariate gaussian (n dimensions) is :
        # ((2 \pi)^n |\Sigma|)^{-1/2} exp(-1/2 (x - \mu)^T \Sigma^-1 (x - \mu))
        # We suppose a diagonal covariance matrix, so we have :
        #   => |\Sigma| = \prod_n \sigma_n^2
        #   => (x - \mu)^T \Sigma^-1 (x - \mu) = \sum_n ((x_n - \mu_n) / \sigma_n)^2
        n = self.dataset.symb_targets.shape[3]
        likelihood = -0.5 * (n * np.log(2 * np.pi) + T.sum(2 * T.log(sigma) + T.sqr((targets - mu) / sigma), axis=4))
        max_term = T.max(likelihood, axis=3)
        m = model_output.shape[3]

        # nll.shape :(batch_size, seq_len, K)
        nll = T.log(m) - max_term - T.log(T.sum(T.exp(likelihood - max_term[:, :, :, None]), axis=3))

        # Return NLLs summed over K, meaned over sequence steps
        return T.sum(T.sum(nll, axis=2) * mask, axis=1) / T.sum(mask, axis=1)

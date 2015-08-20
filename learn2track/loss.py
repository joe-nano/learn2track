import theano.tensor as T

from smartlearner.interfaces import Loss


class L2DistanceForSequence(Loss):
    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_loss(self, model_output):
        mean_sqr_error = T.mean((model_output[:, :-1] - self.dataset.symb_targets)**2)
        ending_error = T.mean(model_output[:, -1]**2)  # Should be zero, i.e. no direction
        return mean_sqr_error + ending_error

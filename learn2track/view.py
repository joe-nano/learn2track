import numpy as np
import theano
import theano.tensor as T

from smartlearner.interfaces import View
from smartlearner import views


class LossView(View):
    def __init__(self, loss, batch_scheduler):
        super().__init__()

        self.compute_error = theano.function([],
                                             loss._loss,
                                             givens=batch_scheduler.givens,
                                             name="compute_error")


class RegressionError(View):
    def __init__(self, predict_fct, dataset, batch_size=100):
        super(RegressionError, self).__init__()

        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = dataset.symb_inputs
        target = dataset.symb_targets
        model_output = predict_fct(input)
        mean_sqr_error = T.mean(T.mean((model_output[:, :-1] - target)**2, axis=2), axis=1)
        ending_error = T.mean(model_output[:, -1]**2, axis=1)  # Should be zero, i.e. no direction
        regression_error = mean_sqr_error + ending_error

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_error = theano.function([no_batch],
                                             regression_error,
                                             givens=givens,
                                             name="compute_error")

    def update(self, status):
        errors = []
        for i in range(self.nb_batches):
            errors.append(self.compute_error(i))

        errors = np.concatenate(errors)
        return errors.mean(), errors.std(ddof=1) / np.sqrt(len(errors))

    @property
    def mean(self):
        return views.ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return views.ItemGetter(self, attribute=1)

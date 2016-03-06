import numpy as np
from os.path import join as pjoin

import theano

from smartlearner.interfaces import RecurrentTask
import smartlearner.utils as smartutils


class DecayingVariable(RecurrentTask):
    def __init__(self, decay_rate=0.99, name="decaying_variable", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = smartutils.sharedX(np.array(1), name=name)
        self.decay_rate = np.array(decay_rate, dtype=theano.config.floatX)

    def execute(self, status):
        self.var.set_value(self.var.get_value() * self.decay_rate)
        print("New decay: ", self.var.get_value())

    def save(self, path):
        state = {"version": 1,
                 "var": self.var.get_value()}
        smartutils.save_dict_to_json_file(pjoin(path, type(self).__name__ + ".json"), state)

    def load(self, path):
        state = smartutils.load_dict_from_json_file(pjoin(path, type(self).__name__ + ".json"))
        self.var.set_value(state["var"])

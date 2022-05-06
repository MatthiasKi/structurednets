import abc
import numpy as np

class Approximator:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass
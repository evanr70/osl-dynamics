"""Classes for other simulations.

"""

import numpy as np
from vrad.simulation import Simulation


class VRADSimulation(Simulation):
    """Simulate a dataset from the covariances and state time course of a model.

    Parameters
    ----------
    covariances : np.ndarray
        Covariances of the observation model
    state_time_course : np.ndarray
        Time x state array of activations.
    """

    def __init__(self, covariances: np.ndarray, state_time_course: np.ndarray):
        self.state_time_course = state_time_course
        super().__init__(
            n_samples=state_time_course.shape[0],
            n_channels=covariances.shape[-1],
            n_states=covariances.shape[0],
            zero_means=True,
            covariances=covariances,
            observation_error=0.0,
            random_covariance_weights=False,
            simulate=False,
        )

    def generate_states(self):
        return self.state_time_course

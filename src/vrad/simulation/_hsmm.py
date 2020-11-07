"""Classes for simulating Hidden Semi-Markov Models (HSMMs).

"""

import logging

import numpy as np
from vrad.array_ops import get_one_hot
from vrad.simulation import Simulation
from vrad.utils.decorators import auto_repr, auto_yaml

_logger = logging.getLogger("VRAD")


class HSMMSimulation(Simulation):
    """Hidden Semi-Markov Model Simulation.

    We sample the state using a transition probability matrix with zero
    probability for self-transitions. The lifetime of each state is sampled
    from a Gamma distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    n_states : int
        Number of states in the Markov chain.
    zero_means : bool
        Should means vary over channels and states?
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    covariances : numpy.ndarray
        Covariance matrix for each state in the observation model.
    gamma_shape : float
        Shape parameter for the gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of state lifetimes.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    random_covariance_weights : bool
        Randomly sample covariances.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    random_seed : int
        Seed for reproducibility.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        n_states: int,
        zero_means: bool,
        observation_error: float,
        covariances: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
        n_channels: int = None,
        random_covariance_weights: bool = False,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        random_seed: int = None,
        simulate: bool = True,
    ):
        if covariances is not None:
            n_channels = covariances.shape[1]

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob
        self.cumsum_off_diagonal_trans_prob = None

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            covariances=covariances,
            random_seed=random_seed,
            simulate=simulate,
        )

    def construct_off_diagonal_trans_prob(self):
        if self.off_diagonal_trans_prob is not None and (
            self.full_trans_prob is not None
        ):
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob must be "
                "specified. "
            )

        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_states, self.n_states])

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        self.cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob, axis=1
        )
        alpha_sim = np.zeros(self.n_samples, dtype=np.int)

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_state = self._rng.integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha_sim):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)
            alpha_sim[
                current_position : current_position + state_lifetime
            ] = current_state

            rand = random_sample()
            current_state = np.argmin(
                self.cumsum_off_diagonal_trans_prob[current_state] < rand
            )
            current_position += state_lifetime

        _logger.debug(f"n_states present in alpha sim = {len(np.unique(alpha_sim))}")

        one_hot_alpha_sim = get_one_hot(alpha_sim, n_states=self.n_states)

        _logger.debug(f"one_hot_alpha_sim.shape = {one_hot_alpha_sim.shape}")

        return one_hot_alpha_sim


class MixedHSMMSimulation(Simulation):
    """Hidden Semi-Markov Model Simulation with a mixture of states at each time point.

    Each mixture of states has it's own row/column in the transition probability matrix.
    The lifetime of each state mixture is sampled from a Gamma distribution.

    state_mixing_vectors is a 2D numpy array containing mixtures of the
    the states that can be simulated, e.g. with n_states=3 we could have
    state_mixing_vectors=[[0.5, 0.5, 0], [0.1, 0, 0.9]]

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    n_states : int
        Number of states in the Markov chain.
    mixed_state_vectors : np.ndarray
        2D array specifying the allowed state mixings.
    zero_means : bool
        Should means vary over channels and states?
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    covariances : numpy.ndarray
        Covariance matrix for each state in the observation model.
    gamma_shape : float
        Shape parameter for the gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of state lifetimes.
    n_channels : int
        Number of channels in the observation model. Inferred from covariances if None.
    random_covariance_weights : bool
        Randomly sample covariances.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of state transitions.
    random_seed : int
        Seed for reproducibility.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    """

    @auto_yaml
    @auto_repr
    def __init__(
        self,
        n_samples: int,
        n_states: int,
        mixed_state_vectors: np.ndarray,
        zero_means: bool,
        observation_error: float,
        covariances: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
        n_channels: int = None,
        random_covariance_weights: bool = False,
        off_diagonal_trans_prob: np.ndarray = None,
        random_seed: int = None,
        simulate: bool = True,
    ):
        if covariances is not None:
            n_channels = covariances.shape[1]

        if mixed_state_vectors.shape[1] != n_states:
            raise ValueError(f"Each state mixing vector must have {n_states} elements.")

        if np.any(np.sum(mixed_state_vectors, axis=1) != 1):
            raise ValueError("Each state mixing vector must sum to one.")

        self.mixed_state_vectors = mixed_state_vectors
        self.n_mixed_states = mixed_state_vectors.shape[0]

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.cumsum_off_diagonal_trans_prob = None

        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        self.construct_state_vectors(n_states)

        super().__init__(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            zero_means=zero_means,
            random_covariance_weights=random_covariance_weights,
            observation_error=observation_error,
            covariances=covariances,
            random_seed=random_seed,
            simulate=simulate,
        )

    def construct_state_vectors(self, n_states):
        non_mixed_state_vectors = get_one_hot(np.arange(n_states))
        self.state_vectors = np.append(
            non_mixed_state_vectors, self.mixed_state_vectors, axis=0
        )

    def construct_off_diagonal_trans_prob(self):
        if self.off_diagonal_trans_prob is None:
            self.off_diagonal_trans_prob = np.ones(
                [
                    self.n_states + self.n_mixed_states,
                    self.n_states + self.n_mixed_states,
                ]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_states(self):
        self.construct_off_diagonal_trans_prob()
        self.cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob, axis=1
        )
        alpha_sim = np.zeros([self.n_samples, self.n_states])

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_state = self._rng.integers(0, self.n_states)
        current_position = 0

        while current_position < len(alpha_sim):
            state_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)

            alpha_sim[
                current_position : current_position + state_lifetime
            ] = self.state_vectors[current_state]

            rand = random_sample()
            current_state = np.argmin(
                self.cumsum_off_diagonal_trans_prob[current_state] < rand
            )
            current_position += state_lifetime

        return alpha_sim

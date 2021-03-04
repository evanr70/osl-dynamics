"""Class for a Gaussian observation model.

"""

from tensorflow.keras import Model, layers, optimizers
from vrad import models
from vrad.inference.losses import LogLikelihoodLoss
from vrad.models.layers import LogLikelihoodLayer, MARParametersLayer, MARMeanCovLayer
from vrad.utils.misc import replace_argument


class MARO(models.Base):
    """Multivariate Autoregressive Observations (MARO) model.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    n_lags : int
        Order of the multivariate autoregressive observation model.
    learning_rate : float
        Learning rate for updating model parameters/weights.
    multi_gpu : bool
        Should be use multiple GPUs for training? Optional.
    strategy : str
        Strategy for distributed learning. Optional.
    """

    def __init__(
        self,
        n_states: int,
        n_channels: int,
        sequence_length: int,
        n_lags: int,
        learning_rate: float,
        multi_gpu: bool = False,
        strategy: str = None,
    ):
        # Parameters related to the observation model
        self.n_lags = n_lags

        # Initialise the model base class
        # This will build and compile the keras model
        super().__init__(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            multi_gpu=multi_gpu,
            strategy=strategy,
        )

    def build_model(self):
        """Builds a keras model."""
        self.model = _model_structure(
            n_states=self.n_states,
            n_channels=self.n_channels,
            sequence_length=self.sequence_length,
            n_lags=self.n_lags,
        )

    def compile(self):
        """Wrapper for the standard keras compile method.

        Sets up the optimizer and loss functions.
        """
        # Setup optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # Loss functions
        ll_loss = LogLikelihoodLoss()

        # Compile
        self.model.compile(optimizer=optimizer, loss=[ll_loss])

    def fit(
        self,
        *args,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                True,
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_model(self):
        """Reset the model as if you've built a new model.

        Resets the model weights, optimizer and annealing factor.
        """
        self.compile()
        initializers.reinitialize_model_weights(self.model)


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_lags: int,
):
    """Model structure.

    Parameters
    ----------
    n_states : int
        Numeber of states.
    n_channels : int
        Number of channels.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    n_lags : int
        Order of the multivariate autoregressive observation model.

    Returns
    -------
    tensorflow.keras.Model
        Keras model built using the functional API.
    """

    # Layers for inputs
    data = layers.Input(shape=(sequence_length, n_channels), name="data")
    alpha_t = layers.Input(shape=(sequence_length, n_states), name="alpha_t")

    # Observation model:
    # - We use x_t ~ N(mu_t, sigma_t), where
    #      - mu_t = Sum_j Sum_l alpha_jt W_jt x_{t-l}.
    #      - sigma_t = Sum_j alpha^2_jt sigma_jt, where sigma_jt is a learnable
    #        diagonal covariance matrix.
    # - We calculate the likelihood of generating the training data with alpha_t
    #   and the observation model.

    # Definition of layers
    mar_params_layer = MARParametersLayer(
        n_states, n_channels, n_lags, name="mar_params"
    )
    mean_cov_layer = MARMeanCovLayer(
        n_states, n_channels, sequence_length, n_lags, name="mean_cov"
    )
    ll_loss_layer = LogLikelihoodLayer(name="ll")

    # Data flow
    W_jt, sigma_j = mar_params_layer(data)  # data not used
    clipped_data, mu_t, sigma_t = mean_cov_layer([alpha_t, data, W_jt, sigma_j])
    ll_loss = ll_loss_layer([clipped_data, mu_t, sigma_t])

    return Model(inputs=[data, alpha_t], outputs=[ll_loss])

"""Inference network and generative model.

"""

import numpy as np
from tensorflow.keras import Model, layers, models, optimizers
from tensorflow.python import Variable, zeros
from tensorflow.python.distribute.distribution_strategy_context import get_strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from vrad.inference.callbacks import AnnealingCallback, BurninCallback
from vrad.inference.functions import cholesky_factor, cholesky_factor_to_full_matrix
from vrad.inference.initializers import reinitialize_model_weights
from vrad.inference.layers import (
    InferenceRNNLayers,
    KLDivergenceLayer,
    LogLikelihoodLayer,
    MixMeansCovsLayer,
    ModelRNNLayers,
    MultivariateNormalLayer,
    ReparameterizationLayer,
    TrainableVariablesLayer,
)
from vrad.utils.misc import listify


def create_model(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    learn_means: bool,
    learn_covariances: bool,
    initial_means: np.ndarray,
    initial_covariances: np.ndarray,
    n_layers_inference: int,
    n_layers_model: int,
    n_units_inference: int,
    n_units_model: int,
    dropout_rate_inference: float,
    dropout_rate_model: float,
    alpha_xform: str,
    learn_alpha_scaling: bool,
    do_annealing: bool,
    annealing_sharpness: float,
    n_epochs_annealing: int,
    n_epochs_burnin: int,
    n_initializations: int,
    n_epochs_initialization: int,
    learning_rate: float,
    clip_normalization: float,
    multi_gpu: bool = False,
    strategy: str = None,
):
    annealing_factor = Variable(0.0) if do_annealing else Variable(1.0)

    # Stretegy for distributed learning
    if multi_gpu:
        strategy = MirroredStrategy()
    elif strategy is None:
        strategy = get_strategy()

    # Create the model
    with strategy.scope():
        model = _model_structure(
            n_states=n_states,
            n_channels=n_channels,
            sequence_length=sequence_length,
            n_layers_inference=n_layers_inference,
            n_layers_model=n_layers_model,
            n_units_inference=n_units_inference,
            n_units_model=n_units_model,
            dropout_rate_inference=dropout_rate_inference,
            dropout_rate_model=dropout_rate_model,
            learn_means=learn_means,
            learn_covariances=learn_covariances,
            initial_means=initial_means,
            initial_covariances=initial_covariances,
            alpha_xform=alpha_xform,
            learn_alpha_scaling=learn_alpha_scaling,
        )

    # Override the compile method
    model.original_compile = model.compile

    def compile(optimizer=None):
        if optimizer == None:
            # Setup optimizer
            optimizer = optimizers.Adam(
                learning_rate=learning_rate, clipnorm=clip_normalization,
            )

        # Loss functions
        loss = [_ll_loss_fn, _create_kl_loss_fn(annealing_factor=annealing_factor)]

        # Compile
        model.original_compile(optimizer=optimizer, loss=loss)

    model.compile = compile

    # Compile the model
    model.compile()

    # Override the load_weights model (to ensure the model weights are loaded
    # in the correct scope)
    model.original_load_weights = model.load_weights

    def load_weights(filepath):
        with strategy.scope():
            model.original_load_weights(filepath)

    model.load_weights = load_weights

    # Override original predict method
    model.original_predict = model.predict

    def named_predict(*args, **kwargs):
        prediction = model.original_predict(*args, **kwargs)
        return_names = [
            "ll_loss",
            "kl_loss",
            "theta_t",
            "m_theta_t",
            "log_s2_theta_t",
            "mu_theta_jt",
            "log_sigma2_theta_j",
        ]
        prediction_dict = dict(zip(return_names, prediction))
        return prediction_dict

    model.predict = named_predict

    # Method to predict the inferred state time course
    def predict_states(*args, **kwargs):
        m_theta_t = model.predict(*args, **kwargs)["m_theta_t"]
        return np.concatenate(m_theta_t)

    model.predict_states = predict_states

    # Method to calculate the free energy (log likelihood + KL divergence)
    def free_energy(dataset, return_all=False):
        predictions = model.predict(dataset)
        ll_loss = np.mean(predictions["ll_loss"])
        kl_loss = np.mean(predictions["kl_loss"])
        if return_all:
            return ll_loss + kl_loss, ll_loss, kl_loss
        else:
            return ll_loss + kl_loss

    model.free_energy = free_energy

    # Callbacks
    burnin_callback = BurninCallback(epochs=n_epochs_burnin)

    annealing_callback = AnnealingCallback(
        annealing_factor=annealing_factor,
        annealing_sharpness=annealing_sharpness,
        n_epochs_annealing=n_epochs_annealing,
    )

    # Keep original fit method to train the model
    model.original_fit = model.fit

    # A fit method which includes annealing and burn-in in training
    def anneal_burnin_fit(*args, use_tqdm=False, tqdm_class=None, **kwargs):
        args = list(args)

        # Add annealing and burn-in callbacks
        additional_callbacks = [annealing_callback, burnin_callback]
        if use_tqdm:
            if tqdm_class is not None:
                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm_class)
            else:
                tqdm_callback = TqdmCallback(verbose=0, tqdm_class=tqdm)
            additional_callbacks.append(tqdm_callback)
        if len(args) > 5:
            args[5] = listify(args[5]) + additional_callbacks
        if "callbacks" in kwargs:
            kwargs["callbacks"] = listify(kwargs["callbacks"]) + additional_callbacks
        else:
            kwargs["callbacks"] = additional_callbacks

        # Train the model
        return model.original_fit(*args, **kwargs)

    # Initialize the means and covariances by training the model for
    # a few epochs and picking the model with the best free energy.
    def initialize_means_covariances(*args, **kwargs):

        # Set the epochs argument to n_epochs_initialization
        args = list(args)
        if len(args) > 3:
            args[3] = n_epochs_initialization
        if "epochs" in kwargs:
            kwargs["epochs"] = n_epochs_initialization

        # Pick the initialization with the lowest free energy
        best_free_energy = np.Inf
        for n in range(n_initializations):
            print(f"Initialization {n}")

            # Reset optimizer, model weights and annealing factor
            model.compile()
            reinitialize_model_weights(model)
            if do_annealing:
                annealing_factor.assign(0.0)

            # Train the model
            anneal_burnin_fit(*args, **kwargs)

            # Get the model weights if it's the best model
            free_energy = model.evaluate(args[0], verbose=0)[0]
            if free_energy < best_free_energy:
                best_initialization = n
                best_free_energy = free_energy
                best_weights = model.get_weights()
                best_optimizer = model.optimizer

        # Use the best model for the rest of training and reset the annealing factor
        print(f"Using initialization {best_initialization}")
        model.compile(optimizer=best_optimizer)
        model.set_weights(best_weights)
        if do_annealing:
            annealing_factor.assign(0.0)

    # Override the fit method to include initialization and annealing/burn-in
    def fit(*args, **kwargs):
        if (
            not hasattr(model, "initialized")
            and n_initializations is not None
            and n_initializations > 0
        ):
            initialize_means_covariances(*args, **kwargs)
            model.initialized = True

        # Train the model
        return anneal_burnin_fit(*args, **kwargs)

    model.fit = fit

    # Method to get the learned means and covariances for each state
    def get_means_covariances():
        mvn_layer = model.get_layer("mvn")
        means = mvn_layer.means.numpy()
        cholesky_covariances = mvn_layer.cholesky_covariances.numpy()
        covariances = cholesky_factor_to_full_matrix(cholesky_covariances)
        return means, covariances

    model.get_means_covariances = get_means_covariances

    # Method to set means and covariances for each state
    def set_means_covariances(means=None, covariances=None):
        mvn_layer = model.get_layer("mvn")
        layer_weights = mvn_layer.get_weights()

        # Replace means in the layer weights
        if means is not None:
            for i in range(len(layer_weights)):
                if layer_weights[i].shape == means.shape:
                    layer_weights[i] = means

        # Replace covariances in the layer weights
        if covariances is not None:
            for i in range(len(layer_weights)):
                if layer_weights[i].shape == covariances.shape:
                    layer_weights[i] = cholesky_factor(covariances)

        # Set the weights of the layer
        mvn_layer.set_weights(layer_weights)

    model.set_means_covariances = set_means_covariances

    # Method to get the alpha scaling of each state
    def get_alpha_scaling():
        mix_means_covs_layer = model.get_layer("mix_means_covs")
        alpha_scaling = mix_means_covs_layer.alpha_scaling.numpy()
        return alpha_scaling

    model.get_alpha_scaling = get_alpha_scaling

    return model


def _ll_loss_fn(y_true, ll_loss):
    """The first output of the model is the negative log likelihood
       so we just need to return it."""
    return ll_loss


def _create_kl_loss_fn(annealing_factor):
    def _kl_loss_fn(y_true, kl_loss):
        """Second output of the model is the KL divergence loss.
        We multiply with an annealing factor."""
        return annealing_factor * kl_loss

    return _kl_loss_fn


def _model_structure(
    n_states: int,
    n_channels: int,
    sequence_length: int,
    n_layers_inference: int,
    n_layers_model: int,
    n_units_inference: int,
    n_units_model: int,
    dropout_rate_inference: float,
    dropout_rate_model: float,
    learn_means: bool,
    learn_covariances: bool,
    initial_means: np.ndarray,
    initial_covariances: np.ndarray,
    alpha_xform: str,
    learn_alpha_scaling: bool,
):
    # Layer for input
    inputs = layers.Input(shape=(sequence_length, n_channels), name="data")

    # Inference RNN
    # - q(theta_t)     ~ N(m_theta_t, s2_theta_t)
    # - m_theta_t      ~ affine(RNN(Y_<=t))
    # - log_s2_theta_t ~ affine(RNN(Y_<=t))

    # Definition of layers
    input_normalisation_layer = layers.LayerNormalization()
    inference_input_dropout_layer = layers.Dropout(dropout_rate_inference)
    inference_output_layers = InferenceRNNLayers(
        n_layers_inference, n_units_inference, dropout_rate_inference
    )
    m_theta_t_layer = layers.Dense(n_states, activation="linear", name="m_theta_t")
    log_s2_theta_t_layer = layers.Dense(
        n_states, activation="linear", name="log_s2_theta_t"
    )

    # Layer to generate a sample from q(theta_t) ~ N(m_theta_t, log_s2_theta_t) via the
    # reparameterisation trick
    theta_t_layer = ReparameterizationLayer(name="theta_t")

    # Inference RNN data flow
    inputs_norm = input_normalisation_layer(inputs)
    inputs_norm_dropout = inference_input_dropout_layer(inputs_norm)
    inference_output = inference_output_layers(inputs_norm_dropout)
    m_theta_t = m_theta_t_layer(inference_output)
    log_s2_theta_t = log_s2_theta_t_layer(inference_output)
    theta_t = theta_t_layer([m_theta_t, log_s2_theta_t])

    # Model RNN
    # - p(theta_t|theta_<t) ~ N(mu_theta_jt, sigma2_theta_j)
    # - mu_theta_jt         ~ affine(RNN(theta_<t))
    # - log_sigma2_theta_j  = trainable constant

    # Definition of layers
    model_input_dropout_layer = layers.Dropout(dropout_rate_model)
    model_output_layers = ModelRNNLayers(
        n_layers_model, n_units_model, dropout_rate_model
    )
    mu_theta_jt_layer = layers.Dense(n_states, activation="linear", name="mu_theta_jt")
    log_sigma2_theta_j_layer = TrainableVariablesLayer(
        [n_states],
        initial_values=zeros(n_states),
        trainable=True,
        name="log_sigma2_theta_j",
    )

    # Layers for the means and covariances for observation model of each state
    observation_means_covs_layer = MultivariateNormalLayer(
        n_states,
        n_channels,
        learn_means=learn_means,
        learn_covariances=learn_covariances,
        initial_means=initial_means,
        initial_covariances=initial_covariances,
        name="mvn",
    )
    mix_means_covs_layer = MixMeansCovsLayer(
        n_states, n_channels, alpha_xform, learn_alpha_scaling, name="mix_means_covs"
    )

    # Layers to calculate the negative of the log likelihood and KL divergence
    log_likelihood_layer = LogLikelihoodLayer(name="ll")
    kl_loss_layer = KLDivergenceLayer(name="kl")

    # Model RNN data flow
    model_input_dropout = model_input_dropout_layer(theta_t)
    model_output = model_output_layers(model_input_dropout)
    mu_theta_jt = mu_theta_jt_layer(model_output)
    log_sigma2_theta_j = log_sigma2_theta_j_layer(inputs)  # inputs not used
    mu_j, D_j = observation_means_covs_layer(inputs)  # inputs not used
    m_t, C_t = mix_means_covs_layer([theta_t, mu_j, D_j])
    ll_loss = log_likelihood_layer([inputs, m_t, C_t])
    kl_loss = kl_loss_layer(
        [m_theta_t, log_s2_theta_t, mu_theta_jt, log_sigma2_theta_j]
    )

    outputs = [
        ll_loss,
        kl_loss,
        theta_t,
        m_theta_t,
        log_s2_theta_t,
        mu_theta_jt,
        log_sigma2_theta_j,
    ]

    return Model(inputs=inputs, outputs=outputs)

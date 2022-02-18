"""Demonstrates a failure of the LSTM-based DyNeMo model in learning long-range
temporal dependencies.

- Simulates hierarchical HSMM+HMM data.
- The lifetime of the top-level HSMM is longer than the sequence length, which
  means the LSTM cannot learn the top-level transitions.
"""

print("Setting up")
import os
import numpy as np
from sklearn.cluster import KMeans
from ohba_models import array_ops
from ohba_models.data import Data
from ohba_models.inference import metrics, modes, tf_ops
from ohba_models.models.dynemo import Config, Model
from ohba_models.simulation import HierarchicalHMM_MVN
from ohba_models.utils import plotting

# Make directory to hold plots
os.makedirs("figures", exist_ok=True)

# GPU settings
tf_ops.gpu_growth()

# Settings
config = Config(
    n_modes=4,
    n_channels=11,
    sequence_length=512,
    inference_n_units=64,
    inference_normalization="layer",
    model_n_units=64,
    model_normalization="layer",
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=200,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=400,
)

# Simulate data
bottom_level_trans_probs = [
    np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    ),
    np.array(
        [
            [0.9, 0, 0.1, 0],
            [0.15, 0.5, 0.2, 0.15],
            [0.1, 0, 0.8, 0.1],
            [0.1, 0, 0.1, 0.8],
        ]
    ),
    np.array(
        [
            [0.7, 0.3, 0, 0],
            [0.1, 0.8, 0.1, 0],
            [0.05, 0.05, 0.9, 0],
            [0.05, 0.05, 0.1, 0.8],
        ]
    ),
]

sim = HierarchicalHMM_MVN(
    n_samples=25600,
    n_modes=config.n_modes,
    n_channels=config.n_channels,
    top_level_hmm_type="hsmm",
    top_level_gamma_shape=1024,
    top_level_gamma_scale=1,
    top_level_trans_prob=None,
    bottom_level_trans_probs=bottom_level_trans_probs,
    means="zero",
    covariances="random",
    top_level_random_seed=123,
    bottom_level_random_seeds=[124, 126, 127],
    data_random_seed=555,
)
sim.standardize()
training_data = Data(sim.time_series)

plotting.plot_mode_lifetimes(sim.top_level_stc, filename="figures/sim_top_lt.png")

# Mean lifetime of top level HMM states
mlt, _ = modes.lifetime_statistics(sim.top_level_stc)
print("Top-level HMM mean lifetimes:", mlt)

# Prepare dataset
training_dataset = training_data.dataset(
    config.sequence_length, config.batch_size, shuffle=True
)
prediction_dataset = training_data.dataset(
    config.sequence_length,
    config.batch_size,
    shuffle=False,
)

# Build model
model = Model(config)
model.summary()

print("Training model")
history = model.fit(
    training_dataset,
    epochs=config.n_epochs,
    save_best_after=config.n_kl_annealing_epochs,
    save_filepath="model/weights",
)

# Free energy = Log Likelihood - KL Divergence
free_energy = model.free_energy(prediction_dataset)
print(f"Free energy: {free_energy}")

# Inferred mode mixing factors and mode time course
inf_alp = model.get_alpha(prediction_dataset)
inf_stc = modes.time_courses(inf_alp)
sim_bot_stc = sim.mode_time_course
sim_top_stc = sim.top_level_stc

orders = modes.match_modes(sim_bot_stc, inf_stc, return_order=True)
inf_stc = inf_stc[:, orders[1]]

dice = metrics.dice_coefficient(sim_bot_stc, inf_stc)
print("Dice coefficient:", dice)

plotting.plot_alpha(sim_bot_stc, n_samples=4000, filename="figures/sim_bot_stc.png")
plotting.plot_alpha(sim_top_stc, n_samples=4000, filename="figures/sim_top_stc.png")
plotting.plot_alpha(inf_stc, n_samples=4000, filename="figures/inf_stc.png")

# Sample from the trained model
sam_alp = model.sample_alpha(10000)
sam_stc = modes.time_courses(sam_alp)
sam_stc = sam_stc[:, orders[1]]

plotting.plot_alpha(sam_stc, n_samples=4000, filename="figures/sam_stc.png")

# Fractional occupancies
sim_fo = modes.fractional_occupancies(sim_bot_stc)
inf_fo = modes.fractional_occupancies(inf_stc)
sam_fo = modes.fractional_occupancies(sam_stc)

print("Fractional occupancies (Simulation):", sim_fo)
print("Fractional occupancies (DyNeMo):", inf_fo)
print("Fractional occupancies (Sample):", sam_fo)

# Sliding window fractional occupancies
#
# Sections of the inferred state time course should have different fractional
# occupancies because they were generated by different HMMs, this will be
# revealed by calculating the fractional occupancy of a short window.
n_window = 500
inf_swfo = np.empty([inf_stc.shape[0] - n_window, inf_stc.shape[1]])
for i in range(inf_stc.shape[0] - n_window):
    inf_swfo[i] = modes.fractional_occupancies(inf_stc[i : i + n_window])
sam_swfo = np.empty([sam_stc.shape[0] - n_window, sam_stc.shape[1]])
for i in range(sam_stc.shape[0] - n_window):
    sam_swfo[i] = modes.fractional_occupancies(sam_stc[i : i + n_window])

plotting.plot_alpha(inf_swfo, n_samples=4000, filename="figures/inf_swfo.png")
plotting.plot_alpha(sam_swfo, n_samples=4000, filename="figures/sam_swfo.png")

# Identify the top-level HSMM by k-means clustering the sliding window
# fractional occupancy
#
# The top-level HSMM lifetime are correct for the inferred state time course,
# but not for the sampled state time course
kmeans = KMeans(n_clusters=3).fit(inf_swfo)
inf_top_stc = array_ops.get_one_hot(kmeans.labels_)
sim_top_stc, inf_top_stc = modes.match_modes(sim_top_stc, inf_top_stc)

kmeans = KMeans(n_clusters=3).fit(sam_swfo)
sam_top_stc = array_ops.get_one_hot(kmeans.labels_)

plotting.plot_alpha(
    inf_top_stc,
    n_samples=4000,
    filename="figures/inf_swfo_kmeans.png",
)

plotting.plot_alpha(
    sam_top_stc,
    n_samples=4000,
    filename="figures/sam_swfo_kmeans.png",
)

# FANO factor
window_lengths = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
inf_ff = modes.fano_factor(inf_stc, window_lengths)
sam_ff = modes.fano_factor(sam_stc, window_lengths)

plotting.plot_scatter(
    [window_lengths] * inf_ff.shape[1],
    inf_ff.T,
    filename="figures/inf_ff.png",
)
plotting.plot_scatter(
    [window_lengths] * sam_ff.shape[1],
    sam_ff.T,
    filename="figures/sam_ff.png",
)

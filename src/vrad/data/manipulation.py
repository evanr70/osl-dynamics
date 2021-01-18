import numpy as np


def standardize(time_series: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardizes a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    axis : int
        Axis on which to perform the transformation.
    """
    time_series -= time_series.mean(axis=axis)
    time_series /= time_series.std(axis=axis)
    return time_series


def time_embed(
    time_series: np.ndarray, n_embeddings: int, output_file: str = None,
) -> np.ndarray:
    """Performs time embedding.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data.
    n_embeddings : int
        Number of samples in which to shift the data.
    output_file : str
    """

    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    n_samples, n_channels = time_series.shape

    # If an output file hasn't been passed we create a numpy array for the
    # time embedded data
    if output_file is None:
        time_embedded_series = np.empty(
            [n_samples - (n_embeddings + 1), n_channels * (n_embeddings + 2)],
            dtype=np.float32,
        )
    else:
        time_embedded_series = output_file

    # Generate time embedded series
    for i in range(n_channels):
        for j in range(n_embeddings + 2):
            time_embedded_series[:, i * (n_embeddings + 2) + j] = time_series[
                n_embeddings + 1 - j : n_samples - j, i
            ]

    return time_embedded_series


def n_batches(arr: np.ndarray, sequence_length: int, step_size: int = None) -> int:
    """Calculate the number of batches an array will be split into.

    Parameters
    ----------
    arr : numpy.ndarray
        Time series data.
    sequence_length : int
        Length of sequences which the data will be segmented in to.
    step_size : int
        The number of samples by which to move the sliding window between sequences.

    Returns
    -------
    int
        Number of batches.
    """
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)


def trim_time_series(
    time_series: np.ndarray, sequence_length: int, discontinuities: list = None,
) -> np.ndarray:
    """Trims a time seris.

    Removes data points lost to separating a time series into sequences.

    Parameters
    ----------
    time_series : numpy.ndarray
        Time series data for all subjects concatenated.
    sequence_length : int
        Length of sequence passed to the inference network and generative model.
    discontinuities : list of int
        Length of each subject's data. Optional. If nothing is passed we
        assume the entire time series is continuous.

    Returns
    -------
    list of np.ndarray
        Trimmed time series.
    """
    if discontinuities is None:
        # Assume entire time series corresponds to a single subject
        ts = [time_series]
    else:
        # Separate the time series for each subject
        ts = []
        for i in range(len(discontinuities)):
            start = sum(discontinuities[:i])
            end = sum(discontinuities[: i + 1])
            ts.append(time_series[start:end])

    # Remove data points lost to separating into sequences
    for i in range(len(ts)):
        n_sequences = ts[i].shape[0] // sequence_length
        ts[i] = ts[i][: n_sequences * sequence_length]

    return ts if len(ts) > 1 else ts[0]

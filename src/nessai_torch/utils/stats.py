import numpy as np


def rolling_mean_numpy(x: np.ndarray, n: int = 10) -> np.ndarray:
    """Compute the rolling mean with a given window size.

    Based on the version in nessai.

    Parameters
    ----------
    x :
        Array of samples
    n :
        Size of the window over which the rolling mean is computed.

    Returns
    -------
    Array containing the rolling mean.
    """
    # np.cumsum is faster but doesn't work with infs in the data.
    return np.convolve(
        np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"),
        np.ones(n) / n,
        mode="valid",
    )

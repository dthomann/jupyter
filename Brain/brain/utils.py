import numpy as np


def softmax(x):
    """Numerically stable softmax for 1D arrays."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)



"""Shared mathematical helpers used across baseline and exploratory modules."""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid that accepts scalars or arrays."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def W_R(R, R_low, R_high, w1=0.05, w2=0.05):
    """Windowed-RhoA function with soft lower and upper bounds."""
    R = np.asarray(R, dtype=float)
    return sigmoid((R - R_low) / w1) * sigmoid((R_high - R) / w2)


def smooth_step(x, width=0.02):
    """Smooth approximation to a Heaviside step."""
    return sigmoid(np.asarray(x, dtype=float) / width)

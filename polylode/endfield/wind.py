"""Explicit scene acceleration and project-owned deterministic cloth response.

Public semantics: https://magicasoft.jp/en/mc2_magicacloth_wind/ and
https://magicasoft.jp/en/mc2_wind_setup/. Neither specifies a force formula.
Our strength is acceleration (m/s²), not MagicaWindZone's wind speed (m/s).

Response is field * influence * (1-depthWeight*(1-depth)) * max(0,1+turbulence*wave).
Wave blends a sine with cubic-interpolated deterministic scalar noise. Frequency
is cycles per second; synchronization reduces phase differences between chain
baselines. These choices are our arithmetic, not recovered game behavior.
"""
from __future__ import annotations

import numpy as np


def normalize_wind(direction, strength: float) -> np.ndarray:
    """Finite normalized world acceleration; a zero direction gives no wind."""
    direction = np.asarray(direction, dtype=np.float64)
    strength = float(strength)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('wind direction must be a finite three-component vector')
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('wind strength must be finite and nonnegative')
    scale = float(np.max(np.abs(direction)))
    if scale == 0:
        return np.zeros(3)
    unit = direction / scale
    return unit / np.linalg.norm(unit) * strength


def _noise(sample: np.ndarray) -> np.ndarray:
    """SplitMix64 at integer samples, mapped to [-1,1] without global RNG state."""
    value = sample.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> 27)) * np.uint64(0x94D049BB133111EB)
    value ^= value >> 31
    return (value >> 11).astype(np.float64) * (2.0 / 2**53) - 1.0


def response_acceleration(field, settings: dict, time: float, depth, baselines) -> np.ndarray:
    """One acceleration per particle; a baseline denotes its fixed chain root.

    Moving wind is a separate unsupported effect; the solver reports nonzero
    declarations. This function responds only to the explicit scene field.
    """
    depth = np.asarray(depth, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if not np.any(field):
        return np.zeros((len(depth), 3))
    influence = float(settings.get('influence', 0.0))
    if influence == 0:
        return np.zeros((len(depth), 3))
    synchronization = float(settings.get('synchronization', 0.7))
    phase = (np.asarray(baselines) * 0.6180339887498949 % 1.) * (1.-synchronization)
    clock = time * float(settings.get('frequency', 1.0)) + phase
    integral = np.floor(clock)
    fraction = clock - integral
    smooth = fraction * fraction * (3. - 2. * fraction)
    noise = _noise(integral) * (1.-smooth) + _noise(integral+1.) * smooth
    blend = float(settings.get('blend', 0.7))
    wave = (1.-blend) * np.sin(2.*np.pi*clock) + blend * noise
    gain = np.maximum(0., 1.+float(settings.get('turbulence', 1.0))*wave)
    gain *= influence * (1.-float(settings.get('depthWeight', 0.0))*(1.-depth))
    return gain[:, None] * field

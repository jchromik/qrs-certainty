"""Tools for adding noise to signals."""

from copy import copy
from math import sqrt, log10

from wfdb import Record
import numpy as np


def power(signal):
    """Mean power of a signal."""
    return np.mean(np.array(signal)**2)


def snr(signal, noise):
    """Signal-to-noise ratio."""
    return power(signal) / power(noise)


def snrdb(signal, noise):
    """Signal-to-noise ratio in decibel (dB)."""
    return snr_to_snrdb(snr(signal, noise))


def snr_to_snrdb(snr):
    """Transform SNR to SNR in decibel (dB)."""
    return 10*log10(snr)


def snrdb_to_snr(snrdb):
    """Transform SNR in decibel (dB) to SNR."""
    return 10**(snrdb/10)


def scale(signal, ratio):
    """Multiplies the amplitude of all sampling points by ratio."""
    return np.array(signal)*ratio


def repeat_until(signal, length):
    """Increases the length of a signal by repeating it until the desired
    length is reached.
    """
    tiles = length // len(signal) + 1
    return np.tile(signal, tiles)[:length]


def add_repeating(*signals):
    """Brings all signals to equal length by repeating shorter signals.
    Then adds signals samplewise.
    """
    length = max([len(signal) for signal in signals])
    return np.array(
        [repeat_until(signal, length) for signal in signals]
    ).sum(axis=0)


def _apply_noise(signal, noise, target_snr):
    if len(noise) > len(signal):
        noise = noise[:len(signal)]
    current_snr = snr(signal, noise)
    ratio = sqrt(current_snr / target_snr)
    scaled_noise = scale(noise, ratio)
    return add_repeating(signal, scaled_noise)


def _apply_noises(signals, noises, target_snr):
    return np.array([
        _apply_noise(signal, noise, target_snr)
        for signal, noise in zip(signals, noises)])


def _apply_noise_record(signal, noise, target_snr):
    """Apply noise to signal, when signal is a WFDB Record."""
    signals = signal.p_signal.T
    noises = (
        noise.p_signal.T if isinstance(noise, Record)
        else [noise]*len(signals))

    contaminated_signals = _apply_noises(signals, noises, target_snr)

    noisy_record = copy(signal)
    noisy_record.p_signal = contaminated_signals.T

    return noisy_record


def _apply_noise_iterable(signal, noise, target_snr):
    """Apply noise to signal, when signal is an Iterable."""
    noise_iterable = noise.p_signal.T[0] if isinstance(
        noise, Record) else noise
    return _apply_noise(signal, noise_iterable, target_snr)


def apply_noise(signal, noise, target_snr):
    """Applies a noise template to a signal producing a given SNR.

    Args:
        signal (Iterable or WFDB Record): The clean signal.
        noise (Iterable or WFDB Record): The noise template added to signal.
        target_snr (int or float): The signal-to-noise ratio to achieve.
    """
    if isinstance(signal, Record):
        return _apply_noise_record(signal, noise, target_snr)
    return _apply_noise_iterable(signal, noise, target_snr)


def apply_noise_db(signal, noise, target_snrdb):
    """Like apply_noise, but with SNR given in dB."""
    target_snr = snrdb_to_snr(target_snrdb)
    return apply_noise(signal, noise, target_snr)

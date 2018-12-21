from collections import Iterable
from copy import copy
from math import sqrt

from wfdb import Record
import numpy as np


def power(signal):
    return np.mean(np.array(signal)**2)


def snr(signal, noise):
    # TODO: Better return this in dB?
    return power(signal) / power(noise)


def scale(signal, ratio):
    return np.array(signal)*ratio


def repeat_until(signal, length):
    tiles = length // len(signal) + 1
    return np.tile(signal, tiles)[:length]


def add_repeating(*signals):
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
    signals = signal.p_signal.T
    noises = (
        noise.p_signal.T if isinstance(noise, Record)
        else [noise]*len(signals))

    contaminated_signals = _apply_noises(signals, noises, target_snr)

    noisy_record = copy(signal)
    noisy_record.p_signal = contaminated_signals.T

    return noisy_record


def _apply_noise_iterable(signal, noise, target_snr):
    noise_iterable = noise.p_signal.T[0] if isinstance(
        noise, Record) else noise
    return _apply_noise(signal, noise_iterable, target_snr)


def apply_noise(signal, noise, target_snr):
    if isinstance(signal, Record):
        return _apply_noise_record(signal, noise, target_snr)
    return _apply_noise_iterable(signal, noise, target_snr)

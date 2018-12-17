from itertools import cycle, islice
from math import sqrt

def power(signal):
    return sum([sample**2 for sample in signal]) / len(signal)

def snr(signal, noise):
    return power(signal) / power(noise)

def scale(signal, ratio):
    return [sample*ratio for sample in signal]

def repeat_until(signal, length):
    return list(islice(cycle(signal), length))

def add_shortest(*signals):
    return [sum(samples) for samples in zip(*signals)]

def add_repeating(*signals):
    length = max([len(signal) for signal in signals])
    signals = [repeat_until(signal, length) for signal in signals]
    return add_shortest(*signals)

def apply_noise(signal, noise, target_snr):
    current_snr = snr(signal, noise)
    ratio = sqrt(current_snr / target_snr)
    scaled_noise = scale(noise, ratio)
    return (
        add_repeating(signal, scaled_noise) if len(signal) > len(noise)
        else add_shortest(signal, scaled_noise))
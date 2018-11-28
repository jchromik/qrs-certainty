import itertools
import numpy as np

# Utility

def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def discretize(signal, threshold=.5):
    return list(map(lambda x: 1. if x > threshold else 0., signal))

def differentiate(signal):
    s1, s2 = itertools.tee(signal)
    next(s2)
    return (y-x for x, y in zip(s1, s2))

# Converting trigger signal to list of trigger points

def remove_ripple(signal, tolerance):
    """Removes ripple from a discretized signal.

    Args:
        signal: Discretized signal. May only contain 0 and 1 elements.
        tolerance: Maximum length of a 0-spike to count as ripple.
    Returns:
        Signal without ripple.
    """
    for idx in range(len(signal)):
        yield signal[idx]
        if idx+tolerance >= len(signal): continue
        if signal[idx] and signal[idx+tolerance]:
            signal[idx:idx+tolerance] = itertools.repeat(1.0, tolerance)

def signal_to_spikes(signal):
    """Finds and returns 1-spikes in a discretized signal.

    Args:
        signal: Discretized signal. May only contain 0 and 1 elements.
    Returns:
        List of (begin, end) tuples with begin being the begin index of a
        1-spike and end being the end index of a 1-spike.
    """
    ds = differentiate(itertools.chain(iter([0.0]), signal, iter([0.0])))
    ds1, ds2 = itertools.tee(ds)
    begins = (idx for idx, val in enumerate(ds1) if val == 1)
    ends = (idx for idx, val in enumerate(ds2) if val == -1)
    return zip(begins, ends)

def spikes_to_points(spikes):
    """Compute trigger points as center of spikes.

    Args:
        spikes: List of 1-spikes in a trigger signal denoted as (begin, end).
    Return:
        List of trigger points.
    """
    points = []
    for begin, end in spikes:
        points.append(int((begin + end) / 2))
    return points

def signal_to_points(signal, threshold=.5, tolerance=3, with_certainty=False):
    """Generate trigger points from a trigger signal.

    Args:
        signal: Trigger signal the trigger points are generated from.
        threshold: Defines which trigger signal intensity is interpreted as
            'QRS complex' detected.
        tolerance: Ripple tolerance. Defines how large areas below threshold
            might be to account for the same trigger point.
        with_certainty: Whether to return how certain a QRS complex is at the
            corresponding trigger point.
    Returns:
        List of trigger points. Also, list of certainties if with_certainty
        is True.
    """
    normalized_signal = normalize(signal)
    discretized_signal = discretize(normalized_signal, threshold)
    sanitized_signal = remove_ripple(discretized_signal, tolerance)

    spikes = list(signal_to_spikes(sanitized_signal))
    
    points = spikes_to_points(spikes)
    if not with_certainty: return points
    
    certainties = spikes_to_certainties(spikes, normalized_signal)
    return points, certainties

# Synthesizing a trigger signal from trigger points

def points_to_signal(points, signal_length, window_size):
    """Synthesize a trigger signal from trigger points.
    
    Args:
        points: List of trigger points.
        signal_length: List of the returning trigger signal.
        window_size: Width of the 1-spikes around the trigger points.
    Returns:
        Sythesized trigger signal with 1-spikes of window_size around the given
        trigger points. All signal samples are either 0 or 1 and there is no
        ripple.
    """
    signal = np.zeros(signal_length)
    for point in points:
        start = point - window_size // 2
        end = point + window_size // 2
        signal[start:end] = 1.
    return signal

# Certainty assessment

def spikes_to_certainties(spikes, signal):
    """Assess certainty of 1-spikes in a normalized trigger signal.

    Args:
        spikes: List of 1-spikes in a trigger signal denoted as (begin, end).
        signal: Normalized trigger signal.
    Returns:
        List of certainties.
    """
    return [spike_certainty(spike, signal) for spike in spikes]

def spike_certainty(spike, signal):
    """Assess certainty of a single 1-spike in a normalizes trigger signal.
    This is done by computing the mean of all signal samples in the range
    defined by the spike.

    Args:
        spike: Single 1-spike denoted as (begin, end).
        signal: Normalized trigger signal.
    Returns:
        Certainty of the spike in the signal, i.e. mean of signal samples in
        spike range.
    """
    begin, end = spike
    return np.mean(signal[begin:end])
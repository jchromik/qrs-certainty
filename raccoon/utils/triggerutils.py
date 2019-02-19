import itertools

# Utility

def discretize(signal, threshold=.5):
    return list(map(lambda x: 1. if x >= threshold else 0., signal))

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
    tolerance += 1
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
        Iterable of (begin, end) tuples with begin being the begin index of a
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
        points.append((begin + end) // 2)
    return points

def signal_to_points(
        signal, threshold=.5, tolerance=3, with_certainty=False, min_width=0
):
    """Generate trigger points from a trigger signal.

    Args:
        signal: Trigger signal the trigger points are generated from.
        threshold: Defines which trigger signal intensity is interpreted as
            'QRS complex' detected.
        tolerance: Ripple tolerance. Defines how large areas below threshold
            might be to account for the same trigger point.
        with_certainty: Whether to return how certain a QRS complex is at the
            corresponding trigger point.
        min_width: Minimum width of a spike. Smaller spikes are ignored and not
            converted to trigger points.
    Returns:
        List of trigger points. Also, list of certainties if with_certainty
        is True.
    """
    discretized_signal = discretize(signal, threshold)
    sanitized_signal = remove_ripple(discretized_signal, tolerance)

    spikes = [
        (begin, end)
        for begin, end
        in signal_to_spikes(sanitized_signal)
        if end-begin >= min_width]

    points = spikes_to_points(spikes)
    if not with_certainty: return points
    
    certainties = spikes_to_certainties(spikes, signal)
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
    signal = [0.]*signal_length
    for point in points:
        start = max(point - window_size // 2, 0)
        end = min(start + window_size, signal_length)
        signal[start:end] = [1.]*(end-start)
    return signal

# Certainty assessment

def spikes_to_certainties(spikes, signal):
    """Assess certainty of 1-spikes in a trigger signal.

    Args:
        spikes: List of 1-spikes in a trigger signal denoted as (begin, end).
        signal: Trigger signal.
    Returns:
        List of certainties.
    """
    return [spike_certainty(spike, signal) for spike in spikes]

def spike_certainty(spike, signal):
    """Assess certainty of a single 1-spike in a trigger signal.
    This is done by computing the mean of all signal samples in the range
    defined by the spike.

    Args:
        spike: Single 1-spike denoted as (begin, end).
        signal: Trigger signal.
    Returns:
        Certainty of the spike in the signal, i.e. mean of signal samples in
        spike range.
    """
    begin, end = spike
    return sum(signal[begin:end]) / (end-begin)
BEAT_CLASSIFIERS = ["N","L","R","B","A","a","J","S","V","r","F","e","j","n","E","/","f","Q","?","|"]

def _extract(annotations):
    """
    Extracts positions and labels from wfdb annotation object.
    Removes first element which is begin marker.
    """
    positions = annotations.sample[1:]
    labels = annotations.symbol[1:]
    return positions, labels

def _filter(positions, labels, keep=BEAT_CLASSIFIERS):
    """
    Filter out all annotations not having labels as specified in keep (e.g. not denoting beats).
    """
    filtered_positions = [val for idx, val in enumerate(positions) if labels[idx] in keep]
    filtered_labels = [val for idx, val in enumerate(labels) if labels[idx] in keep]
    return filtered_positions, filtered_labels

def trigger_points(annotation):
    positions, labels = _extract(annotation)
    positions, _ = _filter(positions, labels)
    return positions
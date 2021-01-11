import numpy as np

def merge(true_trigger, detected_trigger, tolerance):
    for true_point in true_trigger:
        matches = [detected_point
            for detected_point in detected_trigger
            if abs(detected_point - true_point) <= tolerance]
        
        if len(matches) == 0:
            yield (true_point, None, 'FN')
            continue
            
        matches.sort(key=lambda m: abs(true_point - m))
        yield (true_point, matches[0], 'TP')
        
        for match in matches[1:]:
            yield (true_point, match, 'FP')
            
    for detected_point in detected_trigger:
        matches = [true_point
            for true_point in true_trigger
            if abs(true_point - detected_point) <= tolerance]
        
        if len(matches) == 0:
            yield (None, detected_point, 'FP')

def trigger_metrics(true_trigger, detected_trigger, tolerance):
    tp, tn, fp, fn = 0, 0, 0, 0

    for true_point in true_trigger:
        matches = len([detected_point
            for detected_point in detected_trigger
            if abs(detected_point - true_point) <= tolerance])
        if matches == 0: fn += 1
        if matches > 0: tp += 1
        if matches > 1: fp += matches-1

    for detected_point in detected_trigger:
        matches = len([true_point
            for true_point in true_trigger
            if abs(true_point - detected_point) <= tolerance])
        if matches == 0: fp += 1

    return (tp, tn, fp, fn)

def triggers_metrics(true_triggers, detected_triggers, tolerance):
    result = (0, 0, 0, 0)
    for true_trigger, detected_trigger in zip(true_triggers, detected_triggers):
        metrics = trigger_metrics(true_trigger, detected_trigger, tolerance)
        result = tuple(np.add(result, metrics))
    return result

def sensitivity(tp, fn):
    try:
        return tp / (tp+fn)
    except ZeroDivisionError:
        return float('nan')

def ppv(tp, fp):
    try:
        return tp / (tp+fp)
    except ZeroDivisionError:
        return float('nan')

def f1(tp, fp, fn):
    try:
        return (2*tp) / (2*tp + fp + fn)
    except ZeroDivisionError:
        return float('nan')
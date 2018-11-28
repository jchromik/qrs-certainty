import numpy as np

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
        metrics = trigger_metrics(true_trigger, detected_trigger, 5)
        result = tuple(np.add(result, metrics))
    return result

def sensitivity(tp, fn):
    try:
        return tp / (tp+fn)
    except ZeroDivisionError:
        return float('nan') if tp == 0 else float('inf')

def ppv(tp, fp):
    try:
        return tp / (tp+fp)
    except ZeroDivisionError:
        return float('nan') if tp == 0 else float('inf')

def f1(tp, fp, fn):
    try:
        return (2*tp) / (2*tp + fp + fn)
    except ZeroDivisionError:
        return float('nan') if tp == 0 else float('inf')
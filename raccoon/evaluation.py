from evaluationutils import trigger_metrics, sensitivity, ppv, f1

import matplotlib.pyplot as plt
import numpy as np
import wfdb


class Evaluation():

    def __init__(
        self, output_dir, evaluation_id, detector,
        train_names, train_signals, train_triggers,
        test_names, test_signals, test_triggers,
        actual_detected_distance
    ):
        self.output_dir = output_dir
        self.id = evaluation_id
        self.detector = detector

        if not len(train_names) == len(train_signals) == len(train_triggers):
            raise ValueError('Training data have different length.') 
        self.train_names = train_names
        self.train_signals = train_signals
        self.train_triggers = train_triggers

        if not len(test_names) == len(test_signals) == len(test_triggers):
            raise ValueError('Test data have different length.') 
        self.test_names = test_names
        self.test_signals = test_signals
        self.test_triggers = test_triggers

        self.actual_detected_distance = actual_detected_distance

    def run(self):
        self.detector.reset()
        self.detector.train(self.train_signals, self.train_triggers)
        self.trigger_signals, self.detected_triggers = (
            self.detector.triggers_and_signals(self.test_signals))
        self.metrics = [
            trigger_metrics(true, detected, self.actual_detected_distance)
            for true, detected
            in zip(self.test_triggers, self.detected_triggers)]

    def report(self):
        report = []
        for test, metric in zip(self.test_names, self.metrics):
            tp, tn, fp, fn = metric
            report.append({
                'ID': self.id,
                'Detector': self.detector_name(),
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                'Train Records': self.train_names,
                'Test Record': test,
                'Sensitivity': sensitivity(tp, fn),
                'PPV': ppv(tp, fp),
                'F1': f1(tp, fp, fn)})
        return report

    # SAVING DATA TO FILES

    def save_annotations(self):
        for name, trigger in zip(self.test_names, self.detected_triggers):
            if len(trigger) == 0: continue
            filename = "{}_{}_{}".format(self.id, self.detector_name(), name)
            wfdb.wrann(
                filename, "atr", np.array(trigger), ['N']*len(trigger),
                write_dir=self.output_dir)

    def save_model(self):
        self.detector.save_model("{}/{}_{}.h5".format(
            self.output_dir,
            self.id,
            self.detector_name()))

    # PLOTTING

    def plot_detections(self, xlim):
        for idx in range(len(self.test_names)):
            self._plot_detection(xlim, idx)

    def _plot_detection(self, xlim, idx):
        plt.figure(figsize=(xlim // 100, 5))
        plt.xlim((0, xlim))
        plt.plot(self.test_signals[idx])
        plt.plot(self.trigger_signals[idx])
        if len(self.test_triggers) > 0:
            plt.plot(self.test_triggers, [1]*len(self.test_triggers), 'go')
        if len(self.detected_triggers) > 0:
            plt.plot(self.detected_triggers, [1]*len(self.detected_triggers), 'ro')
        plt.savefig("{}/{}_{}_{}.{}".format(
            self.output_dir, self.id, self.detector_name(),
            self.test_names[idx], 'svg'))
        plt.close()

    # CONVENIENCE

    def detector_name(self):
        return self.detector.__class__.__name__
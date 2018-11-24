from evaluationutils import trigger_metrics, sensitivity, ppv, f1

import matplotlib.pyplot as plt
import numpy as np
import wfdb


class Evaluation():

    def __init__(
        self, output_dir, evaluation_id, detector,
        train_records, train_triggers,
        test_records, test_triggers,
        actual_detected_distance
    ):
        self.output_dir = output_dir
        self.id = evaluation_id
        self.detector = detector

        if not len(train_records) == len(train_triggers):
            raise ValueError('Training data have different length.') 
        self.train_records = train_records
        self.train_triggers = train_triggers

        if not len(test_records) == len(test_triggers):
            raise ValueError('Test data have different length.') 
        self.test_records = test_records
        self.test_triggers = test_triggers

        self.actual_detected_distance = actual_detected_distance

    def run(self):
        self.detector.reset()
        self.detector.train(self.train_records, self.train_triggers)
        self.trigger_signals, self.detected_triggers = (
            self.detector.triggers_and_signals(self.test_records))
        self.metrics = [
            trigger_metrics(true, detected, self.actual_detected_distance)
            for true, detected
            in zip(self.test_triggers, self.detected_triggers)]

    def report(self):
        report = []
        for test_record, metric in zip(self.test_records, self.metrics):
            tp, tn, fp, fn = metric
            report.append({
                'ID': self.id,
                'Detector': self.detector_name(),
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                'Train Records': [r.record_name for r in self.train_records],
                'Test Record': test_record.record_name,
                'Sensitivity': sensitivity(tp, fn),
                'PPV': ppv(tp, fp),
                'F1': f1(tp, fp, fn)})
        return report

    # SAVING DATA TO FILES

    def save_annotations(self):
        for test_record, trigger in zip(self.test_records, self.detected_triggers):
            if len(trigger) == 0: continue
            filename = "{}_{}_{}".format(
                self.id, self.detector_name(), test_record.record_name)
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
        for idx in range(len(self.test_records)):
            self._plot_detection(xlim, idx)

    def _plot_detection(self, xlim, idx):
        plt.figure(figsize=(xlim // 100, 5))
        plt.xlim((0, xlim))
        plt.plot(self.test_records[idx].p_signal.T[0])
        plt.plot(self.trigger_signals[idx])
        if len(self.test_triggers) > 0:
            plt.plot(self.test_triggers, [1]*len(self.test_triggers), 'go')
        if len(self.detected_triggers) > 0:
            plt.plot(self.detected_triggers, [1]*len(self.detected_triggers), 'ro')
        plt.savefig("{}/{}_{}_{}.{}".format(
            self.output_dir, self.id, self.detector_name(),
            self.test_records[idx].record_name, 'svg'))
        plt.close()

    # CONVENIENCE

    def detector_name(self):
        return self.detector.__class__.__name__
from collections import OrderedDict
from time import time
from utils.evaluationutils import trigger_metrics, sensitivity, ppv, f1

import matplotlib.pyplot as plt
import numpy as np
import wfdb


class Evaluation():

    def __init__(
        self, output_dir, evaluation_id, detector,
        train_records, train_triggers,
        test_records, test_triggers,
        trigger_distance
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

        self.trigger_distance = trigger_distance

    def run(self):
        self.detector.reset()
        self.detector.train(self.train_records, self.train_triggers)
        self._timed_detection()
        self.metrics = [
            trigger_metrics(true, detected, self.trigger_distance)
            for true, detected
            in zip(self.test_triggers, self.detected_triggers)]

    def _timed_detection(self):
        self.trigger_signals = []
        self.detected_triggers = []
        self.runtimes = []
        for record in self.test_records:
            self._timed_detection_for(record)

    def _timed_detection_for(self, record):
        start_time = time()
        signal, trigger = self.detector.trigger_and_signal(record)
        end_time = time()
        runtime = end_time - start_time
        self.trigger_signals.append(signal)
        self.detected_triggers.append(trigger)
        self.runtimes.append(runtime)

    def report(self):
        report = []
        report_data = zip(self.test_records, self.metrics, self.runtimes)
        for test_record, metric, runtime in report_data:
            tp, tn, fp, fn = metric
            report.append(OrderedDict([
                ('ID', self.id),
                ('Detector', repr(self.detector)),
                ('Train Records', [r.record_name for r in self.train_records]),
                ('Test Record', test_record.record_name),
                ('TP', tp), ('TN', tn), ('FP', fp), ('FN', fn),
                ('Sensitivity', sensitivity(tp, fn)),
                ('PPV', ppv(tp, fp)),
                ('F1', f1(tp, fp, fn)),
                ('Detection Runtime', runtime)]))
        return report

    # SAVING DATA TO FILES

    def save_annotations(self):
        for test_record, trigger in zip(self.test_records, self.detected_triggers):
            if len(trigger) == 0: continue
            wfdb.wrann(
                self._file_name_for(test_record), "atr", np.array(trigger),
                ['N']*len(trigger), write_dir=self.output_dir)

    def save_model(self):
        self.detector.save_model("{}/{}.h5".format(
            self.output_dir, self._file_name()))

    # PLOTTING

    def plot_detections(self, xlim):
        for idx in range(len(self.test_records)):
            self._plot_detection(xlim, idx)

    def _plot_detection(self, xlim, idx):
        record = self.test_records[idx]
        ecg_signal = self.test_records[idx].p_signal.T[0]
        trigger_signal = self.trigger_signals[idx]
        test_trigger = self.test_triggers[idx]
        detected_trigger = self.detected_triggers[idx]
        
        plt.figure(figsize=(xlim // 100, 5))
        plt.xlim((0, xlim))
        plt.plot(ecg_signal)
        plt.plot(trigger_signal)
        if len(test_trigger) > 0:
            plt.plot(test_trigger, [1]*len(test_trigger), 'go')
        if len(detected_trigger) > 0:
            plt.plot(detected_trigger, [1]*len(detected_trigger), 'ro')
        plt.savefig("{}/{}.svg".format(
            self.output_dir, self._file_name_for(record)))
        plt.close()

    # HELPERS

    def _file_name(self):
        return "_".join([
            str(self.id), self.detector.name, self.detector.__class__.__name__])

    def _file_name_for(self, record):
        return "_".join([self._file_name(), record.record_name])
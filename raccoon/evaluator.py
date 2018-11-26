from annotationutils import trigger_points
from evaluation import Evaluation

from itertools import product
from sklearn.model_selection import KFold, LeaveOneOut, PredefinedSplit
from tabulate import tabulate

import csv
import numpy as np
import wfdb


def _select(lst, idxs):
    return [lst[i] for i in idxs]


class Evaluator():

    def __init__(
        self, input_dir, output_dir, sampto=None,
        generate_plots=False,
        save_annotations=False,
        save_model=False,
        actual_detected_distance = 5
    ):
        # instance variable set with constructor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sampto = sampto
        self.save_annotations = save_annotations
        self.save_model = save_model
        self.generate_plots = generate_plots
        self.actual_detected_distance = actual_detected_distance
        # instance variables set later on
        self.detectors = []
        self.records = []
        self.triggers = []
        # instance variables having default values
        self.plot_xlim = 10000

    def __repr__(self):
        return "Evaluator"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tReading from: {}".format(self.input_dir),
            "\tWriting to: {}".format(self.output_dir),
            "\tReading {} samples per signal.".format(self.sampto),
            "\tMaximum allowed distance between actual and detected trigger " +
            "points: {} samples".format(self.actual_detected_distance),
            "\tScikit-learn Cross Validation Method: {}".format(self.cval)])

    # PRIVATE DATA ACCESSING HELPERS 

    def _read_record(self, record_name):
        record_path = '/'.join([self.input_dir, record_name])
        record = wfdb.rdrecord(record_path, sampto=self.sampto)
        annotation = wfdb.rdann(record_path, 'atr', sampto=self.sampto)
        trigger = trigger_points(annotation)
        return record, trigger

    def _records(self, idxs):
        return _select(self.records, idxs)
    
    def _triggers(self, idxs):
        return _select(self.triggers, idxs)

    # PUBLIC DATA ACCESSING

    def add_detectors(self, *detectors):
        self.detectors.extend(detectors)
    
    def add_records(self, *record_names):
        for record_name in record_names:
            record, trigger = self._read_record(record_name)
            self.records.append(record)
            self.triggers.append(trigger)

    # PUBLIC EVALUATION INTERFACE

    def kfold(self, k):
        self.cval = KFold(n_splits=k)
        return self._eval_cross_validator()

    def loocv(self):
        self.cval = LeaveOneOut()
        return self._eval_cross_validator()

    def defined(self, test_record_names):
        test_fold = [
            0 if record.record_name in test_record_names else -1
            for record in self.records]
        self.cval = PredefinedSplit(test_fold)
        return self._eval_cross_validator()

    # PRIVATE EVALUATION FUNCTIONS

    def _eval_cross_validator(self):
        reports = self._eval_detectors()
        self._save_header()
        self._save_reports(reports)
        self._print_reports(reports)
        return reports

    def _eval_detectors(self):
        reports = []
        splits = self.cval.split(self.records)
        combinations = product(self.detectors, splits)
        for eval_id, combination in enumerate(combinations):
            detector, split = combination
            train, test = split
            reports.extend(self._eval_detector(eval_id, detector, train, test))
        return reports

    def _eval_detector(self, eval_id, detector, train, test):
        evaluation = Evaluation(
            self.output_dir, eval_id, detector,
            self._records(train), self._triggers(train),
            self._records(test), self._triggers(test),
            self.actual_detected_distance)
        
        evaluation.run()

        if self.save_annotations: evaluation.save_annotations()
        if self.save_model: evaluation.save_model()
        if self.generate_plots: evaluation.plot_detections(self.plot_xlim)
        
        return evaluation.report()

    # SAVING AND PRINTING

    def _save_header(self):
        header = (
            str(self)
            + "\n\n\nDETECTORS:\n\n"
            + "\n".join(
                [str(detector) for detector in self.detectors])
            + "\n\n\nRECORDS:\n\n"
            + ", ".join([record.record_name for record in self.records])
            + "\n")
        with open("{}/header.txt".format(self.output_dir), 'w') as f:
            f.write(header)

    def _save_reports(self, reports):
        keys = reports[0].keys()
        with open("{}/report.csv".format(self.output_dir), 'w') as f:
            dw = csv.DictWriter(f, keys)
            dw.writeheader()
            dw.writerows(reports)

    def _print_reports(self, reports):
        keys = reports[0].keys()
        table = [[report[i] for i in keys] for report in reports]
        print(tabulate(table, headers=keys))

from utils.annotationutils import trigger_points
from evaluation import Evaluation

from itertools import product
from sklearn.model_selection import KFold, LeaveOneOut, PredefinedSplit
from tabulate import tabulate

import csv
import numpy as np
import os
import wfdb


def _select(lst, idxs):
    return [lst[i] for i in idxs]


class Evaluator():

    def __init__(
        self, input_dir, output_dir, sampto=None,
        plot_limit=0,
        save_annotations=False,
        save_model=False,
        trigger_distance = 5
    ):
        """The Evaluator compares different Detectors by first providing them
        with training records and subsequently testing their performace on
        other records.

        Args:
            input_dir (str): Directory path to read records from.
            output_dir (str): Directory path to write generated files to.
            sampto (int, optional): How many samples are read from records.
                All samples are read if unspecified.
            plot_limit (int, optional): How many samples are plotted. If 0
                (zero), no plots are generated. Otherwise plots are written
                to output_dir.
            save_annotations (bool, optional): If True, MIT Annotation files
                containing trigger points are written to output_dir.
            save_model (bool, optional): If True, HDF5 model files are written
                to output_dir. (Only for neural network based detectors.)
            trigger_distance (int, optional): How far (in samples) detected and
                actual trigger points can be apart from each other to be
                recognized as 'correctly detected' (true positive).
        """
        # instance variable set with constructor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sampto = sampto
        self.plot_limit = plot_limit
        self.save_annotations = save_annotations
        self.save_model = save_model
        self.trigger_distance = trigger_distance

        # instance variables set later on
        self.detectors = []
        self.records = []
        self.triggers = []

        # instance variables having default values
        self.cval = None

        # create output dir if not exists
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    def __repr__(self):
        return "Evaluator"

    def __str__(self):
        return "\n".join([
            repr(self),
            f"\tReading from: {self.input_dir}",
            f"\tWriting to: {self.output_dir}",
            f"\tReading {self.sampto} samples per signal.",
            f"\tTrigger Distance: {self.trigger_distance} samples",
            f"\tScikit-learn Cross Validation Method: {self.cval}"])

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
        """Perform k-fold cross-validation with previously specified detectors
        on previously specified records.
        
        Args:
            k (int): Number of iterations/splits.
        """
        self.cval = KFold(n_splits=k)
        return self._eval_cross_validator()

    def loocv(self):
        """Perform leave-one-out cross-validation with previously specified
        detectors on previously specified records. Number of iterations/splits
        equals number of records with this cross-validation method per
        definitionem.
        """
        self.cval = LeaveOneOut()
        return self._eval_cross_validator()

    def defined(self, test_record_names):
        """Run evaluation ith previously specified detectors on previously
        specified records. Do not use cross-validation but do a predefined
        split taking the given records as test records and all other records
        as training records.

        Args:
            test_record_names (list of str): List of record names to use for
                testing. All other records known to this evaluator are used for
                training the detectors.
        """
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
            self.trigger_distance)
        
        evaluation.run()

        if self.save_annotations: evaluation.save_annotations()
        if self.save_model: evaluation.save_model()
        if self.plot_limit > 0: evaluation.plot_detections(self.plot_limit)
        
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
        with open(f"{self.output_dir}/header.txt", 'w+') as f:
            f.write(header)

    def _save_reports(self, reports):
        keys = reports[0].keys()
        with open(f"{self.output_dir}/report.csv", 'w+') as f:
            dw = csv.DictWriter(f, keys)
            dw.writeheader()
            dw.writerows(reports)

    def _print_reports(self, reports):
        keys = reports[0].keys()
        table = [[report[i] for i in keys] for report in reports]
        print(tabulate(table, headers=keys))

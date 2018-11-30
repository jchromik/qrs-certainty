#!/usr/bin/env python3

# Avoid threading issues with Tk, needs to be done before first importing wfdb,
# since wfdb (somehow) uses matplotlib.
import matplotlib
matplotlib.use("agg")

from argparse import ArgumentParser
from os.path import dirname, isabs, join, realpath

import json
import os
import sys
import warnings

# Parse CLI command

parser = ArgumentParser(description=(
    "Evaluate different methods for finding QRS complexes "
    "in single channel ECG signals."))
parser.add_argument(
    "configuration_file", metavar="CONFIGURATION_FILE", type=str,
    help="Configuration file specifying the evaluation.")
args = parser.parse_args()

def absify(path, dir):
    return path if isabs(path) else realpath(join(dir, path))

def all_signals_in(path):
    return list(set([f.split('.')[0] for f in os.listdir(path)]))

# Read configuration file

with open(args.configuration_file, "r") as f:
    conf = json.load(f)
    conf["input_dir"] = absify(conf["input_dir"], dirname(f.name))
    conf["output_dir"] = absify(conf["output_dir"], dirname(f.name))
    if not "records" in conf:
        conf["records"] = all_signals_in(conf["input_dir"])

if not conf["verbose"]:
    # disable Tensorflow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # disable Python warnings
    if not sys.warnoptions: warnings.simplefilter("ignore")

# Build Evaluator

sys.path.append(join(dirname(__file__), "./raccoon"))
from utils.builders import evaluator_from_dict

evaluator = evaluator_from_dict(conf)
cv_method = conf["cv_method"]
test_records = conf["test_records"] if "test_records" in conf else None

if cv_method == "loocv": evaluator.loocv()
elif cv_method == "k2": evaluator.kfold(k=2)
elif cv_method == "k10": evaluator.kfold(k=10)
elif cv_method == 'defined': evaluator.defined(test_records)

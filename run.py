#!/usr/bin/env python3

import json
import os
import sys
import warnings

from argparse import ArgumentParser
from collections import namedtuple
from os.path import dirname, join, normpath

# Avoid threading issues with Tk, needs to be done before first importing wfdb,
# since wfdb (somehow) uses matplotlib.
import matplotlib
matplotlib.use("agg")


def absify(path, dir):
    """Convert a path to an absolute path."""
    return normpath(join(dir, path))


def all_records_in(path):
    """Find all records in a record directory.
    The given directory must contain no other files!
    """
    return list(set([f.split('.')[0] for f in os.listdir(path)]))


def load_configuration(conf_path):
    """Load and rectify a configuration file.
    Returns a namedtuple to avoid dictionary lookups,
    hence improving readability.
    """
    with open(conf_path, "r") as conf_file:
        conf_dict = json.load(conf_file)
    conf_dir = dirname(conf_path)
    conf_dict["input_dir"] = absify(conf_dict["input_dir"], conf_dir)
    conf_dict["output_dir"] = absify(conf_dict["output_dir"], conf_dir)
    if not "records" in conf_dict:
        conf_dict["records"] = all_records_in(conf_dict["input_dir"])
    if not "test_records" in conf_dict:
        conf_dict["test_records"] = []
    return namedtuple('Configuration', conf_dict.keys())(*conf_dict.values())


parser = ArgumentParser(description=(
    "Evaluate different methods for finding QRS complexes "
    "in single channel ECG signals."))
parser.add_argument(
    "configurations", metavar="CONFIGURATION_FILE", type=str, nargs='+',
    help="Configuration file specifying the evaluation.")
args = parser.parse_args()


sys.path.append(join(dirname(__file__), "./raccoon"))
from raccoon.utils.builders import evaluator_from_dict, NameBuilder


for configuration in args.configurations:

    conf = load_configuration(configuration)

    if not conf.verbose:
        # disable Tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # disable Python warnings
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

    name_builder = NameBuilder()
    evaluator = evaluator_from_dict(conf._asdict(), name_builder)

    if conf.cv_method == "loocv":
        evaluator.loocv()
    elif conf.cv_method == "k2":
        evaluator.kfold(k=2)
    elif conf.cv_method == "k10":
        evaluator.kfold(k=10)
    elif conf.cv_method == 'defined':
        evaluator.defined(conf.test_records)

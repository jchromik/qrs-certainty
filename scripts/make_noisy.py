#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import OrderedDict
from itertools import product
from os.path import dirname, splitext
from shutil import copy

import sys

import wfdb

sys.path.append("../qrsc")
from qrsc.utils.noiseutils import apply_noise_db


# SOURCE:
# module: wfdb.io == 2.2.0
# file: _signal.py
# function: _digi_bounds(fmt)
# lines: 1416--1439
FMT_RANGES = OrderedDict([
    ('80', (-128, 127)),
    ('212', (-2048, 2047)),
    ('16', (-32768, 32767)),
    ('24', (-8388608, 8388607)),
    ('32', (-2147483648, 2147483647))
])


def noisy_filename(filename, noisename, snr):
    name, ext = splitext(filename)
    return "{}_{}_{}{}".format(name, noisename, snr, ext)


def noisy_recordname(recordname, noisename, snr):
    return "{}_{}_{}".format(recordname, noisename, snr)


def adjust_fmt(record):
    """Adjusts data format for serializing a record.
    This is necessary because adding noise to the records signals can cause
    the samples values to be out of the range the format (fmt) can express.
    A digital signal is expected, i.e. d_signal must be present.
    """
    samp_min, samp_max = record.d_signal.min(), record.d_signal.max()
    for fmt, (fmt_min, fmt_max) in FMT_RANGES.items():
        if samp_min < fmt_min or samp_max > fmt_max:
            continue
        record.fmt = [fmt]*len(record.fmt) if isinstance(record.fmt, list) else fmt
        return
    raise RuntimeError("Records samples are too high to be serialized.")


parser = ArgumentParser(description="A tool for adding noise to record files.")
parser.add_argument(
    '-n', '--noise', type=str, required=True,
    help="Record file containing noise template."
)
parser.add_argument(
    '-s', '--snrs', metavar='SNR', type=int, nargs='+', required=True,
    help="Desired signal-to-noise ratios in decibel (dB)."
)
parser.add_argument(
    '-r', '--records', metavar='RECORD', type=str, nargs='+', required=True,
    help="Records to add noise to."
)
parser.add_argument(
    '-a', '--annotations', action='store_true', help="Copy annotation files."
)
parser.add_argument(
    '-d', '--destination', type=str, required=False, default=dirname(__file__),
    help="Directory to write noisy records to."
)

args = parser.parse_args()
args.records = set([splitext(record)[0] for record in args.records])
noise = wfdb.rdrecord(args.noise)

for record_name, snr in product(args.records, args.snrs):
    record = wfdb.rdrecord(record_name)
    record = apply_noise_db(record, noise, snr)

    record.record_name = noisy_recordname(record.record_name, noise.record_name, snr)
    record.file_name = [
        noisy_filename(name, noise.record_name, snr)
        for name in record.file_name]

    record.adc(inplace=True)
    adjust_fmt(record)
    record.wrsamp(write_dir=args.destination)

    if args.annotations:
        copy("{}.atr".format(record_name),
             "{}/{}.atr".format(args.destination, record.record_name))

    print("[DONE] {}".format(record.record_name))

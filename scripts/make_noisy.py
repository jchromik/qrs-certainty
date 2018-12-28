#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import product
from os.path import dirname, splitext
from shutil import copy

import sys

import wfdb

sys.path.append("../raccoon")
from raccoon.utils.noiseutils import apply_noise_db


def noisy_filename(filename, snr):
    name, ext = splitext(filename)
    return "{}e{}{}".format(name, snr, ext)


def noisy_recordname(recordname, snr):
    return "{}e{}".format(recordname, snr)


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

noise = wfdb.rdrecord(args.noise)

for record_name, snr in product(args.records, args.snrs):
    record = wfdb.rdrecord(record_name)
    record = apply_noise_db(record, noise, snr)
    record.record_name = noisy_recordname(record.record_name, snr)
    record.file_name = [noisy_filename(name, snr) for name in record.file_name]
    record.adc(inplace=True)
    record.wrsamp(write_dir=args.destination)
    if args.annotations:
        copy(
            "{}.atr".format(record_name),
            "{}/{}.atr".format(args.destination, record.record_name))

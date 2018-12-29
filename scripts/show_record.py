#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import wfdb

parser = ArgumentParser(description="A tool for showing ECG recordings.")
parser.add_argument(
    'name', type=str, help="Name of the record to show."
)
parser.add_argument(
    '-a', '--annotation', type=str, required=False,
    help="Specify a custom annotation file."
)

args = parser.parse_args()

record = wfdb.rdrecord(args.name)
signals = record.p_signal.T

annotation_name = args.name if args.annotation is None else args.annotation
try:
    ann = wfdb.rdann(annotation_name, 'atr')
    samples, symbols = ann.sample, ann.symbol
except FileNotFoundError:
    samples, symbols = [], []

fig, axes = plt.subplots(nrows=len(signals), sharex=True)
axes = np.array(axes).reshape(-1) # in case len(signal) == 1

for ax, signal in zip(axes, signals):
    ax.plot(signal)
    for symbol in set(symbols):
        x = [samp for samp, sym in zip(samples, symbols) if sym == symbol]
        y = signal[x]
        marker = "${}$".format(symbol) if symbol != '~' else r"$\backsim$"
        ax.scatter(x, y, c='red', marker=marker, s=100)

plt.show()
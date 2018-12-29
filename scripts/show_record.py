#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import wfdb

parser = ArgumentParser(description="A tool for showing ECG recordings.")
parser.add_argument(
    'name', type=str, help="Name of the record to show."
)

args = parser.parse_args()

record = wfdb.rdrecord(args.name)
signals = record.p_signal.T

try:
    ann = wfdb.rdann(args.name, 'atr')
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
        ax.scatter(x, y, c='red', marker="${}$".format(symbol), s=100)

plt.show()
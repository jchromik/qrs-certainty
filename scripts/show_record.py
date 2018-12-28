#!/usr/bin/env python3

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import wfdb

parser = ArgumentParser(description="A tool for showing ECG recordings.")
parser.add_argument(
    'name', type=str, help="Name of the record to show."
)

args = parser.parse_args()

record = wfdb.rdrecord(args.name)

plt.plot(record.p_signal.T[0])
plt.show()
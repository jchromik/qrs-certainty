#!/usr/bin/env python3

from itertools import product
from subprocess import run
from sys import argv

snrs = range(-6, 25, 6)
noise = "nstdb/em"
database = "mitdb"
records = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219",
    "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234"]

for record, snr in product(records, snrs):
    run([
        "nst",
        "-i", "/".join([database, record]), noise,
        "-o", "{}e{}".format(record, snr),
        "-s", str(snr),
        "-F", "212"])

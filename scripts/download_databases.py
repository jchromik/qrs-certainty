#!/usr/bin/env python3

from urllib.request import urlretrieve

import argparse
import itertools
import os

# Database Specification

databases = {
    "mitdb": {
        "remote": "https://physionet.org/physiobank/database/mitdb",
        "files": [
            "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
            "111", "112", "113", "114", "115", "116", "117", "118", "119",
            "121", "122", "123", "124",
            "200", "201", "202", "203", "205", "207", "208", "209",
            "210", "212", "213", "214", "215", "217", "219",
            "220", "221", "222", "223", "228",
            "230", "231", "232", "233", "234"],
        "suffixes": ["atr", "hea", "dat"]},
    "nsrdb": {
        "remote": "https://physionet.org/physiobank/database/nsrdb",
        "files": [
            "16265", "16272", "16273", "16420", "16483", "16539", "16773",
            "16786", "16795", "17052", "17453", "18177", "18184", "19088",
            "19090", "19093", "19140", "19830"],
        "suffixes": ["atr", "hea", "dat"]},
    "nstdb": {
        "remote": "https://physionet.org/physiobank/database/nstdb",
        "files": [
            "118e00", "118e06", "118e12", "118e18", "118e24", "118e_6",
            "119e00", "119e06", "119e12", "119e18", "119e24", "119e_6"],
        "suffixes": ["atr", "hea", "dat"]},
    "noises": {
        "remote": "https://physionet.org/physiobank/database/nstdb",
        "files": ["em", "ma", "bw"],
        "suffixes": ["hea", "dat"]}}

# Parsing CLI Arguments

parser = argparse.ArgumentParser(description=(
    "A tool for downloading waveform databases from Physionet.")
)
parser.add_argument(
    "-d", "--dir", type=str, required=True,
    help="Path to directory the databases are stored in."
)
parser.add_argument(
    "-k", "--db-keys", dest="db_keys", type=str, nargs="+", required=True,
    choices=list(databases.keys()), help="Databases to be downloaded."
)

args = parser.parse_args()

# Downloading Databases

for key in args.db_keys:
    if key not in databases: continue
    db = databases[key]
    db_dir = "/".join([args.dir, key])
    if not os.path.exists(db_dir): os.makedirs(db_dir)

    for f, s in itertools.product(db["files"], db["suffixes"]):
        remote = "{}/{}.{}".format(db["remote"], f, s)
        local = "{}/{}.{}".format(db_dir, f, s)
        print(" --> ".join([remote, local]))
        urlretrieve(remote, local)
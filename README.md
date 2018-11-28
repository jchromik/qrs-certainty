# Project Raccoon

This is part of my master's thesis. I evaluate how neural networks perform on the task of finding QRS complexes in single-channel ECG signals of varying signal quality. The name (raccoon) has no further meaning, I just like raccoons and needed a name. Think of a zealous raccoon searching for QRS complexes!

## What is this?

I compare how different detectors (algorithms searching for QRS complexes) perform. This is done by using the detectors on ECG signals and comparing the QRS complex positions found with the positions of actual QRS complexes in the signal.
Some of these detectors employ neural networks and need a training step, some others employ means of signal processing and do not need training. Whilst the first are subject of this work the latter one establish a baseline to compare the performance of neural networks with.

## How to use this?

For using this, three steps are to be performed:
1. Install requirements
2. Download databases
3. Run evaluation

### Install Requirements

You need Python3 and PIP. Everything else is easily done with:
```shell
pip install -r requirements.txt
```

### Download Databases

We provide a Python script for doing this. Call `download_databases.py` and specify which databases you want to get.

We support:
 * MIT-BIH Arrhythmia Database (`mitdb`)
 * MIT-BIH Normal Sinus Rhythm Database (`nsrdb`)
 * MIT-BIH Noise Stress Test Database (`nstdb`) (for now just the noise signals)

For example, for downloading the `mitdb` to a directory called `data`:
```shell
./download_databases.py -d data -k mitdb
```

### Run Evaluation

Call `run.py` specifying a configuration file. Configurations files tell the evaluator what to do. You can find example configuration files in the `configurations` directory. Feel free to change them according to needs and research questions. Parameters specified in the configuration file are the same the constructors of Evaluator and Detector classes take, so you can use them as reference.

Example call:
```shell
./run.py configurations/example01.json
```

## Which detectors do we compare?

We use three NN-based detectors and three signal processing based detectors.

The NN-based detectors use neural network architectures proposed by the following papers:
 * C. García-Berdonés, J. Narváez, U. Fernández, and F. Sandoval, “A new QRS detector based on neural network,” in Biological and Artificial Computation: From Neuroscience to Technology, 1997, pp. 1260–1269.
 * M. Šarlija, F. Jurišić, and S. Popović, “A convolutional neural network based approach to QRS detection,” in Proceedings of the 10th
 International Symposium on Image and Signal Processing and Analysis, 2017, pp. 121–125.
 * Y. Xiang, Z. Lin, and J. Meng, “Automatic QRS complex detection using two-level convolutional neural network,” Biomed Eng Online, vol. 17, Jan. 2018.

 The signal processing based detectors are GQRS and XQRS by MIT's [wdfb](https://github.com/MIT-LCP/wfdb-python) Python module and a handwritten version of the Pan-Tompkins algorithms.
 
 Cf.: J. Pan and W. J. Tompkins, “A Real-Time QRS Detection Algorithm,” IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230–236, Mar. 1985.

## How do we compare?

For each ECG signal, we do have a list of positions telling us where the QRS complexes actually are. The detectors generate lists of positions where QRS complexes where detected. We compare these lists and produce the classification metrics `true positives`, `true negatives`, `false positives`, and `false negatives`. From this we can compute further classification metrics such as `sensitivity`, `F1 score`, and `positive predictive value`.

This is done by applying the following rules:
 * A detected QRS complex counts as true positive if there is an actual QRS complex within a given range.
 * A detected QRS complex counts as false positive is there is *no* actual QRS complex within a given range.
 * An actual QRS complex having no detected QRS complex within a given range counts as false negative.
 * There are no true negatives.

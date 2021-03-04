# QRS Certainty

We evaluate how neural networks perform on the task of finding QRS complexes in single-channel ECG signals of varying signal quality.

## What is this?

I compare how different detectors (algorithms searching for QRS complexes) perform. This is done by using the detectors on ECG signals and comparing the QRS complex positions found with the positions of actual QRS complexes in the signal.
Some of these detectors employ neural networks and need a training step, some others employ means of signal processing and do not need training. Whilst the first are subject of this work the latter one establish a baseline to compare the performance of neural networks with.

## How to use this?

First, you need to download the ECG databases in use.

You just need:

 * MIT-BIH Arrhythmia Database (`mitdb`)
 * MIT-BIH Noise Stress Test Database (`nstdb`)

This is easily done by:

```shell
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
wget -r -N -c -np https://physionet.org/files/nstdb/1.0.0/
```

Then you can explore the Jupyter notebooks in the `notebooks` folder.

## Which detectors do we employ?

We use three NN-based detectors proposed by the following papers:
 * C. García-Berdonés, J. Narváez, U. Fernández, and F. Sandoval, “A new QRS detector based on neural network,” in Biological and Artificial Computation: From Neuroscience to Technology, 1997, pp. 1260–1269.
 * M. Šarlija, F. Jurišić, and S. Popović, “A convolutional neural network based approach to QRS detection,” in Proceedings of the 10th
 International Symposium on Image and Signal Processing and Analysis, 2017, pp. 121–125.
 * Y. Xiang, Z. Lin, and J. Meng, “Automatic QRS complex detection using two-level convolutional neural network,” Biomed Eng Online, vol. 17, Jan. 2018.


## How do we compare?

For each ECG signal, we do have a list of positions telling us where the QRS complexes actually are. The detectors generate lists of positions where QRS complexes where detected. We compare these lists and produce the classification metrics `true positives`, `true negatives`, `false positives`, and `false negatives`. From this we can compute further classification metrics such as `sensitivity`, `F1 score`, and `positive predictive value`.

This is done by applying the following rules:
 * A detected QRS complex counts as true positive if there is an actual QRS complex within a given range.
 * A detected QRS complex counts as false positive is there is *no* actual QRS complex within a given range.
 * An actual QRS complex having no detected QRS complex within a given range counts as false negative.
 * There are no true negatives.

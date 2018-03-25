# Network Anomaly Detection with discriminative restricted Boltzmann machine (DRBM)

Implementation of Network anomaly detection with the discriminative restricted Boltzmann machine (DRBM) as described in [1].

This code is mostly based on the DRBM Theano implementation in https://bitbucket.org/freakanth/theano-drbm.git

## Requirements
Python 2.7

## How to run

```console
# Install required python packages
pip install -r requirements.txt
# Download kddcup.data_10_percent dataset
wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
# Download kddcup.data_10_percent dataset
wget corrected.gz Test data with corrected labels.
python script.py
```

[1] https://www.sciencedirect.com/science/article/pii/S0925231213005547 

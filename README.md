# Network Anomaly Detection with discriminative restricted Boltzmann machine (DRBM)

Implementation of Network anomaly detection with the discriminative restricted Boltzmann machine (DRBM) as described in [1].

This code is mostly based on the DRBM Theano implementation in https://bitbucket.org/freakanth/theano-drbm.git

## Requirements
Python 2.7

## How to run
```bash
# Install required python packages
pip install -r requirements.txt
# Download kddcup.data_10_percent (10% subset of the dataset)
wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
# Download corrected.gz (test data with corrected labels).
wget http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
python script.py
```

[1] https://www.sciencedirect.com/science/article/pii/S0925231213005547 

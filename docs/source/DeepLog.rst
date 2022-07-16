July 16th 2022 - DeepLog
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. This page analyses the Python implementation of DeepLog, an 
anomaly detection model authored by Du et al., 2017. DeepLog uses bi-directional 
LSTM layers to identify anomalies. 

| *Note about DeepLog* 
It appears that multiple DeepLog implementations exist online. This documentation 
studies the LogPAI implementation within the loglizer repository. Future docs 
may study other implementations to identify any differences.  

| *DeepLog overview* 
DeepLog uses an LSTM RNN model to predict anomalies from a sequence of "log keys", 
or event templates, and parameter values. Du et al. describes two LSTM networks in 
their original paper: "Execution Path" anomaly detection and "Parameter Value" 
performance anomaly detection (2017). The LogPAI implementation appears to focus 
solely on Execution Path anomalies, given that the parsed HDFS results do not 
contain parameter arrays. This is something that must be investigated. 


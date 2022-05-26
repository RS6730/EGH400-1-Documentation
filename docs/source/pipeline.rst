May 27th 2022 - pipeline (log parsing)
===================================================================================
| *Research Goal* 
A script was used to form a full end-to-end pipeline of the log transformation 
process. The purpose of this script is to collects raw logs, parse them using
state-of-the-art log parsing methods, then pass the parsed logs to detection models 
for identifying anomalies. 

| *pipeline application.* 

The pipeline script (main.py) is used to run Drain, Spell and LenMa on a batch of 
log files. 
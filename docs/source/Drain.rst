May 13th 2022 - SP-1 Log Parsing - Drain
===================================================================================
| *Research Goal* 
Drain was the first log parser that was evaluated as part of SP-1. The purpose of 
this subproject is to understand the log transformation process (parsing raw logs). 
This is part of the broader goal to empirically test existing anomaly detection 
models. 

| *Drain application* 

Fork the Automate-Log-Processing repository using Git. The original WS-DREAM repositories
are included as subdirectories (logparser-master, loglizer-master), and contain all the 
sample log datasets, log parsing and anomaly detection models that are required. 

Within logparser-master, navigate to the custom_demo directory. This directory includes 
scripts for running the log-parser demos. This seperate directory is created to enable 
extension/modification from the original demo scripts written by the LogPAI team, found 
in the demo directory. 

To run drain, ensure the input_dir and output_dirs are set, and the code points to the 
correct log file (which must be in the input_dir): 

.. code-block:: python 

    input_dir  = <directory to log set>. 
    output_dir = <results directory>
    log_file   = <log file name>
    log format = <format of the log set>






| *Algorithm Explanation* 


| *Results* 
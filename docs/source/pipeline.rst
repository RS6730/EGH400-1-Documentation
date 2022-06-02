May 27th 2022 - pipeline (log parsing)
===================================================================================
| *Research Goal* 
A script was used to form a full end-to-end pipeline of the log transformation 
process. The purpose of this script is to collects raw logs, parse them using
state-of-the-art log parsing methods, then pass the parsed logs to detection models 
for identifying anomalies. 

.. note:: 

   This script is under active development. Currently, the script only runs 
   parsers on a batch of provided log files. 

| *pipeline application.* 

The [current] pipeline script (main.py) is used to run log parsers on a range of 
different log datasets. This script provides a single location to test different 
log parsers, without having to manually execute different scripts. Furthermore, a 
trace option redirects and saves the parsing output to a text file for later use. 

To run the pipeline script, navigate to the pipeline directory and enter the 
following command: 

.. code-block:: shell 

    python3 main.py [TRACE: on/off] [LOG FILES: file_1 file_2 ... file_n]

TRACE indicates whether print statements should be redirected to a tracefile.
LOG_FILES is a list of log files the pipeline should use - main.py checks the 
logs directory for the supplied log files. If a log file does not exist for a 
supplied name, it is ignored. 

One limitation is the log parsers to be used must manually be specified in 
main.py - this will be ammended in future updates. 


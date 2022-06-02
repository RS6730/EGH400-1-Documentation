May 27th 2022 - pipeline (log parsing)
===================================================================================
Note: this documentation was written and committed on June 2nd 2022. 

| *Research Goal* 
A script was used to form a full end-to-end pipeline of the log transformation 
process. The purpose of this script is to collects raw logs, parse them using
state-of-the-art log parsing methods, then pass the parsed logs to detection models 
for identifying anomalies. 

.. note:: 

   This script is under active development. Currently, the script only runs 
   parsers on a batch of provided log files. 

| *Pipeline script application.* 

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
main.py - this will be ammended in future updates. Currently, Linux logs are 
the only log type that can be analysed, as SP-1 involves working with Linux. 

| *Pipeline script code explanation.* 

The pipeline acts as an interface between log parsers and their outputs. The core 
script begins by reading the command line inputs to activate trace and obtain the 
log files for use. 

.. code-block:: python3
    print("Starting...")
    trace = False 
    trace_file = None

    if sys.argv[1] == "on":
        # trace is on. Redirect print to tracefile. 
        trace = True
        trace_file = "pipeline_run_" + str( len(os.listdir('traces/')) + 1 ) + ".txt"
        sys.stdout = open("traces/" + trace_file, "w")
        print("Beginning pipeline run...\n")

    if len(sys.argv[2:]) < 1: 
        print("No log data provided. Exiting.\n")
        sys.exit(1)

    log_files = generateData(sys.argv[2:])

The generateData function evaluates the log file names that have been passed in, 
checks to see if they exist in the /logs directory, and returns a list of valid 
log files (wit the '.logs' extention appended) to use. 

.. code-block:: python3
    def generateData(inputs): 
        '''
        Check if the log-file exists. If yes, add file extention and append to list. 
        '''
        logs = [] 
        for log_set in inputs:
            name = log_set + ".log"
            if name in os.listdir('logs'):
                logs.append(name)
            else: 
                print(f"In generateData -> Ignoring {log_set}: not found\n")
        
        print(f"In generateData -> Using the following log files: {logs}\n")
        return logs 

After this log dataset is created, the constants for log parsing are declared. 

.. code-block:: python3
    # Constants for log parsing.
    input_dir  = 'logs/'
    output_dir = 'results/'
    # format and regex configured for Linux logs. 
    log_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
    regex = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']

This is followed by calling the runParser function to run the log parsers. The following 
example runs all the LogParsers - currently, these lines can be commented/uncommented to 
specify which parser to run. 

.. code-block:: python3 
    # Run log parsers. 
    # -- online 
    runParser(log_files, 'drain', input_dir, output_dir, log_format, regex, trace, trace_file)
    runParser(log_files, 'spell', input_dir, output_dir, log_format, regex, trace, trace_file)
    runParser(log_files, 'lenma', input_dir, output_dir, log_format, regex, trace, trace_file)    
    # -- offline 

The runParser function works by iterating through each log file in the log_files list, 
and calls the parserSetup factory method to generate a log parser object. The type string 
specifies which log parser to create. The returned LogParser object will be used to 
parse the current log file. A new LogParser is created for each log file. This is by design, 
as re-using a parser on a new dataset mixes the data from seperate log files, which is undesireable. 

.. code-block:: python3 
    def runParser(log_files, type, input_dir, output_dir, log_format, rgx, trace, trace_file):
    '''
    Run each log parser. Print the outputs to the trace file. 
    '''    
    customPrint(f"Running {type} on all log sets.", trace, trace_file)
  
    print(f"--------------Starting {type} parsing------------------\n")
    for file in log_files: 
        customPrint(f"Started parsing {file} with {type}", trace, trace_file)

        print(f"Started parsing {file} with {type}")
        # create a new parser for each log file. 
        parser = parserSetup(type, input_dir, output_dir, log_format, rgx)
        parser.parse(file) 
        print(f"Ended parsing {file} with {type} \n")

    print(f"--------------Ending {type} parsing------------------\n")

The parserSetup function takes the input and output directories, log format and 
regex as parameters, and creates the required log parser based on the type string. 
This method acts as a 'factory' for generating the requested log parser. 

.. code-block:: python3 
    def parserSetup(option, input, output, format, rgx):
        '''
        Setup function for generating a log parser object. 
        '''
        parser = None
        if option == 'drain': 
            depth = 4 
            st = 0.39
            output = output + 'Drain/'
            parser = Drain.LogParser(format, indir=input, outdir=output, depth=depth, st=st, rex=rgx)

        if option == 'spell':
            tau = 0.55
            output = output + 'Spell/'
            parser = Spell.LogParser(indir=input, outdir=output, log_format=format, tau=tau, rex=rgx)

        if option == 'lenma': 
            threshold = 0.88
            output = output + 'LenMa/'
            parser = LenMa.LogParser(input, output, format, threshold=threshold, rex=rgx)
        
        print(f"In parserSetup -> generated {parser} for {option}")
        return parser

Finally, the customPrint method is used to temporarily turn-off the trace and output the  
parsing progress to the command line. Redirection to trace is resumed. 

Two sets of results are kept once main.py completes. Firstly, the results/ directory contains 
any parsing results ('.log_structured.csv' and '.log_templates.csv'). Secondly, the trace/ directory 
contains recent and historical trace files, which can be used for debugging. Trace files are automatically
named by counting the number of existing runs - thus, the most recent trace file will have the highest 
number. 

| *Resetting Directories*

The results/ and trace/ directories can be reset by calling reset_directories.py. This is useful for 
resetting the pipeline workspace without manually deleting files. However, this script must only be 
used once all log parsing and trace results are saved. The results_locked/ directory exists as a 
save point. 

To reset directories, use: 

.. code-block:: shell 

    python3 reset_directories.py 

A warning will appear to suggest saving all results. To proceed manually enter 'Delete All' to 
continue deleting the results. 

| *Future Work* 

main.py will be extended to include anomaly detection in the future. The ultimate goal is to 
create a script that performs the end-to-end process of parsing logs and detecting anomalies. 





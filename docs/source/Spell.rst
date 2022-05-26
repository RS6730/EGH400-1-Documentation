May 27th 2022 - Spell 
===================================================================================
| *Research Goal* 
Spell was the second log parser that was evaluated as part of SP-1. The purpose of 
this subproject is to understand the log transformation process (parsing raw logs). 
This is part of the broader goal to empirically test existing anomaly detection 
models. 

| *Spell application* 

Note: the code presented in this page is authored by the LogPAI team. As part of this 
research project, additional scripts were created to explore and understand the original 
source code. 

Fork the Automate-Log-Processing repository using Git. The original WS-DREAM repositories
are included as subdirectories (logparser-master, loglizer-master), and contain all the 
sample log datasets, log parsing and anomaly detection models that are required. 

Within logparser-master, navigate to the custom_demo directory (added manually, and not
found in the original WS-DREAM fork) This directory includes scripts for running the log-parser demos. 
This seperate directory is created to enable extension/modification from the original 
demo scripts written by the LogPAI team, found in the demo directory. 

To run spell, ensure the input_dir and output_dirs are set, and the code points to the 
correct log file (which must be in the input_dir): 

.. code-block:: python 

    # Author: LogPAI team. 

    input_dir  = '<directory to log set>' 
    output_dir = '<results directory>'
    log_file   = '<log file name>'
    log_format = '<format of the log set>'

To test spell on linux, these variables can be left as their default values.
Furthermore, when testing different log files, ensure the log_format is updated to suit 
the type of log that is being used - log formats can be found in the benchmark directory. 

To ensure parameters are extracted during parsing, ensure the regex list is updated. For 
Linux, this is:

.. code-block:: python 
    
    # Author: LogPAI team. 

    regex = [
    r'(\d+\.){3}\d+',             # IP address of format ##.###.##.##
    r'\d{2}:\d{2}:\d{2} \d{4}',   # Timestamp 00:00:00 0000
    ]

Regex patterns for each log can be found in the benchmark directory. Additional patterns 
can be added, based on domain specific knowledge of the logs. 

The tau value (LCS threshold) can be left at the default value. 

To run the log parser, simply run the demo script. 

.. code-block:: shell 
    python3 customSpellDemo.py 

The parsing progress will be displayed on the command prompt, and the results directory 
(output_dir) will update with two csv result files, once parsing is complete. The 
<Log_set>.log_structured.csv file contains the entire parsing result set. The 
<Log_set>.log_templates.csv file contains information on event templates and their occurences. 

| *Spell - Algorithm and Code Explanation* 

Spell (Streaming Parser for Event Logs using LCS) is a log-parsing method that uses 
the longest common subsequence (LCS) approach for processing raw logs. LCS is an approach used 
to identify sequence similarities between two strings. Di & Li (2016) reason that, as log messages
are formed by constant string messages, these messages can be evaluated using LCS to identify 
common logging groups in a set of system logs. 

Spell works by using the following series of steps: 
    1. tokenize an incoming log sequence (using whitespace)
    2. For each LCSObject (that has previously been created), find the LCS between  
       the log template in that object, and the current log message. Save the maximum 
       LCS score, which is checked against a similarity threshold. 
    3. Update the LCSObject that has the highest LCS score with the current log message. 
       Add the ID of the current log to the LCSObject. 

To complete LCS in O(N) time complexity, Spell uses a tree structure to check if the 
sequence has already been parsed. This 'prefix' tree uses each token in the log 
message as a child node of the preceeding token, where the final leaf node contains the 
LCS object of that sequence. 

The LCSObject is a class that represents a cluster (group) of logs that contain 
a given log template.  

.. code-block:: python 

    # Author: LogPAI team

    class LCSObject:
        """ Class object to store a log group with the same template
        """
        def __init__(self, logTemplate='', logIDL=[]):
            self.logTemplate = logTemplate
            self.logIDL = logIDL

Much like Drain.py, the main method used by Spell.py parse method. After creating Regex 
named groups to 'tokenize' the log message (based on the log format), the log set is 
loaded into a Pandas DataFrame. The parse method iterates over this DataFrame to evaluate 
each log line. 

Firstly, the PrefixTreeMatch method is called, as part of the pre-filtering step to check 
if a suitable LCSObject already exists for this log - this prevents manually computing the 
LCS for each incoming log. 

.. code-block:: python 

    # Author: LogPAI team. 

    def PrefixTreeMatch(self, parentn, seq, idx):
        """
        Params: 
            - parentn - the current node 
            - seq - the current log message (not counting <*> parameters)
            - idx - lowerbound for iteration 
        Returns: a matching log cluster or None. 
        """
        retLogClust = None
        length = len(seq)

        # traverse through the prefix-tree. 
        for i in range(idx, length):
            
            # current token is a child node, then traverse to next token.
            if seq[i] in parentn.childD: 
                childn = parentn.childD[seq[i]]
                
                # check if a log cluster exists as a leaf. If so, then this
                # token sequence has already been parsed. 
                if (childn.logClust is not None):
                    
                    constLM = [w for w in childn.logClust.logTemplate if w != '<*>']
                    
                    if float(len(constLM)) >= self.tau * length:
                        return childn.logClust

                # if a log cluster leaf does not exist, recursvely continue 
                # down the tree.
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        # return the log cluster (either valid or None)
        return retLogClust

The PrefixTreeMatch method traverses down a path from the root node (this path 
forms the token sequence) until an LCSObject is found - if an LCSObject exists, then the 
LCS of this token sequence has been considered before. Thus, the LCSObject for this 
sequence is returned (otherwise, an empty LCSObject is returned). 

If an empty LCSObject is returned, the parse method proceeds with checking if a second 
prefiltering step to verify if the log message has been parsed. 

.. code-block:: python 

    # Author: LogPAI team. 

    def SimpleLoopMatch(self, logClustL, seq):
        """
        Params:
            - logClustL: list of current log clusters
            - current log sequence (array of tokens, not counting parameters)
        """
        # iterate through the LCSMap (the log cluster list) 
        for logClust in logClustL:

            # integrity check: ensure the parsed event template is sufficiently 
            # long enough. 
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
                
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)

            # check if the token in the log sequence is in the token_set, or if the token is a parameter <*>  
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
                # if all cluster tokens are in the current log message, 
                # it is a subsequence. 
                return logClust
        return None

The SimpleLoopMatch method iterates through the current list of LCSObjects (LCSMap) to check if a subsequence of 
the current log message exists. If it does, the LCSObject for this subsequence is returned. 

If the SimpleLoopMatch method returns none, Spell finally performs a manual LCS search using the LCSMatch 
method. 

.. code-block:: python 

    # Author: LogPAI team. 

    def LCSMatch(self, logClustL, seq):
        """
        Params:
            - the log cluster list. 
            - the log message (content, with parameters)
        """
        retLogClust = None
        maxLen = -1 
        maxlcs = []
        maxClust = None

        set_seq = set(seq)  # unique elements in log message 
        size_seq = len(seq) # size of the log message 

        # iterate through all LCSObjects
        for logClust in logClustL:

            set_template = set(logClust.logTemplate) 

            # perform an integrity check
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue

            # get the LCS for this log sequence, and the current log template. 
            lcs = self.LCS(seq, logClust.logTemplate)

            # check if this is the largest LCS. 
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen = len(lcs)
                maxlcs = lcs
                maxClust = logClust

        # perform the integrity check with the LCS threshold.
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        # return this log cluster (none, if the above check fails)
        return retLogClust

The LCS match method fulfils the second step in the three step process outlined above. 

After this function, if a suitable LCSObject is not found, then a new LCSObject is created for this log sequence 
(all pre-filtering and manual LCS checks have failed, so this log sequence currently unique). This new LCSObject 
is appended to the LCSMap. The prefix-tree is updated, so this log sequence can be checked in future iterations. 

If LCSMatch returns a suitable LCSObject, then a getTemplate method is used to get an updated log template. The prefix
tree is also updated to include this new sequence. 

| *Results* 



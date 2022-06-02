May 13th 2022 - Drain
===================================================================================
| *Research Goal* 
Drain was the first log parser that was evaluated as part of SP-1. The purpose of 
this subproject is to understand the log transformation process (parsing raw logs). 
This is part of the broader goal to empirically test existing anomaly detection 
models. 

| *Drain application* 

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

To run drain, ensure the input_dir and output_dir is set, and the code points to the 
correct log file (which must be in the input_dir): 

.. code-block:: python 

    # Author: LogPAI team. 

    input_dir  = '<directory to log set>' 
    output_dir = '<results directory>'
    log_file   = '<log file name>'
    log_format = '<format of the log set>'

To test Drain on linux, these variables can be left as their default values.
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

The st (similarity threshold) and depth (fixed-tree depth) can be left at their default 
values. 

To run the log parser, simply run the demo script. 

.. code-block:: shell 
    python3 customDrainDemo.py 

The parsing progress will be displayed on the command prompt, and the results directory 
(output_dir) will update with two csv result files, once parsing is complete. The 
<Log_set>.log_structured.csv file contains the entire parsing result set. The 
<Log_set>.log_templates.csv file contains information on event templates and their occurences. 

| *Drain - Algorithm and Code Explanation* 

Drain uses a fixed-depth tree, where the internal nodes of the tree contain parsing rules, and the final leaf nodes 
contain a list of candidate log clusters (He et al., 2017). 

The tree has three levels after the root node: (1) nodes in the first level determine the length of the log message, (2) nodes in the 
second level check the tokens of the log message, and (3) final leaf nodes contain a list of candidate log clusters (He 
et al., 2017). Incoming logs are parsed by traversing through the internal nodes of the tree until a leaf node is found
(He et al., 2017). The similarity between the current log message and each existing template in the leaf node is 
calculated. Based on a similarity threshold, the log message is either appended to a cluster, or a new cluster is created. 

With respect to code, Drain is implemented by using a combination of regular expressions and Python objects for 
clusters and tree nodes. As per the logparser-master/logparser/Drain/Drain.py script, log clusters and tree nodes
are represented by the following classes: 

.. code-block:: python

    # Author: LogPAI team. 

    class Logcluster:
        def __init__(self, logTemplate='', logIDL=None):
            self.logTemplate = logTemplate
            if logIDL is None:
                logIDL = []
            self.logIDL = 
    
    class Node: 
        def __init__(self, childD=None, depth=0, digitOrtoken=None):
            if childD is None:
                childD = dict()
            self.childD = childD
            self.depth = depth
            self.digitOrtoken = digitOrtoken ## ?? 

The LogParser class (in Drain.py) instantiates LogCluster and Node classes to created a fixed-depth tree. 
The main method used by LogParser is the parse method, which is called in the demo script. This parse method has 
three broad phases – loading the data, identifying the most suitable log cluster, and updating the tree. 

Data loading occurs across two methods: 

.. code-block:: python 

    # Author: LogPAI team 

    def load_data(self):
        # generate the regex named group from the log_format. 
        headers, regex = self.generate_logformat_regex(self.log_format) 

        # use this named group to isolate substrings in each log-line, and create
        # a Pandas DataFrame. 
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


Firstly, the generate_logformat_regex method uses the log_format  to construct a named 
group regex object. For example, the log format string that was created for Linux is:

    log_format = '<Month> <Date> <Time> <Level> <Component> <PID> <Content>'

From this, the regex object that is constructed is: 

    re.compile('^(?P<Month>.*?)\\s+(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?)\\s+'(?P<PID>.*?)\\s+(?P<Content>.*?)\\s+)

This regex object can be used to isolate different substrings in a log message, effectively tokenizing the log message 
based on the format that is provided. Furthermore, generate_logformat_regex returns a list of headers, which are the 
properties within the < > braces in the log_format string. 

These objects are passed to the log_to_dataframe method to construct a Pandas dataframe. log_to_dataframe iterates 
over each log entry in the log file, 'tokenizes' it using the regex object, appends it to a log messages list, and 
returns a dataframe of logs that are categorized by the headers provided. 

The next stage of the parse method is identifying the most suitable log cluster in the fixed-depth tree. The treeSearch
method is used to traverse down the nodes of the tree, which is represented as a series of “Node” objects. 

.. code-block:: python 

    # Author: LogPAI team 

       def treeSearch(self, rn, seq):
        """
        Inputs:
            rn  : root node for the tree. 
            seq : current log message (preprocessed) 
        """
        retLogClust = None

        # Check length first. 
        seqLen = len(seq)               
        if seqLen not in rn.childD:     
            return retLogClust       

        # Traverse down the tree.   
        parentn = rn.childD[seqLen]
        currentDepth = 1

        # Check tokens next. 
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break
            
            # Traverse down the tree. 
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                # Suitable child node not found, 
                # return
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        # Check leaf node for a suitable cluster. 
        retLogClust = self.fastMatch(logClustL, seq)
        return retLogClust

The treeSearch method uses the ‘content’ of the log message (the part of the log that forms the event template). 
The length of the log content is checked first. If a length node exists, it becomes the new parent node (traverse down by one layer). 
Next, the first token of the log is evaluated – if a node for this token exists, then this becomes the new parent node (again, 
traverse down by one layer). At this depth, the parent node is a token, and the child node is a leaf with a list of log clusters. 
The treeSearch method calls fastMatch, a method to identify the cluster that best matches the current log message. If a suitable log cluster 
is found (based on a similarity threshold), it is returned. While traversing the tree, if child nodes for length and token do not exist, then a 
LogCluster object with ‘None’ as the value is returned.

Finally, the parse method updates the fixed-depth tree. If a suitable log cluster was not identified (as the child nodes 
do not exist), then a new cluster is created with the current log message content and log ID. The addSeqToPrefixTree
method creates new length and token nodes in the tree (if required) and stores the log message as a new cluster
(represented as a list of log cluster objects, initially with a size of one). 

.. code-block:: python 

    # Author: LogPAI team 

    def addSeqToPrefixTree(self, rn, logClust):

        seqLen = len(logClust.logTemplate)

        # Check if length node exists. If not, create 
        # this node. 
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            # Traverse down the tree. 
            firtLayerNode = rn.childD[seqLen]
        parentn = firtLayerNode
        currentDepth = 1

        for token in logClust.logTemplate:
            
            # Do this last. Conditional here is executed once 
            # we traverse to the token node. 
            if currentDepth >= self.depth or currentDepth > seqLen:
            # Add log cluster to the group. 
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # Find the token node. 
            if token not in parentn.childD:
                if not self.hasNumbers(token):

                    # Special node <*> used to prevent 
                    # branch explosion. Check if this exists. 
                    if '<*>' in parentn.childD:

                        # Create a new token node.
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        
                        # Traverse to special node. 
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        # Create a new token node.
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        
                        # Create a special node.
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode

                        # Traverse to special node. 
                        else:
                            parentn = parentn.childD['<*>']
                else:
                    # Create a new special node. Traverse to this node.
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']
            else:
                parentn = parentn.childD[token]
            currentDepth += 1


If a matching cluster was identified, then the Log ID of the current log message is appended to the cluster. 
This process occurs for every log entry in the dataframe. 

Once all logs have been evaluated, the outputresult method is called to update the dataframe with event templates 
and log ID’s. This dataframe is saved with the _structured.csv label. A separate dataframe that counts the occurrences
of each event template is also created. This is saved with the _templates.csv label. 

| *Results* 

Preliminary tests with Drain were conducted without the optional regex for preprocessing, leading to poorly parsed 
Linux logs (where the parameters were not extracted). Subsequent tests included the regex preprocessing. 
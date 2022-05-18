May 13th 2022 - SP-1 Log Parsing 
===================================

*Research Goal* 
Fork the WS-DREAM repository and begin understanding the log transformation 
process (parsing raw logs). This is part of the broader goal to empirically 
test existing anomaly detection models.

*Outcomes* 
The Python script for the Drain log parser was examined this week. This parser was selected after reviewing Zhu et 
al.’s (2019) research on benchmarking different log parsing tools. In this study, Drain exhibited the highest accuracy 
with minimal variation when tested on various logs. Drain uses a fixed-depth tree, where the internal nodes of the 
tree contain parsing rules, and the final leaf nodes contain a list of candidate log clusters (He et al., 2017). 

The tree has three levels after the root node: (1) nodes in the first level determine the length of the log message, (2) nodes in the 
second level check the tokens of the log message, and (3) final leaf nodes contain a list of candidate log clusters (He 
et al., 2017). Incoming logs are parsed by traversing through the internal nodes of the tree until a leaf node is found
(He et al., 2017). The similarity between the current log message and each existing template in the leaf node is 
calculated. Based on a similarity threshold, the log message is either appended to a cluster, or a new cluster is created. 

With respect to code, Drain is implemented by using a combination of regular expressions and Python objects for 
clusters and tree nodes. The main method for Drain is parse, which takes a log file as input. This parse method has 
three broad phases – loading the data, identifying the most suitable log cluster, and updating the tree. Data loading 
occurs across two methods. Firstly, a generatelogformat_regex method uses a log format string to construct a named 
group regex object. For example, the log format string that was created for Linux is:

    log_format = '<Month> <Date> <Time> <Level> <Component> <PID> <Content>'

From this, the regex object that is constructed is: 

    re.compile('^(?P<Month>.*?)\\s+(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?)\\s+'(?P<PID>.*?)\\s+(?P<Content>.*?)\\s+)

This regex object can be used to isolate different substrings in a log message, effectively tokenizing the log message 
based on the format that is provided. Furthermore, generatelogformat_regex returns a list of headers, which are the 
properties within the < > braces in the log_format string. These objects are passed to the log_to_dataframe method
to construct a Pandas dataframe. log_to_dataframe iterates over each log entry in the log file, tokenizes it using the 
regex object, appends it to a log messages list, and returns a dataframe of logs that are categorized by the headers
provided. 

The next stage of the parse method is identifying the most suitable log cluster in the fixed-depth tree. The treeSearch
method is used to traverse down the nodes of the tree, which is represented as a series of “Node” objects, where child 
nodes are stored in a dictionary. The treeSearch method uses the ‘content’ of the log message (the part of the log that 
forms the event template). The length of the log content is checked first. If a length node exists, it becomes the new 
parent node (traverse down by one layer). Next, the first token of the log is evaluated – if a node for this token exists, 
then this becomes the new parent node (again, traverse down by one layer). At this depth, the parent node is a token, 
and the child node is a leaf with a list of log clusters. treeSearch calls fastMatch, a method to identify the cluster that 
best matches the current log message. If a suitable log cluster is found (based on a similarity threshold), it is returned.
While traversing the tree, if child nodes for length and token do not exist, then a LogCluster object with ‘None’ as the 
value is returned.

Finally, the parse method updates the fixed-depth tree. If a suitable log cluster was not identified (as the child nodes 
do not exist), then a new cluster is created with the current log message content and log ID. The addSeqToPrefixTree
method creates new length and token nodes in the tree (if required) and stores the log message as a new cluster
(represented as a list of log cluster objects, initially with a size of one). If a matching cluster was identified, then the 
Log ID of the current log message is appended to the cluster. This process occurs for every log entry in the dataframe. 
Once all logs have been evaluated, the outputresult method is called to update the dataframe with event templates 
and log ID’s. This dataframe is saved with the _structured.csv label. A separate dataframe that counts the occurrences
of each event template is also created. This is saved with the _templates.csv label. 

*References*
He, P., Zhu, J., Zheng, Z., Lyu, M. R. (2017, June 25-30). Drain: An Online Log Parsing Approach with Fixed Depth Tree
[Conference paper]. 2017 IEEE International Conference on Web Services (ICWS), Honolulu, HI, USA. 
https://doi.org/10.1109/ICWS.2017.13

Zhu, J., He, S., Liu, J., He, P., Xie, Q., Zheng, Z. (2019, May 27). Tools and benchmarks for automated log parsing
[Conference paper]. 41st International Conference on Software Engineering, Montreal, Quebec, Canada. 
https://doi.org/10.1109/ICSE-SEIP.2019.00021



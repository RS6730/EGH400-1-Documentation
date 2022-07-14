May 27th 2022 - LenMa
===================================================================================
Note: this documentation was written and committed on June 2nd 2022. 

| *Research Goal* 
LenMa was the final log parser that was evaluated as part of SP-1. The purpose of 
this subproject is to understand the log transformation process (parsing raw logs). 
This is part of the broader goal to empirically test existing anomaly detection 
models. 

| *LenMa application*

Note: the code presented in this page is authored by the LogPAI team. As part of this 
research project, additional scripts were created to explore and understand the original 
source code. 

The pipeline script can be used to run LenMa on Linux logs. 

Fork the Automate-Log-Processing repository using Git. The original WS-DREAM repositories
are included as subdirectories (logparser-master, loglizer-master), and contain all the 
sample log datasets, log parsing and anomaly detection models that are required. 

Within logparser-master, navigate to the custom_demo directory (added manually, and not
found in the original WS-DREAM fork) This directory includes scripts for running the log-parser demos. 
This seperate directory is created to enable extension/modification from the original 
demo scripts written by the LogPAI team, found in the demo directory. 

To run LenMa, ensure the input_dir, output_dir and regex is set, and the code points to the correct 
log file (which must be in the input_dir). 

.. code-block:: python 

    input_dir  = '<directory to log set>'
    output_dir = '<results directory>'
    log_file   = '<log file name>'
    log_format = '<format of the log set>'
    regex = '<[Regex patterns for the log set]>'

To test LenMa on Linux, these variables can be left at their default values (in the 
customLenMaDemo.py script). Furthermore, leave the threshhold variable at the default 
value (0.9). 

To run LenMa, simply execute the python script. 

.. code-block:: shell 
    python3 customLenMaDemo.py 

The parsing progress will be displayed on the command prompt, and the results directory 
(output_dir) will update with two csv result files, once parsing is complete. The 
<Log_set>.log_structured.csv file contains the entire parsing result set. The 
<Log_set>.log_templates.csv file contains information on event templates and their occurences. 

| *LenMa Algorithm and Code Explanation - LenMa.py* 

The LenMa ('Length Matters') approach to log parsing uses token length and positions to 
organise log messages into clusters. This online method compares incoming logs to existing 
log clusters (by calculating the length and token similarity). The incoming log message is either appended to 
an existing cluster, or a new cluster is created (Shima, 2016). 

The LenMa LogParser object relies on the LenmaTemplateManager class to cluster log messages.
The constructor of the LenMa LogParser object is provided below - note the templ_mgr object, 
which is an instance of the LenmaTemplateManager class. This object is used for clustering. 

.. code-block:: python 

    # Author: LogPAI team. 

    def __init__(self, indir, outdir, log_format, threshold=0.9, predefined_templates=None, rex=[]):
        self.path = indir
        self.savePath = outdir
        self.logformat = log_format
        self.rex = rex
        self.wordseqs = []
        self.df_log = pd.DataFrame()
        self.wordpos_count = defaultdict(int)
        self.logname = None

        # template manager object to handle clustering 
        self.templ_mgr = lenma_template.LenmaTemplateManager(threshold=threshold, predefined_templates=predefined_templates)

As with Drain and Spell, the LogParser class in LenMa.py contains methods for tokenizing log 
messages using regex named groups, and representing the logs in a dataframe. The parse method 
iterates through each log message. The 'Content' of the current log line is retrieved - this 
forms the event template. The parameters are filtered using regex, and the resulting string 
is represented as a list using split(). This list, and the LogID, are passed to the LenmaTemplateManager.infer_template 
function for clustering. 

.. code-block:: python 

    # Author: LogPAI team

    def parse(self, logname):
        print('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        starttime = datetime.now()
        
        ## tokenize the log line using the log format. 
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logname), regex, headers, self.logformat)
        
        for idx, line in self.df_log.iterrows():
            line = line['Content']
            if self.rex:
                for currentRex in self.rex:
                    line = re.sub(currentRex, '<*>', line)
            words = line.split()

            # clustering occurs in the infer_template function 
            self.templ_mgr.infer_template(words, idx)
        
        self.dump_results()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

Once the clustering is complete, the dump_results() method is called to save the parsing results in csv files -
"_templates.csv" and "_structured.csv" results files are created. 

| *LenMa Algorithm and Code Explanation - lenma_template.py*

The lenma_template.py script holds the core logic for LenMa - this script is written by the original authors of 
LenMa, as per the WS-DREAM repository README on LenMa. 

The LenmaTemplateManager class has two class attributes: templates (list) and threshold (float). The templates
list holds a list of template objects, which are instances of the LenmaTemplate class. The constructors for 
both classes are represented below. 

.. code-block:: python 

    # Author: Shima, K. (2016) 

    # -- Class representing an event template cluster.. 
    class LenmaTemplate(Template): 
        def __init__(self, index=None, words=None, logid=None, json=None):
            if json is not None:
                # restore from the jsonized data.
                self._restore_from_json(json)
            else:
                # initialize with the specified index and words vlaues.
                assert(index is not None)
                assert(words is not None)
                self._index = index # 
                self._words = words # -- the log event content 
                self._nwords = len(words) # -- length of the event message 
                self._wordlens = [len(w) for w in words] # -- word length vector 
                self._counts = 1
                self._logid = [logid]

    # -- Class for managing clusters (control flow for creating/appending incoming 
    # -- log messages to new clusters)
    class LenmaTemplateManager(TemplateManager):
        def __init__(self, threshold=0.9, predefined_templates=None):    
            # -- hold all template objects
            self._templates = [] 
            # -- threshold for similarity 
            self._threshold = threshold

The LenmaTemplateManager class contains the infer_template method, which is used to evaluate incoming 
log messages and update the self._templates list. 

.. code-block:: python 

    # Author: Shima, K. (2016) 

    def infer_template(self, words, logid):
        # -- get the length of the words list
        nwords = len(words)
        # -- keep track of candidate templates 
        candidates = [] 

        # -- iterate through the templates list 
        for (index, template) in enumerate(self.templates):
            
            # -- skip, if length does not match 
            if nwords != template.nwords:
                continue

            # -- get the cosine similarity score between the current cluster
            # -- and the incoming log message
            score = template.get_similarity_score(words)
            
            # -- integrity check - ensure the similarity is sufficiently 
            # -- high
            if score < self._threshold:
                continue

            # -- add this template to the candidates list
            candidates.append((index, score))

        # -- sort by score similarity score. 
        candidates.sort(key=lambda c: c[1], reverse=True)

        if False:
            # never execute. 
            for (i,s) in candidates:
                print('    ', s, self.templates[i])
        
        # -- if there are matching candidates.
        if len(candidates) > 0:
            # -- get the first candidate 
            index = candidates[0][0]
        
            # -- add the incoming log to the current cluster 
            self.templates[index].update(words, logid)

            # -- return this cluster 
            return self.templates[index]

        # -- create a new template, if a similar cluster has not been found
        # -- the _append_template function belongs to the TemplateManager superclass, 
        # -- it simply appends a new template to the self._templates list. 
        new_template = self._append_template( LenmaTemplate(len(self.templates), words, logid) )
        
        # -- return the new cluster 
        return new_template

infer_template method iterates through the templates list. Initially, a new cluster is created by instantiating a 
LenmaTemplate object, and appending this to the self._templates list. As the self._templates list gets populated, 
each template in this list is compared to the incoming log message. If the log messages have the same length, 
then the similarity (cosine and token position) between the two logs is evaluated - this is done by calling the 
template.get_similarity_score method. 

An integrity check ensures the similarity is sufficiently high, before appending this template to a candidates list. 
After iterating through each template, the candidates list is sorted, and the incoming log message is appended to the 
cluster with the highest similarity (candidates[0][0]). The self._templates list is then updated. 

The LenmaTemplate.get_similarity_score method performs two checks: firstly, it checks the length similarity between the 
currrent template, and the incoming message - accomplished using self._get_accuracy_score and self._get_similarity_score_cosine. 
Once the cosine score is obtained, the token positions are checked using self._count_same_word_positions (the conditional evaluates 
to case==6 by default, due to hardcoding).  

.. code-block:: python 

    # Author: Shima, K. (2016) 

    def get_similarity_score(self, new_words):
        # heuristic judge: the first word (process name) must be equal
        if self._words[0] != new_words[0]:
            return 0

        # check exact match
        ac_score = self._get_accuracy_score(new_words)
        if  ac_score == 1:
            return 1

        # -- get the cosine similarity 
        cos_score = self._get_similarity_score_cosine(new_words)

        case = 6 # hardcoded value 

        #...
        # -- conditional 'case' statement redacted for simplicity. 
        #...
        elif case == 6:
            if self._count_same_word_positions(new_words) < 3:
                return 0
            return cos_score

The _get_similarity_score_cosine and _count_same_word_positions methods perform the essential checks that 
determine whether this log message belongs to the cluster. The cosine similarity method is shown below:

.. code-block:: python

    # Author: Shima, K. (2016) 

    def _get_similarity_score_cosine(self, new_words):

        # get self._wordlens as a 2D array
        wordlens = np.asarray(self._wordlens).reshape(1, -1)

        #print(self._wordlens, wordlens)
        
        # get the word length vector of the new words
        new_wordlens = np.asarray([len(w) for w in new_words]).reshape(1, -1)

        # get the cosine similarity between the current and new word length vectors. 
        cos_score = cosine_similarity(wordlens, new_wordlens)

        return cos_score

This method obtains the word length vectors of the two log messages, and calls the cosine_similarity 
function from sklearn.metrics.pairwise library. The purpose of this method is to identify log messages
that have the same (or similar) distribution of word lengths as the current template. To ensure sufficient 
similarity, a threshold value of 0.9 is used in the infer_template method. 

However, log messages with the same word length distribution may not be similar in practice - that is, there 
may be instances where logs have high cosine similarity, but the actual tokens have different semantic meanings 
(Shima, 2016).

.. code-block:: python 

    # Author: Shima, K. (2016) 

    def _count_same_word_positions(self, new_words):
        c = 0
        for idx in range(self.nwords):
            if self.words[idx] == new_words[idx]:
                c = c + 1
        return c

Thus, the position of tokens is also checked to ensure the log messages are similar. The _count_same_word_positions, 
outlined above, checks to see if the cluster template and incoming log message have the same token positions. 

If the token position count is sufficiently high (> 3), then the cosine similarity is returned to the infer_template 
method, which in-turn updates the templates list.


| *Results* 

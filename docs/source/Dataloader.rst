July 15th 2022 - Loglizer DataLoader 
===================================================================================
Note: this documentation was written and committed on June 2nd 2022. 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. Each demo script uses a dataloader script to split the 
data into training and test sets. This dataloader script was studied to 
understand this preprocessing stage. 

| *DataLoader overview*

All anomaly detection models use the dataloader file to generate the 
training and test data. One finding is that the dataloader script 
uses custom functions for each type of log - it is not a generic file 
that works with every log dataset. The current repository only supports 
data generation for HDFS logs. Functions need to be written to generate 
data for Linux logs. 

| *Dataloader process (HDFS)* 

The process for DeepLog was studied. The load_HDFS function is the entry 
point to generate Hadoop training and test data. The load_HDFS function 
has the following signature: 

.. code-block:: python

    # Author: He et al., 2016, LogPAI team 
    def load_HDFS(log_file, label_file=None, window='session', 
                  train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0)

    # log_file -> path to the parsed log file 
    # label_file -> path to a label file (for supervised learning)
    # window -> the window options (?)
    # train_ratio -> split between training and test data 
    # split_type -> either uniform or sequential. Uniform splits positive & negative samples
    #               equally when using a label file (??). Sequential splits the data without 
    #               the label file. 

The load_HDFS function houses conditional statements that check the 
file type of the parsed logs, and split the data accordingly. The following pseudocode 
describes the function: 

.. code-block:: python 

    # Authors: LogPAI Team

    def load_HDFS(...):
        if log_file.endswith('.npz'): 
            # generate (x_train, y_train), (x_test, y_test)
        elif log_file.endswith('.csv'):
            # create a dictionary of blk_ID's and Event ID's 
            # convert this dictionary into a pandas dataframe ("data_df")
            
            if label_file: 
                # Load labeled data into a pandas dataframe. Index by blk_ID's 
                # Convert the labels to a dictionary 
                # Add a label column to "data_df". Set to 1 if an anomaly, 0 otherwise
                # generate (x_train, y_train), (x_test, y_test) 

            if save_csv:
                # convert "data_df" into a csv 
            
            if window_size > 0: 
                # generate x_train, window_y_train, y_train using the slice_hdfs function
                # generate x_test, window_y_test, y_test using slice_hdfs function
                # return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

            if label_file is None: 
                # generate (x_train, _), (x_test, _) 
                #   there is no label file supplied, so only the entries exist 
                # return (x_train, None), (x_test, None), data_df 
        else:
            raise NotImplementedError('load_HDFS() only supports csv and npz files!')

        # get values for training/test data:         
            # num_train -> x_train.shape[0]
            # num_test  -> x_test.shape[0]
            # num_total -> num_train + num_test 
            # num_train_pos -> sum(y_train)
            # num_test_pos -> sum(y_test)
            # num_pos -> num_train_pos + num_test_pos 

        # return (x_train, y_train), (x_test, y_test)

As observed, generating the training and test data 
(for HDFS) depends on the blk_ID metric in parsed logs. The data_df dataframe 
holds all essential values required to generate the training and test data. This
dataframe has the following format: 

.. code-block:: shell  
                        BlockId                                      EventSequence  Label
    0  blk_-1608999687919862906  [E5, E22, E5, E5, E11, E11, E9, E9, E11, E9, E...      0
    1   blk_7503483334202473044  [E5, E5, E22, E5, E11, E9, E11, E9, E11, E9, E...      0
    2  blk_-3544583377289625738  [E5, E22, E5, E5, E11, E9, E11, E9, E11, E9, E...      1
    3  blk_-9073992586687739851  [E5, E22, E5, E5, E11, E9, E11, E9, E11, E9, E...      0
    4   blk_7854771516489510256  [E5, E5, E22, E5, E11, E9, E11, E9, E11, E9, E...      0

Where the event ID corresponds to a parsed event that holds that blk_ID. A similar 
method must be identified for linux logs: that is, what do we center our event sequences 
around when creating a data_df dataframe? Unlike HDFS, which has blk_ID, the log content 
in Linux logs are more random. This must be investigated further. 

As observed in the previous code block, the data loader file uses a _split_data 
function to generate the (x_train, y_train), (x_test, y_test) data. This function 
uses the training ratio (0.5 in all ML models, 0.2 for DeepLog) to slice the EventSequences (x data)
and Labels (y data, if provided) into appropriate training/test arrays. These arrays are shuffled 
and then returned. 

A final slice_HDFS function (custom implementation for HDFS logs) is also used to 
generate "windows" of training and test data. The following pseudocode describes the 
slice_hdfs function:

.. code-block:: python 

    # Authors: LogPAI Team
    
    def slice_hdfs(x, y, window_size):

        # initialise empty array for results
        results_data = []

        for idx, sequence in enumerate(x):
            # get the length of the event ID sequences 
            seqlen = len(sequence)
            i = 0 

            while (i + window_size) < seqlen:
                # obtain a 'slice' of event ID's 
                slice = sequence[i: 1 + window_size] 
                
                # update the array
                # the results array has the following format: 
                #   [index number, slice of event IDs, 
                #               the event ID for this slice (label), results at this index]
                results_data.append([idx, slice, sequence[i + window_size], y[idex]])
            else: 
                # once the seqlen has been exceesed, pad this entry in the results data 
                slice = sequence[i: i + window_size]
                slice += ["#Pad"] * (window_size - len(slice))
                results_data.append([idx, slice, "#Pad", y[idx]])
            
        # convert this array into a dataframe 
        results_df = pd.Dataframe(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])

        # return the 'sliced' results 
        return results_df[["SessionID", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]

For DeepLog, this sliced data is then supplied into the preprocessing file to generate 
vectorizer and iterator (pytorch Dataloader) objects. 

| *Dataloader observations*

Some development time will be required to code custom functions for 
Linux logs. HDFS logs use blk_ID, but a common identifier such as that 
does not exist for Linux logs. Therefore, an intermediate challenge is 
finding the best way to generate training + test + window data for Linux. 


    


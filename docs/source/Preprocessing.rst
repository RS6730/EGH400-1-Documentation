July 15th 2022 - Loglizer Preprocessing I (for DeepLog) 
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. Each demo script uses a preprocessing script before feeding 
the data into anomaly detectioin models. This preprocessing script was studied. 

| *Preprocessing I overview* 
The preprocessing.py script contains three classes - Vectorizer, Iterator and 
FeatureExtractor. As per the code, the Vectorizer and Iterator classes are 
used by DeepLog. ML models (InvariantsMiner, PCA, SVM, LR) use the 
FeatureExtractor class. This documentation describes the Vectorizer and 
Iterator classes used by DeepLog. 

| *Preprocessing I - Vectorizer* 
DeepLog uses the Vecotorizer class to transform event ID strings 
into numerical representations ("dimensionality reduction"?). 
A sample is shown below: 

.. code-block:: python
        SessionId                                   EventSequence
    0          0    [E5, E22, E5, E5, E11, E9, E11, E11, E9, E9]

        SessionId                   EventSequence
    0          0    [9, 0, 9, 9, 6, 5, 6, 6, 5, 5]

The Vectorizer class has two functions: fit_transform (used for training data)
and transform (used for test data). The code for this class is shown below: 

.. code-block:: python 

    # Authors: LogPAI Team 

    class Vecotorizer(object): 
        """ Vectorize (numerically represent) all event ID labels """

        def fit_transform(self, x_train, window_y_train, y_train): 
            """
            Fit transform is used to enumerate the event ID labels 
            in window_y_train (convert EID labels into integers)
            """
            # enumerate all labels provided in window_y_train, and 
            # store this enumeration in a dictionary
            self.label_mapping = {eid: idx for idx, eid in enumeratue(window_y_train.unique(), 2)}
            
            # set #Pad to 0, #OOV to 1
            self.label_mapping["#OOV"] = 0 
            self.label_mapping["#Pad"] = 1 
            
            # get the length of the labels dictionary
            self.num_labels = len(self.label_mapping)

            return self.transform(x_train, window_y_train, y_train)

        def transform(self, x, window_y, y): 
            """
            Transform is used to update the x and window_y data. 
            """
            # convert the event sequence into a numerical representation 
            x["EventSequence"] = x["EventSequence"].map(lambda x: [self.label_mapping.get(item, 0) for item in x])

            # convert the labels into numerical representations 
            window_y = window_y.map(lambda x: self.label_mapping.get(x, 0))

            # create a dictionary with updated data 
            data_dict = {"SessionID": x["SessionID"].values, "window_y": window_y.values, 
                         "y": y.values, "x": np.array(x["EventSequence"].tolist())}
            
            return data_dict 

As per ML conventions, the fit_transform function is used by training data, 
while the transform function is used by the test data (as the training data 
is vectorized first, the label_mappings can then be used for the test_data). 

The resulting training and test dataset is then passed into the Iterator 
class. 

| *Preprocessing I - Iterator*
The Iterator class converts the data into a Pytorch dataset object, 
with an internal dataloader object used to iterate through the data.  

.. code-block:: python 

    # Authors: LogPAI Team 

    class Iterator(Dataset):
        """ Create a Pytorch dataset object from the data """

        def __init__(self, data_dict, batch_size=32, shuffle=False, num_workers=1):
            """
            Construct the DataLoader object. 
            """
            self.data_dict = data_dict 
            self.keys = list(data_dict.keys())
            # iterable object. The sessionID, window_y, y and x datasets are represented as 
            # tensors, which will later be used by the deep learning model. 
            self.iter = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        def __get__item(self, index):
            return {k: self.data_dict[k][index] for k in self.keys} 
        
        def __len__(self):
            return self.data_dict["SessionId"].shape[0] 

The Iterator.iter object (the pytorch DataLoader) holds all of the pre-processed data 
as tensors, which will later be used in the deep learning model. 

| *Preprocessing I observations*
There may be a need to write custom iterator and vectorizer 
functions of Linux logs - it depends on how the data is represented 
from dataloader.py. 

This document only looked at Iterator and Vectorizer. A custom FeatureExtractor 
class is also used by ML models, which is explained in Preprocessing II.  
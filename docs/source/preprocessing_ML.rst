July 22nd 2022 - Loglizer Preprocessing II (for ML models) 
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. Each demo script uses a preprocessing script before feeding 
the data into anomaly detectioin models. This preprocessing script was studied. 

| *Preprocessing II overview* 
The preprocessing.py script contains three classes - Vectorizer, Iterator 
and FeatureExtractor. Vecotorizer and Iterator are used for deep learning 
models, and were studied in Preprocessing I. This page studies the 
FeatureExtractor class, which is used by machine learning models, such as 
SVM and PCA. dataloader.py is also studied, as the representation of 
data is different for general machine learning models. 

| *Data Representation - dataloader.py* 

Event sequence data for machine learning does not require session windows.
Thus, the load_HDFS function simply returns numpy arrays of (x_train, y_train), 
(x_test, y_test). x data (input) is represented as a numpy list of event sequences 
for a given HDFS block ID (a nested list). The output data is represented 
as a numpy array of anomaly labels (0 or 1). 

The x_train and x_test data structures are passed into the FeatureExtractor 
class from the preprocessing script. The training data is fitted and transformed, 
while the test data is transformed. 

| *preprocessing.py - FeatureExtractor class*

The FeatureExtractor class has two functions and a constructor. The first function, 
fit_transform, is used for vectorising the training dataset. The first function, 
fit_transform, calculates the inverse-document-frequency (idf) of all event ID's 
that appear in the training dataset. This is multiplied by the frequency of event ID's, 
thereby generating a numerical representation for the event IDs. 

.. code-block:: python 

    def fit_transform(self, X_seq, term_weighting=None, normalization=None, oov=False, min_count=1)
        # X_seq is the training data. For HDFS, it is a matrix of event

        # set the class variables. Term weighting is set to 'tf-idf', 
        # while normalization can either be 'zero-mean' or 'sigmoid'. 
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.oov = oov

        X_counts = [] 

        # iterate through all event ID's 
        for i in range(X_seq.shape[0]): 

            # create a collection of event IDs with frequency 
            # of occurences. Append to the master list. 
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)

        # create a dataframe of event IDs
        X_df = pd.DataFrame(X_counts)
        x_df = X_df.fillna(0)

        # capture all events within the training set 
        self.events = X_df.columns 
        
        # get all frequency values 
        X = X_df.values 

        if self.oov: 
            # oov is false by default. 
            # this code will be revisited later

        num_instance, num_event = X.shape 
        if self.term_weighting == 'tf-idf':
            # calculate the tf-idf scores for each event 
            # sum all true (1) instances within the dataframe 
            df_vec = np.sum(X > 0, axis=0)

            # calculate the idf
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))

            # calculate the idf matrix 
            # np.tile replicates the idf_vec by the number of rows. This 
            # is multiplied by X to generate an idf matrix. 
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))

            # update X to be the idf_matrix
            X = idf_matrix 

        if self.normalization == 'zero-mean':
            # calculate the mean for each column (event ID)
            mean_vec = X.mean(axis=0)

            # reshape this to a vector of (1, 14) (for hdfs logs and the 
            # collected event IDs)
            self.mean_vec = mean_vec.reshape(1, num_event)

            # adjust the weight of each idf score by the mean 
            X = X - np.title(self.mean_vec, (num_instance, 1))

        elif self.normalization == 'sigmoid':
            # convert all non-zero idf values into 
            # a range between 0 and 1 
            X[X != 0] = expit(X[X != 0])

        X_new = X 
        
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1])) 
        return X_new

At its core, the fit_transform function vectorizes event ID's by calculating the idf of each 
event ID within the dataset, and multiplying this by its frequency. Normalization is applied 
by calculating adjusting each event ID by the mean (a sigmoid option is also available). 

The fit_transform function is followed by a transform function, which is used exclusively 
for test data. This function follows the same control flow as fit_transform. However, the 
idf_vector and mean_vec structures are re-used from the fit_transform function (that is, the 
calculations from the training data are re-used for the test data). 


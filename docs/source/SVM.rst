July 23rd 2022 - Support Vector Machines (SVM)
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. This page analyses the Python implementation of using support 
vector machines (SVMs) for anomaly detection. The original paper was written by
Liang et al. (2007), who tested the efficacy of SVMs, Nearest Neighbours and the
RIPPER method for failure prediction on the IBM BlueGene supercomputer. 

| *SVM Overview* 
There is very little overlap between the Python implementation of using SVMs 
and Liang et al.'s 2007 paper. One difference is that Liang et al.'s research 
was conducted on IBM BlueGene/L logs, where the logs were seperated into 
different time periods. Historical observations were used to predict whether 
the current time period contained any fatal/failure logs (2007). In contrast, 
the LogPAI implementation uses HDFS logs, where the data is grouped by Block 
ID. An interesting observation  is that Liang et al's, research pertains to 
'failure prediction', and not 'anomaly detection'. There may be a difference between 
the two, as anomalous logs may not necessarily contain failure/fatal severities. 

Liang et al. reference numerous features that were monitored. These include 
the number of events within a time period, the number of accumulated events 
within the current observation period, and distribution of events (among other 
features) (2007). These features are then normalised, and their significance is 
calculated, before being used as inputs into prediction models (2007). In contrast, 
the LogPAI code vectorizes HDFS data (using an IDF matrix) and feeds this 
into an SVM object (using LinearSVC from sklearn). Thus, the Python code for 
implementing an SVM anomaly detection model is relatively simple, as the sklearn 
library can be leveraged. However, appropriate feature extraction is important 
(this is something that must be investigated for Linux logs). 


| *SVM Implementation* 

SVM's are implemented by leveraging the sklearn LinearSVC class. The entire 
SVM object implementation is described below (custom comments with explanations 
have been inserted). 

.. code-block:: python 

    # Authors: LogPAI Team

    class SVM(object):

        def __init__(self, penalty='l1', tol=0.1, dual=False, class_weight=None, max_iter=100):

            # instantiate a linear SVC classifier. This uses a linear kernel function to 
            # generate the support vectors. A lasso regularization method (l1) is used 
            # as a penalty during optimisation. 
            self.classifier = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual, 
                                            class_weight=class_weight, max_iter=max_iter)

        def fit(self, X, y):
            # train the model, based on the given training data 
            self.classifier.fit(X, y)

        def predict(self, X):
            # predict outputs for a given input X 
            y_pred = self.classifier.predict(X)

        def evaluate(self, X, y_true):
            # calculate the F1-measure scores for a dataset. 
            y_pred = self.prediction(X)

            precision, recall, f1 = metrics(y_pred, y_true)
            print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
            return precision, recall, f1

| *Observations and Further Tests*
Model hyper-parameters, such as the regularization method, tolerance and number 
of iterations must be fine-tuned to investigate their effects on prediction results. 
A quick test on regularization was performed by changing the penalty from l1 (lasso)
to l2 (ridge). This resulted in a drop in F1 score from 0.602 to 0.373 (recall drop 
from 0.433 to 0.229). This may be caused due to overfitting introduced by using a 
ridge regularization method, where features are not reduced to zero. The remaining parameters 
must be tested further. 





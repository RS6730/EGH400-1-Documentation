July 15th 2022 - Loglizer DataLoader 
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. Each demo script uses a preprocessing script before feeding 
the data into anomaly detectioin models. This preprocessing script was studied. 

| *Preprocessing overview* 
The preprocessing.py script contains three classes - Vectorizer, Iterator and 
FeatureExtractor. As per the code, the Vectorizer and Iterator classes are 
used by DeepLog. ML models (InvariantsMiner, PCA, SVM, LR) use the 
FeatureExtractor class. 

| *Preprocessing - Vectorizer* 

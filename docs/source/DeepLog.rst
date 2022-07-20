July 20th 2022 - DeepLog
=================================================================================== 

| *Research Goal* 
The loglizer (anomaly detection) repository contains demos of various anomaly 
detection models. This page analyses the Python implementation of DeepLog, an 
anomaly detection model authored by Du et al., 2017. DeepLog uses bi-directional 
LSTM layers to identify anomalies. 

| *Note about DeepLog* 
It appears that multiple DeepLog implementations exist online. This documentation 
studies the LogPAI implementation within the loglizer repository. Future docs 
may study other implementations to identify any differences.  

| *DeepLog overview* 
DeepLog uses an LSTM RNN model to predict anomalies from a sequence of "log keys", 
or event templates, and parameter values. Du et al. describes two LSTM networks in 
their original paper: "Execution Path" anomaly detection and "Parameter Value" 
performance anomaly detection (2017). The LogPAI implementation appears to focus 
solely on Execution Path anomalies, given that the parsed HDFS results do not 
contain parameter arrays. This is something that must be investigated. 

A DeepLog class, which inherits from nn.Module, is used for the model. The 
pseudocode is provided below. Each function is explored throughout this document. 

.. code-block:: python

    # Authors: LogPAI Team 

    class DeepLog(nn.Module): 

        def __init__(...):
            # construct the model layers 

        def forward(...): 
            # function for LSTM computation in the forward pass 

        def set_device(...): 
            # pytorch calibration for setting the 
            # device.  

        def init_hidden(...):
            # initialise the hidden layers 
            # as zero tensors. 
        
        def fit(...): 
            # run the training loop:
            #   - iterate through epochs 
            #   - iterate through batches 
            #   - forward pass, back-propogation, optimization

        def evaluate(...): 
            # run model on test data
            # gather f1, recall, precision 

| *DeepLog constructor*
The constructor for DeepLog sets class variables, and also instantiates 
the layers used by the model. This is observed below: 

.. code-block:: python

    # Authors: LogPAI Team 

    class DeepLog(nn.Module): 
        def __init__(self, num_labels, hidden_size=100, num_directionss=2, topk=9, device="cpu"):
            super(DeepLog, self).__init__() 
            # calibrate hidden layers, directions, device 
            self.hidden_size = hidden_size 
            self.num_directions = num_directions 
            self.device = self.set_device(device) 
            # unsure what topk is 
            self.topk = topk 
            
            # calibrate model layers and loss function  
            self.rnn = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
            self.criterion = nn.CrossEntropyLoss() 
            self.prediction_layer = nn.Linear(self.hidden_size * self.num_directions, num_labels + 1)

The critical variables include "rnn", "criterion" and "predicton_layer". 

* self.rnn 
The LSTM model is instantiated by using torch.nn.LSTM. The input size is 
set to one tensor, while the hidden_size is set to 32 (set in the demo script). 
By setting batch_first as True, the input and output tensors are provided as 
(batch, seq, feature), as per the Pytorch documentation on LSTM (2022) - the 
significance of this configuration must be explored further. Finally, the LSTM 
is set as bidirectional, allowing two-way forward-pass and back-propogation during
training. However, Du et al's original 2017 paper does not explicitly state that 
DeepLog is bidirectional - this parameter may have been introduced by the LogPAI team. 

* self.criterion 
The cross-entropy-loss function is used by DeepLog (Du et al., 2017). This loss 
function calculates the difference between probability distributions, and 
minimises the error. This is instantiated by nn.CrossEntropyLoss() 

* self.prediction_layer 
A Linear layer is used as the prediction layer, instantiated by nn.Linear. 

| *DeepLog fit* 

After model instantiation (where the layers and loss functions are created), 
the model.fit function is called with the training_loader, created by the 
preprocessing.Iterator class. The original LogPAI demo script uses 2 epochs 
for training. The code for this function is shown below: 

.. code-block:: python

    # Authors: LogPAI Team

    def fit(self, training_loader, epoches=10): 
        # set the device 
        self.to(self.device)

        # set the model to training mode 
        model = self.train() 

        # instantiate an Adam optimiser. 
        optimizer = optim.Adam(model.parameters())

        # conduct the training process 
        for epoch in range(epoches): 
            batch_cnt = 0 
            epoch_loss = 0 

            # iterate through batches 
            for batch_input in train_loader: 
                # retrieve the Cross Entropy Loss from the 
                # forward pass 
                loss = model.forward(batch_input)["loss"]
                
                # backpropogation, calculate the gradients for 
                # each weight. 
                loss.backward()

                # update model weights wrt to the 
                # calculated gradients 
                optimizer.step() 

                # empty the current gradients 
                optimizer.zero_grad() 
            
                # get the loss for this epoch 
                epoch_loss += loss.item() 
                batch_cnt += 1 


            epoch_loss = epoch_loss/batch_cnt 
            print("Epoch {}/{}, training loss: {:.5f}".format(epoch+1, epoches, epoch_loss))

The training process follows the standard sequence of function calls 
to run a Pytorch model. That is: forward pass, backward pass (backpropogation), 
optmiser step and emptying gradients. 

| *DeepLog forward*

The forward function acts as a wrapper around the actual 
forward pass process. 

.. code-block:: python

    # Authors: LogPAI Team

    def fit(self, input_dict): 
        # get output labels, used to calculate loss later
        y = input_dict["window_y"].long().view(-1).to(self.device)
        
        self.batch_size = y.size()[0]
        
        # get log key sequences. 
        # Inputs have a shape of [32, 10, 1], where 32 is the batch size, 
        # each batch has 10 log keys, and each log key has a dimension 
        # of 1. 
        x = input_dict["x"].view(self.batch_size, -1, 1).to(self.device)

        # forward pass into LSTM layer, along with the hidden 
        # and cell states. The outputs shape is [32, 10, 64]. 
        # Each log key in the input tensor outputs a 
        # resulting hidden state. As the LSTM network is bi-directional, 
        # the final tensor has a size of 64 (32 x 2). 
        outputs, hidden = self.rnn(x.float(), self.init_hidden()) 
        
        # pass the final hidden state in the output (-1) 
        # to the fully connected layer. [:, -1, :] has the 
        # same shape as [:, 10, :]. Thus, outputs for the final log 
        # key sequence in each batch are passed into the fully 
        # connected layer (use the final log key sequence to predict 
        # the next event). The prediction layer is a linear transformation, 
        # with a shape of (32, 13). 
        logits = self.prediction_layer(outputs[:, -1,:])

        # apply softmax to get output between 0 and 1.       
        y_pred = logits.softmax(dim-1)

        # calculate the cross entrtopy loss  between prediction and outputs. 
        # !! Need to investigate this further. The CrossEntropyLoss function
        # in pytorch combines LogSoftMax and NLLLoss, so the outputs in 
        # logits are flattened to between 0-1. However, the math for this 
        # function needs to be clarified. 
        loss = self.criterion(logits, y)

        # return the loss, and the predicted y. 
        return_dict = {'loss': loss, 'y_pred': y_pred}
        return return_dict 

As observed, this function is where the forward pass, fully connected 
layer and loss calculation take place. The forward function begins 
by obtaining the output labels for this window (32 labels, based on 
the batch size). It then obtains the inputs for the LSTM model, which 
are organised in tensors of shape [32, 10, 1], based on the batch size 
and window size (32 and 10, respectively). The dimension of each input 
is also 1 (one log key sequence). Hidden and cell states are generated 
from the init_hidden function, and the rnn function is then called to 
run the LSTM model. 

The LSTM model outputs a tensor of shape [32, 10, 64]. That is, each of the 
10 log keys in a batch of 32 log sequences outputs the hidden state result, 
with a length of 64 (as this is a bi-directional LSTM). The result for the 
final log key in this tensor, [:, 10, 64], is then passed into the fully 
connected layer, which applies a linear function to get a final output 
result. The shape of this is (32, 13), where 32 is the batch size and 
13 is the number of labels (including 10 log sequences, 'OOV' and 'PAD', plus 
an extra label). y_pred will output a range of probabilities from 0 to 12, where
each index is mapped back to the vectorized representation of the event sequence
(done in preprocessing.py). 

These outputs are then passed into the CrossEntropyLoss function to calculate 
the loss. The CrossEntropyLoss performs LogSoftMax and NLLLoss - these functions, 
and the maths behind cross entrtopy, must be investigated. On a high level, 
the CrossEntropyLoss looks at the given and target probability distributions, and 
calculates the resulting difference. 

The resulting dictionary is then used by the fit function. Specifically, the loss values 
are used to calculated the gradients. These gradients are then used in optimizer.step() 
to update the model weights. 

The fit function (and the forward function within) are used to train the model. 
Following this, the model is evaluated with training and test data. 

| *DeepLog evaluate*

The evaluate function calls forward on pytorch dataloader object that 
holds the test data. The code is provided below: 

.. code-block:: python

    # Authors: LogPAI Team
    def evaluate(self, test_loader): 
        y_pred = []
        store_dict = defaultdict(list)

        # iterate through the dataloader in batches 
        for batch_input in test_loader: 

            # call the model and get the loss, y_pred
            return_dict = self.forward(batch_input)
            # the predicted y values which have gone through softmax 
            y_pred = return_dict["y_pred"]

            # construct a dictionary with the sessionID, anomaly labels and predicted event sequences 
            store_dict["SessionId"].extend(batch_input["SessionId"].data.cpu().numpy().reshape(-1))
            # anomaly labels (0 or 1)
            store_dict["y"].extend(batch_input["y"].data.cpu().numpy().reshape(-1))
            # the predicted log_key values 
            store_dict["window_y"].extend(batch_input["window_y"].data.cpu().numpy().reshape(-1))
            
            # get the probabilities and predicted values from 
            # torch.max (unpack the result from torch.max)
            # Each event sequence is vectorized from 0 to 12. 
            # thus, the indices of each probability indicate what event 
            # sequence (vectorized to between 0-12) is likely to be next. 
            window_prob, window_pred = torch.max(y_pred, 1)

            # add predictions and probabilities to the dictionary 
            store_dict["window_pred"].extend(window_pred.data.cpu().numpy().reshape(-1))
            store_dict["window_prob"].extend(window_prob.data.cpu().numpy().reshape(-1))
            
            # get the top 5 predicted values (topk set as a hyperparameter)
            top_indice = torch.topk(y_pred, self.topk)[1] # b x topk

            # add the top five predicted values to the dictionary 
            store_dict["topk_indice"].extend(top_indice.data.cpu().numpy())

        # get the predicted + actual log keys
        window_pred = store_dict["window_pred"]
        window_y = store_dict["window_y"]

        # store the dictionary as a dataframe 
        store_df = pd.DataFrame(store_dict)

        # create a new column for PREDICTED anomalies. The predicted anomaly 
        # is set to 1 if the log key is not in the top 5 keys (that is, the model 
        # the value is not expected for this sequence of logs). Set this to 0 otherwise. 
        store_df["anomaly"] = store_df.apply(lambda x: x["window_y"] not in x["topk_indice"], axis=1).astype(int)

        ## simplify the dataframe
        store_df.drop(["window_pred", "window_y"], axis=1)
    
        ## group by the session ID 
        store_df = store_df.groupby('SessionId', as_index=False).sum()
                
        ## convert all non-zero anomalies to integers (for the predicted and true values)
        store_df["anomaly"] = (store_df["anomaly"] > 0).astype(int)
        store_df["y"] = (store_df["y"] > 0).astype(int)

        ## get the predicted and true values 
        y_pred = store_df["anomaly"]
        y_true = store_df["y"]

        
        ## calculate the results using sklearn 
        metrics = {"window_acc" : accuracy_score(window_y, window_pred),
        "session_acc" : accuracy_score(y_true, y_pred),
        "f1" : f1_score(y_true, y_pred),
        "recall" : recall_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred)}
     
        # print and return 
        print([(k, round(v, 5))for k,v in metrics.items()])
        return metrics

This function generates anomaly predictions by checking the 
current y value against the top 5 prediced values. As per Du et al., 
keys within the top 5 values are not classes as anomalies (2017). This 
is then compared to the ground truth anomalies, and sklearn is used 
to calculate accuracy metrics. 

| *DeepLog further functions*

DeepLog.py has functions for setting the device (part of calibrating 
Pytorch) and generating the hidden and cell states. 

| *DeepLog observations*

This LogPAI implementation only looks at anomalies from event sequences. 
Parameter anomalies are not considered (parameters are not even collected 
in the HDFS dataset). Futhermore, Du et al. mention functionality, such as 
user feedback and re-training if false positives are identified - these 
are not included in the LogPAI implementation. 

| *Resources and Documentation* 
Pytorch LSTM's, https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
Pytorch CrossEntropyLoss, https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
Pytorch Linear, https://pytorch.org/docs/stable/generated/torch.nn.Linear.html


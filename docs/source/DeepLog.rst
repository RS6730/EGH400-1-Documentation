July __ 2022 - DeepLog
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
            for brach_input in train_loader: 
                # retrieve the Cross Entropy Loss from the 
                # forward pass 
                loss = model.forward(batch_input)["loss"]
                
                # backpropogation, calculate gradients 
                loss.backward()

                # update model parameters and empty gradients 
                optimizer.step() 
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
        
        # get log key sequences. Format into a verticle array 
        x = input_dict["x"].view(self.batch_size, -1, 1).to(self.device)

        # forward pass into LSTM layer 
        # output has the shape (32, 10, 64)
        outputs, hidden = self.rnn(x.float(), self.init_hidden()) 
        
        # forward last output into prediction layer (?)
        # (apply a linear transformation)
        logits = self.prediction_layer(outputs[:, -1,:])

        # apply softmax to get output between 0 and 1          
        y_pred = logits.softmax(dim-1)

        # calculate the cross entrtopy loss  between prediction and outputs 
        loss = self.criterion(logits, y)

        # return 
        return_dict = {'loss': loss, 'y_pred': y_pred}
        return return_dict 

There are several aspects of this function that need to be investigated
further. These include: 














| *Resources and Documentation* 
Pytorch LSTM's, https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

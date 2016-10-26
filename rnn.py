import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import csv
import random
from random import shuffle

class Config(object):
      """Holds model hyperparams and data information.
      The config class is used to store various hyperparameters and dataset
      information parameters. Model objects are passed a Config() object at
      instantiation.
      """
      batch_size = 50
      batches_per_epoch = 10
      step_size= 66 # number of slices in each genom
      input_dim=661 # input_dimension, this is the size of each genom slice
      hidden_dim = 128 # number of nerons per hidden layer
      label_dim = 5 # we have a total of classes (like or not like)
      max_epochs = 24
      early_stopping = 3
      dropout = 1.0
      learning_rate = 0.0001
      forget_bias = 1.0
      model = 'RNN' #'BiRNN'
      cell_type = 'GRU'
      stack = 1
      use_peepholes = False
      cell_clip = 1.0
      train_file = "GO_Quad_Omni_v3_v1_chr2_QC.ped"
      label_file = "factor_scores.csv"
      

class Models(object):
    '''
    Read the data and label file.
    input: file name
    output:
        it outputs a 3 hyper-dimensional structrue as data and a 2 hyper-dimensional structrue as label:
        data : [number of person, [number of slice per genom x dimension of each slice]]
        label: [number of perons, [a list of 5 lables]]
    '''
    def read_file(self, data_file, label_file):
        embedding = []
        labels = []
        #for chrome 2
        input_dim=661
        raw_label={}
        with open(label_file) as csvfile:
            first_row=True
            for row in csv.reader(csvfile.read().splitlines()):
                if first_row:
                    first_row=False
                    continue
                raw_label[row[0]]=[float(i) for i in row[1:]]
                
        for line in open(data_file):
            data = line.strip().split(' ')
            #keep sex and label id
            sex = data[4]
            label_id = data[1]
            #only get meaningful data
            if label_id not in raw_label.keys():
                continue
            if all(v==0 for v in raw_label[label_id]):
                continue
            labels.append(raw_label[label_id])
            for i in range(6):
                del data[0]
             
            for i in range(self.config.step_size):
                data_slice=data[self.config.input_dim*i:self.config.input_dim*(i+1)]
                data_slice.append(sex)
                embedding.append(data_slice)
        self.training_data = np.asarray(embedding).reshape(len(labels), self.config.step_size, self.config.input_dim+1)
        self.training_label = np.asarray(labels)



    def print_model_params(self):
        print 'Model: ' + self.config.model
        print 'Cell type:' + self.config.cell_type
        print 'Learning rate: ' + str(self.config.learning_rate)
        print 'Hidden Units: ' + str(self.config.hidden_dim)
        print 'Dropout: ' + str(self.config.dropout)
        print 'Forget Bias: ' + str(self.config.forget_bias)
        print 'Peehole: ' + str(self.config.use_peepholes)
        print 'Stack: ' + str(self.config.stack)

    '''
    initialize model parameters, note LSTM and BiRNN requires twice the hidden dimenssion due their design
    '''
    def init_variables(self):
        if self.config.model == 'BiRNN' and self.config.cell_type=='LSTM':
            weight_size=2*self.config.hidden_dim
        else:
            weight_size=self.config.hidden_dim

        # Define weights and bias
        with tf.variable_scope(str('test')):
            self.weights = {
                'hidden': tf.Variable(tf.random_normal([self.config.input_dim+1, weight_size])), # Hidden layer weights
                'out1': tf.Variable(tf.random_normal([self.config.hidden_dim, self.config.label_dim])),
                'out2': tf.Variable(tf.random_normal([self.config.hidden_dim, self.config.label_dim])),
                'out3': tf.Variable(tf.random_normal([self.config.hidden_dim, self.config.label_dim])),
                'out4': tf.Variable(tf.random_normal([self.config.hidden_dim, self.config.label_dim])),
                'out5': tf.Variable(tf.random_normal([self.config.hidden_dim, self.config.label_dim]))
            }
            self.biases = {
                'hidden': tf.Variable(tf.random_normal([weight_size])),
                'out1': tf.Variable(tf.random_normal([self.config.label_dim])),
                'out2': tf.Variable(tf.random_normal([self.config.label_dim])),
                'out4': tf.Variable(tf.random_normal([self.config.label_dim])),
                'out5': tf.Variable(tf.random_normal([self.config.label_dim])),
                'out3': tf.Variable(tf.random_normal([self.config.label_dim]))
            }

    '''
    bidirection rnn model
    Note: bidirectional model is most useful when tacking RNNs, in single stack case it just averaging two outputs
    input: information needed to construct a model. F_bias is only relevant when cell type is LSTM
    output:
        linear combination of the rnn results and output weights
    '''
    def BiRNN(self, scope):
        # input shape: (batch_size, step_size, input_dim)
        # we need to permute step_size and batch_size(change the position of step and batch size)
        data = tf.transpose(self.input_data, [1, 0, 2])

        # Reshape to prepare input to hidden activation
        # (step_size*batch_size, n_input), flattens the batch and step
        #after the above transformation, data is now (step_size*batch_size, input_dim)
        data = tf.reshape(data, [-1, self.config.input_dim+1])
        
        # Define lstm cells with tensorflow
        with tf.variable_scope(str(scope)):
            # Linear activation
            data = tf.matmul(data, self.weights['hidden']) + self.biases['hidden']
            data = tf.nn.dropout(data, self.config.dropout)
            # Define a cell 
            if self.config.cell_type == 'GRU':
                lstm_fw_cell = rnn_cell.GRUCell(self.config.hidden_dim)
                lstm_bw_cell = rnn_cell.GRUCell(self.config.hidden_dim)
            else:
                lstm_fw_cell = rnn_cell.LSTMCell(self.config.hidden_dim, forget_bias=self.config.forget_bias,
                                                 use_peepholes=self.config.use_peepholes, cell_clip=self.config.cell_clip)
                lstm_bw_cell = rnn_cell.LSTMCell(self.config.hidden_dim, forget_bias=self.config.forget_bias,
                                                 use_peepholes=self.config.use_peepholes, cell_clip=self.config.cell_clip)
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            data = tf.split(0, self.config.step_size, data) # step_size * (batch_size, hidden_dim)
            # Get lstm cell output
            print 'running single stack Bi-directional RNN.......'
            outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, data,
                                                        initial_state_fw=self.init_state_fw,
                                                        initial_state_bw=self.init_state_bw, scope="RNN1")
            # for basic rnn prediction we really just interested in the last state's output, we need to average them in this case
            total_outputs=tf.div(tf.add_n([outputs[2], outputs[1]]), 2.0)
            return [tf.nn.dropout(tf.matmul(total_outputs, self.weights['out1']) + self.biases['out1'], self.config.dropout),
                    tf.nn.dropout(tf.matmul(total_outputs, self.weights['out2']) + self.biases['out2'], self.config.dropout),
                    tf.nn.dropout(tf.matmul(total_outputs, self.weights['out3']) + self.biases['out3'], self.config.dropout),
                    tf.nn.dropout(tf.matmul(total_outputs, self.weights['out4']) + self.biases['out4'], self.config.dropout),
                    tf.nn.dropout(tf.matmul(total_outputs, self.weights['out5']) + self.biases['out5'], self.config.dropout),]
    '''
    standard rnn model
    input: information needed to construct a model. F_bias is only relevant when cell type is LSTM
    output:
        linear combination of the rnn results and output weights
    '''
    def RNN(self, scope):
        # input shape: (batch_size, step_size, input_dim)
        # we need to permute step_size and batch_size(change the position of step and batch size)
        data = tf.transpose(self.input_data, [1, 0, 2])
        # Reshape to prepare input to hidden activation
        # (step_size*batch_size, n_input), flattens the batch and step
        #after the above transformation, data is now (step_size*batch_size, input_dim)
        data = tf.reshape(data, [-1, self.config.input_dim+1])

        with tf.variable_scope(str(scope)):
            data = tf.nn.dropout(tf.matmul(data, self.weights['hidden']) + self.biases['hidden'], self.config.dropout)

            # Define a lstm cell with tensorflow
            if self.config.cell_type == 'GRU':
                lstm_cell = rnn_cell.GRUCell(self.config.hidden_dim)
            else:
                lstm_cell = rnn_cell.LSTMCell(self.config.hidden_dim, forget_bias=self.config.forget_bias)
                
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            data = tf.split(0, self.config.step_size, data) # step_size * (batch_size, hidden_dim)

            # Get lstm cell output
            outputs, states = rnn.rnn(lstm_cell, data, initial_state=self.init_state)

            # we really just interested in the last state's output
            return [tf.matmul(outputs[-1], self.weights['out1']) + self.biases['out1'],
                    tf.matmul(outputs[-1], self.weights['out2']) + self.biases['out2'], 
                    tf.matmul(outputs[-1], self.weights['out3']) + self.biases['out3'],
                    tf.matmul(outputs[-1], self.weights['out4']) + self.biases['out4'], 
                    tf.matmul(outputs[-1], self.weights['out5']) + self.biases['out5']]


    '''
    feeding information to the input placeholders
    this function is call as the init process, data are feed in by tensor flow graph
    '''
    def add_placeholders(self):
        self.cell_size = self.config.hidden_dim
        if self.config.cell_type == 'LSTM':
            self.cell_size = 2 * self.config.hidden_dim
        # define graph input place holders
        self.input_data = tf.placeholder("float", [None, self.config.step_size, self.config.input_dim+1])
        self.input_label = tf.placeholder("float", [None, self.config.label_dim])
        # Tensorflow LSTM cell requires 2x hidden_dim length (state & cell)
        self.init_state = tf.placeholder("float", [None, self.cell_size])
        self.init_state_fw = tf.placeholder("float", [None, self.cell_size])
        self.init_state_bw = tf.placeholder("float", [None, self.cell_size])

    def get_feed_dict(self, data, label):
        if (self.config.model == 'BiRNN'):
            feed_dict = {self.input_data: data,
                         self.input_label: label,
                         self.init_state_fw: np.zeros((self.config.batch_size, self.cell_size)),
                         self.init_state_bw: np.zeros((self.config.batch_size, self.cell_size))}
        else:
            feed_dict = {self.input_data: data,
                         self.input_label: label,
                         self.init_state: np.zeros((self.config.batch_size, self.cell_size))}
        return feed_dict

    '''
    this is the core function that launches the model, it initializes the weights and call the model specified in the config
    after model execution it records the test and training loss.
    
    TODO:
    This function should saves the training weight (best set)
    and apply it at test time

    input: model, training data, label, test data/label, and all other paramters needed to run the model
    output:
        the best learning rate found through cross vaildation.
    '''
    def run_model(self, scope=None, debug=False):
        self.print_model_params()
        #making predictions, this actives the rnn model
        if (self.config.model =="BiRNN"): pred = self.BiRNN(scope)
        elif (self.config.model=="RNN"): pred = self.RNN(scope)
        
        # Define loss and optimizer
        label1, label2, label3, label4, label5 = tf.split(1, 5, self.input_label)
        cost = tf.reduce_mean(sum([tf.square(tf.sub(pred[0], label1)),tf.square(tf.sub(pred[1], label2)),tf.square(tf.sub(pred[2], label3)),
                    tf.square(tf.sub(pred[3], label4)),tf.square(tf.sub(pred[4], label5))]))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(cost) # Adam Optimizer
        
        # Initializing the variables
        init = tf.initialize_all_variables()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            best_val_epoch = float('inf')
           #-------------------------training starts here-------------------------------------------
            for epoch in xrange(self.config.max_epochs):
                train_accuarcy = []
                test_accuracy = []
                train_loss = []
                counter = 0
                # Training
                total_traing_data = self.training_label.shape[0]
                while counter  < self.config.batches_per_epoch:
                        input_training_data=self.training_data[np.random.randint(total_traing_data, size=self.config.batch_size), :]
                        input_training_label=self.training_label[np.random.randint(total_traing_data, size=self.config.batch_size), :]
                        feed_dict = self.get_feed_dict(input_training_data, input_training_label)
                        sess.run(optimizer, feed_dict)
                        loss = sess.run(cost, feed_dict)
                        train_loss.append(loss)
                        counter += 1
                epoch_loss=sum(train_loss)/counter
                print "Epoch " + str(epoch) + ", Loss= " + "{:.6f}".format(epoch_loss)
                if best_val_epoch>epoch_loss: best_val_epoch=epoch_loss
                
                '''
                TODO:
                  We need to split the data to validation, test and train, and measure the vaildation loss here
                '''
            print "Optimization Finished!"
            return best_val_epoch
      
    def __init__(self, config):
        self.config = config
        self.read_file(self.config.train_file, self.config.label_file)
        self.add_placeholders()
        self.init_variables()

'''
this function loops through learning rate from [0.5, 0.1, 0.05, 0.01, 0.005,0.001,0.0005,0.0001]
output:
    the best learning rate found 
'''
def find_best_lr(config=None):
    if config is None:
         config = Config()
    learning_rate = [1e-3, 5e-3, 1e-4, 5e-5]
    best_accuarcy = 0
    best_lr = 0
    # iterate through the learning rate and keeping the best learning rate
    for i in learning_rate:
        print 'trying learning rate', i
        config.learning_rate = i
        model = Models(config)
        loss_val = model.run_model(scope='learning_rate' + str(i))
        if loss_val > best_accuarcy:
            best_lr = i
            best_accuarcy = loss_val
    print best_lr
    print "\n"
    return best_lr

'''
this function runs a bidirectional rnn with GRU cell
'''
def run_biDirection_gru():
    config = Config()
    config.learning_rate = 0.005
    config.hidden_dim = 128
    config.dropout = 1.0
    config.model='BiRNN'
    model = Models(config)
    train_acc= model.run_model(scope='run')
    print train_acc

'''
this function runs a uni-directional rnn with LSTM cell
'''
def run_lstm():
    config = Config()
    config.learning_rate = 0.005
    config.hidden_dim = 256
    config.dropout = 1.0
    config.forget_bias = 0.5
    config.use_peepholes = False
    config.cell_type = 'LSTM'
    model = Models(config)
    train_acc= model.run_model(scope='run')
    print train_acc

if __name__ == "__main__":
    random.seed(31415)
    find_best_lr()
    #run_lstm()
    #run_biDirection_gru()

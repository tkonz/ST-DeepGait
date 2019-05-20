import numpy as np
import tensorflow as tf
#from process_data import get_data
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf
from collections import Counter
import pickle
import copy
import json

# import data_utils

VAL_PCT = 0.1
TEST_PCT = 0.1
SUMMARY_DIR = "summary_SRNN_model_1"
FROM_CHECKPOINT = False

np.random.RandomState(seed=123456)
training_mode = tf.placeholder(tf.bool, ())


class nodeFC(object):
    def __init__(self,
                 num_hidden,
                 input_tensors):
        self.num_hidden = num_hidden
        self.input_tensors = input_tensors
        # [time_steps, batch_size, feature_dim]
        self.Dense1 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense1 = [self.Dense1(i) for i in self.input_tensors]
        self.Dense2 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense2 = [self.Dense2(i) for i in self.dense1]
        print("nodeFC self.dense2 = ", self.dense2)


class edgeRNN(object):
    def __init__(self,
                 cell,
                 num_hidden,
                 input_tensors):
        self.cell = cell
        self.num_hidden = num_hidden
        self.input_tensors = input_tensors
        #self.inputs = [input_tensor for i in range(self.num_of_inputs)]
        self.Dense1 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense1 = [self.Dense1(i) for i in self.input_tensors]
        self.Dense2 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense2 = [self.Dense2(i) for i in self.dense1]
        self.outputs = [tf.nn.dynamic_rnn(self.cell, i, dtype=tf.float32, time_major=True)[0]
                        for i in self.dense2]
        print("edge self.outputs = ", self.outputs)


class nodeRNN(object):
    def __init__(self, cell, num_hidden, input_tensors):
        self.input_tensors = input_tensors
        self.cell = cell
        self.num_hidden = num_hidden
        self.outputs = [tf.nn.dynamic_rnn(self.cell, i, dtype=tf.float32, time_major=True)[0]
                        for i in self.input_tensors]

        self.Dense1 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense1 = [self.Dense1(i) for i in self.outputs]

        self.Dense2 = tf.layers.Dense(self.num_hidden, activation=tf.nn.relu)
        self.dense2 = [self.Dense2(i) for i in self.dense1]
        self.outputs = tf.concat(self.dense2, axis=2)


class StRNNModel(object):
    '''
    Structural RNN Model
    edgeRNNs and nodeRNNs
        dicts : keys = RNN name
                value =  list of layers
    nodeFCs
        dict : keys = node name
    en_connections
        dict : keys = nodeRNNs name
               value = list = edgeRNN dependancies for nodeRNN
        for example: Spine: temp_edge_spine, sum_legs, sum_arms,
    learning_rate
    summaries_dir path to log tensorboard

    '''


class SRNN_model(object):
    def __init__(self,
                 num_hidden):
        self.num_hidden = num_hidden
        self.edgeRNNs = {}
        self.nodeRNNs = {}
        self.nodeFCs = {}
        self.en_connections = {}
        self.ARM_INPUT_SIZE = 12
        self.SPINE_INPUT_SIZE = 9
        self.LEG_INPUT_SIZE = 9

        self.edgeList = ["temp_edge_spine",
                         "temp_edge_arms",
                         "temp_edge_legs",
                         "sum_arms",
                         "sum_legs",
                         "sp_edge_arms",
                         "sp_edge_legs"]

        self.nodeList = ["spine",
                         "arms",
                         "legs"]


    def __call__(self, data, reuse=False):
        print("Calling with reuse: %s" % reuse)
        self.reuse = reuse
        # Slicey stuff here
        self.spine_input = data[:, :, 0:9]
        self.arm_r_input = data[:, :, 9:21]
        self.arm_l_input = data[:, :, 21:33]
        self.leg_r_input = data[:, :, 33:42]
        self.leg_l_input = data[:, :, 42:51]
        self.sum_arms_input = data[:, :, 51:63]
        self.sum_legs_input = data[:, :, 63:72]
        self.sp_arm_r_input = data[:, :, 72:84]
        self.sp_arm_l_input = data[:, :, 84:96]
        self.sp_leg_r_input = data[:, :, 96:105]
        self.sp_leg_l_input = data[:, :, 105:114]
        self.temp_edge_spine_input = data[:, :, 114:123]
        self.temp_edge_arm_r_input = data[:, :, 123:135]
        self.temp_edge_arm_l_input = data[:, :, 135:147]
        self.temp_edge_leg_r_input = data[:, :, 147:156]
        self.temp_edge_leg_l_input = data[:, :, 156:165]

        ### Temporal Edges
        with tf.variable_scope("temp_edge_spine", reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["temp_edge_spine"] = edgeRNN(cell, self.num_hidden, [self.temp_edge_spine_input])
        with tf.variable_scope("temp_edge_arms", reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["temp_edge_arms"] = edgeRNN(cell,self.num_hidden,[self.temp_edge_arm_r_input,
                                                                            self.temp_edge_arm_l_input])
        with tf.variable_scope("temp_edge_legs", reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["temp_edge_legs"] = edgeRNN(cell,self.num_hidden,[self.temp_edge_leg_r_input,
                                                                            self.temp_edge_leg_l_input])
        ### Spatial Edges
        with tf.variable_scope("sum_arms",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["sum_arms"] = edgeRNN(cell,self.num_hidden,[self.sum_arms_input])
        with tf.variable_scope("sum_legs",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["sum_legs"] = edgeRNN(cell,self.num_hidden,[self.sum_legs_input])
        with tf.variable_scope("sp_edge_arms",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["sp_edge_arms"] = edgeRNN(cell,self.num_hidden,[self.sp_arm_r_input,
                                                                          self.sp_arm_l_input])
        with tf.variable_scope("sp_edge_legs",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.edgeRNNs["sp_edge_legs"] = edgeRNN(cell,self.num_hidden,[self.sp_leg_r_input,
                                                                          self.sp_leg_l_input])
        ### NodeFCs
        with tf.variable_scope("spine",reuse=self.reuse):
            self.nodeFCs["spine"] = nodeFC(self.num_hidden,[self.spine_input])
        with tf.variable_scope("arms",reuse=self.reuse):
            self.nodeFCs["arms"] = nodeFC(self.num_hidden,[self.arm_r_input,
                                                           self.arm_l_input])
        with tf.variable_scope("legs",reuse=self.reuse):
            self.nodeFCs["legs"] = nodeFC(self.num_hidden,[self.leg_r_input,
                                                           self.leg_l_input])        \

        # Get node-edge dependencies and create FC Layer for node features
        for node in self.nodeList:
            if "arms" in node:
                edge_deps = ["sp_edge_arms", "sum_arms", "temp_edge_arms"]
            elif "legs" in node:
                edge_deps = ["sp_edge_legs", "sum_legs", "temp_edge_legs"]
            else:
                edge_deps = ["sum_arms", "sum_legs", "temp_edge_spine"]
            self.en_connections[node] = edge_deps
        self.nodeRNN_input = {}
        #node_types = self.nodeRNNs.keys()
        for node in self.nodeList:
            with tf.variable_scope(node, reuse=self.reuse):
                edges_connected_to = self.en_connections[node]
                right = []
                left = []
                one_input = []
                if node == "arms":
                    for edge in edges_connected_to:
                        if edge is not "sum_arms":
                            right.append(self.edgeRNNs[edge].outputs[0])
                            left.append(self.edgeRNNs[edge].outputs[1])
                        elif edge is "sum_arms":
                            right.append(self.edgeRNNs[edge].outputs[0])
                            left.append(self.edgeRNNs[edge].outputs[0])
                    right.append(self.nodeFCs[node].dense2[0])
                    left.append(self.nodeFCs[node].dense2[1])
                    self.nodeRNN_input[node] = [tf.concat(right, axis=2), tf.concat(left, axis=2)]
                elif node == "legs":
                    for edge in edges_connected_to:
                        if edge is not "sum_legs":
                            right.append(self.edgeRNNs[edge].outputs[0])
                            left.append(self.edgeRNNs[edge].outputs[1])
                        else:
                            right.append(self.edgeRNNs[edge].outputs[0])
                            left.append(self.edgeRNNs[edge].outputs[0])
                    right.append(self.nodeFCs[node].dense2[0])
                    left.append(self.nodeFCs[node].dense2[1])
                    self.nodeRNN_input[node] = [tf.concat(right, axis=2), tf.concat(left, axis=2)]
                else:  # spine
                    for edge in edges_connected_to:
                        one_input.append(self.edgeRNNs[edge].outputs[0])
                    one_input.append(self.nodeFCs[node].dense2[0])
                    self.nodeRNN_input[node] = [tf.concat(one_input, axis=2)]

        concat_features = []

        with tf.variable_scope("spine_concatenated",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.nodeRNNs["spine"] = nodeRNN(cell,self.num_hidden,self.nodeRNN_input["spine"]).outputs
        with tf.variable_scope("arms_concatenated",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.nodeRNNs["arms"] = nodeRNN(cell,self.num_hidden,self.nodeRNN_input["arms"]).outputs
        with tf.variable_scope("legs_concatenated",reuse=self.reuse):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
            self.nodeRNNs["legs"] = nodeRNN(cell,self.num_hidden,self.nodeRNN_input["legs"]).outputs

        concat_features = [self.nodeRNNs["spine"], self.nodeRNNs["arms"], self.nodeRNNs["legs"]]
       
        with tf.variable_scope("grandaddy_rnn", reuse=self.reuse):
            self.grandaddy_input = tf.concat(concat_features, axis=2)
            print("final rnn input = ", self.grandaddy_input)
            self.grandaddy_cell = tf.contrib.rnn.LSTMCell(1024)
            self.grandaddy_node_rnn = nodeRNN(self.grandaddy_cell, 512, [self.grandaddy_input]).outputs  # input: cell, num_hidden
            ############## CHECK THIS -1 INDEX DUE TO TIME MAJOR=FALSE ##############
            self.grandaddy_output = self.grandaddy_node_rnn[-1, :, :]
            print("final rnn output = ", self.grandaddy_output)
            self.outputs0 = tf.layers.dense(self.grandaddy_output, 256, activation=tf.nn.relu)
            self.embeddings = tf.layers.dense(self.outputs0, 128, activation=tf.nn.relu)

        return self.embeddings

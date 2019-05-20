import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import copy
import json
import time
from SRNN_model_1 import SRNN_model

DATA_FILE = "3_24_100_inputs.pickle"
LABEL_FILE = "3_24_100_labels.pickle"
INDICES_FILE = "80_20_indices.txt"

tf.app.flags.DEFINE_string("data_file",DATA_FILE, "Path to data file.")
tf.app.flags.DEFINE_string("label_file",LABEL_FILE,"Path to labels file.")
tf.app.flags.DEFINE_string("indices_file",INDICES_FILE,"Path to indices file.")
tf.app.flags.DEFINE_float("learning_rate",0.00003, "Learning rate to use for training. Will decay over time.")
tf.app.flags.DEFINE_integer("num_epochs",55,"Number of epochs to use for training.")
TF_FLAGS = tf.app.flags.FLAGS

_pad_width = 15
print("Training Parameters".center(45,"-"))
print("data_file:".ljust(20),TF_FLAGS.data_file)
print("label_file:".ljust(20),TF_FLAGS.label_file)
print("indices_file:".ljust(20),TF_FLAGS.indices_file)
print("learning_rate:".ljust(20),TF_FLAGS.learning_rate)
print("num_epochs:".ljust(20),TF_FLAGS.num_epochs)
print("-"*45)


def get_one_hot(label_list):
    l = np.unique(label_list)
    eye = np.eye(len(l))
    key_map = {sid: eye[ind] for ind, sid in enumerate(l)}
    return key_map


def train_softmax(train_data, train_labels, val_data, val_labels, batch_size=1, num_epochs=40, learning_rate=0.00003):
    # 2. Feedforward pass
    # 3. Get "embedding"
    # 4. loss softmax_embedding vs label

    current_best_acc = 0
    onehot = get_one_hot(train_labels)

    srnn = SRNN_model(128)
    plc = tf.placeholder(tf.float32, (None, 1, 165))
    label = tf.placeholder(tf.float32, (1, 100))
    max_label = tf.argmax(label, axis=-1)
    output = srnn(plc, reuse=False)

    final_dense = tf.layers.dense(output, 100, activation=tf.nn.tanh)
    smax = tf.nn.softmax(final_dense)
    prediction = tf.argmax(smax,axis=1)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=max_label, logits=final_dense)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_grads, norm = tf.clip_by_global_norm(gradients, 1)
    updates = optimizer.apply_gradients(zip(clipped_grads, params))

    optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate*0.9)
    updates_2 = optimizer_2.apply_gradients(zip(clipped_grads, params))

    saver = tf.train.Saver(max_to_keep=10)
    best_saver = tf.train.Saver(max_to_keep=4)
    writer = tf.summary.FileWriter("logdir")
    writer.add_graph(tf.get_default_graph())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_loss = []
        train_acc = []
        val_loss = []
        val_accs = []
        first_time = True
        for epoch in range(num_epochs):
            if epoch > 10 and first_time:
                updates = updates_2
                first_time = False

            # Permute the training set each epoch
            perm_idcs = np.random.permutation(len(train_data))
            train_data = [train_data[i] for i in perm_idcs]
            train_labels = [train_labels[i] for i in perm_idcs]

            # Keep a list of predictions
            train_pred_list = []
            val_pred_list = []
            # Accumulate loss
            acc_loss = 0
            val_acc_loss = 0
            for i in range(len(train_data)):
                _, [loss_val], [pred_val] = sess.run([updates, loss, prediction],feed_dict={plc: train_data[i],
                                                                                      label: np.expand_dims(onehot[train_labels[i]],axis=0)})
                print("Iteration %d" % i)
                acc_loss += loss_val
                train_pred_list.append(pred_val)
            print(("Epoch %d" % epoch).center(35,"-"))
            print("Loss: %f" % (acc_loss/len(train_data)))
            train_loss.append(acc_loss/len(train_data))
            oh_train_labels = [np.argmax(onehot[train_labels[i]]) for i in range(len(train_labels))]
            oh_val_labels = [np.argmax(onehot[val_labels[i]]) for i in range(len(val_labels))]
            print("Train Acc: %f" % accuracy_score(oh_train_labels,train_pred_list))
            train_acc.append(accuracy_score(oh_train_labels,train_pred_list))
            train_unique = np.unique(train_labels)
            train_labels_ordered = [np.argmax(onehot[x]) for x in train_unique]

            plt.figure(figsize=(16,16))
            train_mat = conf(oh_train_labels,train_pred_list,labels=train_labels_ordered)
            np.savetxt("train_cfmat_epoch_%d.txt" % epoch,train_mat)
            plt.imshow(train_mat)
            plt.imshow(conf(oh_train_labels,train_pred_list,labels=train_labels_ordered))
            plt.xticks(range(train_unique.shape[0]),train_unique,rotation="vertical")
            plt.yticks(range(train_unique.shape[0]),train_unique)
            plt.savefig("train_conf_mat_epoch_%d.PNG" % epoch)
            plt.show()
            # Validation data
            for i in range(len(val_data)):
                [loss_val], [pred_val] = sess.run([loss, prediction],feed_dict={plc: val_data[i],
                                                                                      label: np.expand_dims(onehot[val_labels[i]],axis=0)})
                print("Iteration %d" % i)
                val_acc_loss += loss_val
                val_pred_list.append(pred_val)

            print("Validation Loss: %f" % (val_acc_loss/len(val_data)))
            val_loss.append(val_acc_loss/len(val_data))
            print("Validation Acc: %f" % accuracy_score(oh_val_labels,val_pred_list))
            val_acc = accuracy_score(oh_val_labels,val_pred_list)
            val_accs.append(val_acc)

            val_unique = np.unique(val_labels)
            val_labels_ordered = [np.argmax(onehot[x]) for x in val_unique]
            
            print(cr(oh_val_labels,val_pred_list,labels=val_labels_ordered,target_names=val_unique))
            with open("losses.txt","a+") as f:
                f.write("%f,%f\n" % (train_loss[-1],val_loss[-1])) 
            with open("accuracy.txt","a+") as f:
                f.write("%f,%f\n" % (train_acc[-1],val_accs[-1]))
            plt.figure(figsize=(16,16))
            val_mat = conf(oh_val_labels,val_pred_list,labels=val_labels_ordered)
            np.savetxt("val_cfmat_epoch_%d.txt",val_mat)
            plt.imshow(val_mat)
            plt.xticks(range(val_unique.shape[0]),val_unique,rotation="vertical")
            plt.yticks(range(val_unique.shape[0]),val_unique)
            plt.savefig("val_conf_mat_epoch_%d.PNG" % epoch)
            
            if val_acc > current_best_acc:
                current_best_acc = val_acc
                best_saver.save(sess,"best_model/model",global_step=epoch)
                print("Updating Best Model.")
                
            saver.save(sess,"save_dir/model",global_step = epoch)
            print("Saved model.")


def get_embeddings(data):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.reset_default_graph()


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        srnn = SRNN_model(128)
        placeholder = tf.placeholder(tf.float32, (None, 1, 165))
        output = srnn(placeholder, reuse=False)
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint("save_dir"))

        embeddings = []
        for i in range(len(data)):
            embedded_out = sess.run(output,feed_dict={placeholder:data[i]})
            embeddings.append(embedded_out)

    return embeddings


def make_softmax_data(data):
    concat_arr = []
    for i in range(len(data[0])):
        # For all joints
        temp_arr = []
        for j in range(len(data)):
            temp_arr.append(data[j][i])
        stacked = np.concatenate(temp_arr, axis=-1)
        concat_arr.append(stacked)
    return concat_arr


def split_data(data,labels,test_size=0.2,debug_log=True):
    labels = np.array(labels)
    data = np.array(data)
    #np.random.seed(123456)
    label_set = np.unique(labels)
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    indices_dict = {}
    for label in label_set:
        print(str(label).center(35,"-"))
        # Get the runs that correspond to this label.
        label_runs = np.squeeze(np.argwhere(labels == label))
        # Get the number of runs for this label (for calculating the train/test sizes)
        num_runs = label_runs.shape[0]
        num_test = int(test_size*num_runs)
        num_train = num_runs - num_test
        if debug_log:
            print("Total # of runs: %d" % num_runs)
            print("Train runs: %d" % num_train)
            print("Test runs: %d" % num_test)
        # Choose the indices randomly from set of all indices
        train_idcs = np.random.choice(label_runs,num_train,replace=False)
        test_idcs = np.setdiff1d(label_runs,train_idcs)
        if debug_log:
            print("Train indices", train_idcs)
            print("Test indices", test_idcs)
        train_data.append(data[train_idcs])
        train_labels.append(labels[train_idcs])
        test_data.append(data[test_idcs])
        test_labels.append(labels[test_idcs])
        indices_dict[label] = (train_idcs.tolist(),test_idcs.tolist())
    
    print("Writing indices to file...")
    with open("indices.txt","w") as f:
        json.dump(indices_dict,f)
    print("Finished writing.")
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)

    return train_data,test_data,train_labels,test_labels

# Split data based on existing indices.
def split_data_existing(arrays, labels, indices):
    split_arrs = []

    # Build labels first
    train_labels = []
    test_labels = []
    for lbl in np.unique(labels):
        train_labels.append(labels[indices[lbl][0]])
        test_labels.append(labels[indices[lbl][1]])
    train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)
    for arr in arrays:
        train_data = []
        test_data = []
        for lbl in np.unique(labels):
            train_data.append(arr[indices[lbl][0]])
            test_data.append(arr[indices[lbl][1]])
        train_data = np.concatenate(train_data)
        test_data = np.concatenate(test_data)

        split_arrs.append([train_data, test_data])
    return split_arrs, train_labels, test_labels

p_1_in = open(TF_FLAGS.data_file, "rb")
p_2_in = open(TF_FLAGS.label_file, "rb")
images = pickle.load(p_1_in)
labels = pickle.load(p_2_in)
images = images[0]
labels = np.array(labels)

#left_out_sid = np.random.choice(np.unique(labels),20,replace=False)

indices = json.load(open(TF_FLAGS.indices_file,"r"))

data = make_softmax_data(images)
data = np.array(data)
print(data.shape)

[[trX,testX]],trY, testY = split_data_existing([data],labels,indices)

# print("Size of Train Data prior: %d" % trX.shape)
# print("Size of Train Labels prior: %d" % trY.shape)
# print("Size of Test Data prior: %d" % testX.shape)
# print("Size of Test Labels prior: %d" % testY.shape)

# trX = trX[~np.isin(trY,left_out_sid)]
# trY = trY[~np.isin(trY,left_out_sid)]
# testX = testX[~np.isin(testY,left_out_sid)]
# testY = testY[~np.isin(testY,left_out_sid)]

# print("Size of Train Data: %d" % trX.shape)
# print("Size of Train Labels: %d" % trY.shape)
# print("Size of Test Data: %d" % testX.shape)
# print("Size of Test Labels: %d" % testY.shape)

print("Number of Unique Labels in Train Data %d" % np.unique(trY).shape[0])

# np.savetxt("left_out_sid.txt",left_out_sid,fmt="%s",delimiter=",")

train_softmax(trX, trY, testX, testY, num_epochs=TF_FLAGS.num_epochs, learning_rate=TF_FLAGS.learning_rate)
embeddings = get_embeddings(data)
embeddings = np.concatenate(embeddings)
np.save("full_embeddings.npy",embeddings)
np.save("full_labels.npy",np.array(labels))

print("Done")

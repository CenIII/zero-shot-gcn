from __future__ import division
from __future__ import print_function

import time

import os
import tensorflow as tf
import pickle as pkl
import numpy as np

from utils import *
from models import GCN_dense_mse

import networkx as nx
import random
import sys
import json

def get_adj():    
    dataset_str = '../data/glove_res50/'
    with open("{}/ind.NELL.{}".format(dataset_str, 'graph'), 'rb') as f:
        print("{}/ind.NELL.{}".format(dataset_str, 'graph'))
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
        else:
            graph = pkl.load(f)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj

def get_2hop_neighbors(gind,adj,hop_depth=2):
    if hop_depth<=0:
        return []
    inds = []
    hops = np.argwhere(adj[gind]==1)
    for ind in hops:
        inds.append(ind[1])
        inds += get_2hop_neighbors(ind[1],adj,hop_depth-1)
    return inds

def get_hops_dict():
    data_dir = '../data/list/'
    classids_file_retrain = os.path.join(data_dir, 'corresp-2-hops.json')   
    word2vec_file = '../data/word_embedding_model/glove_word2vec_wordnet.pkl'  
    with open(classids_file_retrain) as fp:
        classids = json.load(fp)

    with open(word2vec_file, 'rb') as fp:
        word2vec_feat = pkl.load(fp,encoding='latin1')

    adj = get_adj()

    valid_clss = np.zeros(22000)
    cnt_zero_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:
            twv = word2vec_feat[j]
            twv = twv / (np.linalg.norm(twv) + 1e-6)

            if np.linalg.norm(twv) == 0:
                cnt_zero_wv = cnt_zero_wv + 1
                continue
            valid_clss[classids[j][0]] = 1

    # process 'train' classes. they are possible candidates during inference
    cnt_zero_wv = 0
    labels_train, word2vec_train = [], []
    ind_remap = []
    hops_dict = {}
    for j in range(len(classids)):
        if classids[j][1] == 0 and classids[j][0] >= 0:     # preserve train classes only!!!
            twv = word2vec_feat[j]  
            if np.linalg.norm(twv) == 0:
                cnt_zero_wv = cnt_zero_wv + 1
                continue
            labels_train.append(classids[j][0])
            word2vec_train.append(twv)
            ind_remap.append(j)
            gind_2hpnbs = get_2hop_neighbors(j,adj)
            gind_2hpnbs = list(set(gind_2hpnbs))
            gind_2hpnbs.remove(j)
            random.shuffle(gind_2hpnbs)
            hops_dict[j] = gind_2hpnbs
    return hops_dict


def compute_mean_var(output,hops_dict):
    var_list = []
    for k in hops_dict:
        values = [hops_dict[k]]
        nbs = hops_dict[k]
        for j in nbs:
            values.append(output[j])
        var_list.append(np.var(np.array(values),axis=1))
    mean_var = np.mean(var_list)
    return mean_var

def compute_rand_var(output):
    var_list = []
    for i in range(1000):
        # random sample 5 numbers
        inds = np.random.choice(32200, 5)
        # get 5 values
        values = [output[inds[i]] for i in range(5)]
        var_list.append(np.var(np.array(values),axis=1))
    mean_var = np.mean(var_list)
    return mean_var

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../../data/glove_res50/', 'Dataset string.')
flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_string('save_path', '../output/', 'save dir')
flags.DEFINE_integer('epochs', 350, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

use_trainval = True
feat_suffix = 'allx_dense'

# Load data
adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask = \
        load_data_vis_multi(FLAGS.dataset, use_trainval, feat_suffix)

# Some preprocessing
features, div_mat = preprocess_features_dense2(features)

if FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(features.shape)

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

save_epochs = [300, 3000]

savepath = FLAGS.save_path
exp_name = os.path.basename(FLAGS.dataset)
savepath = os.path.join(savepath, exp_name)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('!!! Make directory %s' % savepath)
else:
    print('### save to: %s' % savepath)

# Train model
hops_dict = get_hops_dict()

now_lr = FLAGS.learning_rate
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr, model.outputs], feed_dict=feed_dict)

    # get neighbors indices of all known classes
    # calculate and average variances for all 1000 classes
    mean_var_1k = compute_mean_var(outs[4],hops_dict)
    # calculate and average variances for random classes
    mean_var_rand = compute_rand_var(outs[4])

    if epoch % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "mean_var_1k=","{:.5f}".format(mean_var_1k.sum()),"mean_var_rand=","{:.5f}".format(mean_var_rand.sum()),"train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))

    flag = 0
    for k in range(len(save_epochs)):
        if save_epochs[k] == epoch:
            flag = 1

    if flag == 1 or epoch % 500 == 0:
        outs = sess.run(model.outputs, feed_dict=feed_dict)
        filename = savepath + '/feat_' + os.path.basename(FLAGS.dataset) + '_' + str(epoch)
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()

print("Optimization Finished!")

sess.close()

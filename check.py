from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from skimage import color
import cv2
import random
import cPickle as pickle

def load_data(type):
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    f = open(filename, 'r')
    train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        # img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        # img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    if type=="train":
        return (rel_train, norel_train)
    else:
        return (rel_test, norel_test)
    # return (rel_train, rel_test, norel_train, norel_test)

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

def tensor_data(data, i, bs):
    img = np.asarray(data[0][bs*i:bs*(i+1)])
    qst = np.asarray(data[1][bs*i:bs*(i+1)])
    ans = np.asarray(data[2][bs*i:bs*(i+1)])
    return (img, qst, ans)




def train():
    """Train Relational network"""

    
    rel, norel = load_data("train")
    # batch_idxs = (len(rel)+len(norel))// self.batch_size
    batch_idxs = (len(rel))// 4  # training only for relational dataset
    print "len of rel 1=",len(rel)
    for epoch in xrange(2):

        
        print "batch size=",batch_idxs
        np.random.shuffle(rel)
        np.random.shuffle(norel)

        rel_norel=rel+norel  ####
        random.shuffle(rel_norel)####

        print "len of data=",len(rel_norel)
        rel_norel_tuple = cvt_data_axis(rel_norel) ####
        # norel_tuple = cvt_data_axis(norel) ####
        print "len of rel 3=",len(rel_norel)
        for idx in xrange(0, 3):
            print "len of rel 4=",len(rel_norel)
            img, qst, ans = tensor_data(rel_norel_tuple, idx, 4) ####
            print "len of rel 5=",len(rel_norel)

            print "Batch images shape", img.shape
            print "Batch question shape", qst.shape
            print "Batch answer shape", ans.shape

train()
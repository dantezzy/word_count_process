#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-11 14:21:03
# @Author  : Ziyi Zhao
# @Version : 1.0
# 1.0 : use word2vec to train the model
#       and use the kmean to do the cluster

import os
import nose
import numpy
import scipy
import gensim
import logging
import math
import sys 
import string
import pickle
# multi process
from collections import OrderedDict
from multiprocessing import Process, Lock, Manager
# word2vec process
import gensim
# kmean cluster

###############################################################################################################################################
# load the document from pickle file
def load_pickle(path):

	dataset_list = list()
	dataset_list = pickle.load(open(path, 'rb'))

	return dataset_list

###############################################################################################################################################
# save final document into file
def save_into_file(final_test_result):

	test_dataset_file = open('./OMDB_test_dataset_with_word_pair_based_2TFIDF.txt','w')
	for item in final_test_result:
		test_dataset_file.write("%s\n" % item)

	with open('./OMDB_test_dataset_with_word_pair_based_2TFIDF.pkl','w') as f:
	    pickle.dump(final_test_result,f)

###############################################################################################################################################
# save final document into file
def load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path):
	model = gensim.models.Word2Vec.load_word2vec_format(google_pre_trained_word2vec_model_path, binary=True)  
	return model

###############################################################################################################################################
# save final document into file
def train_word2vec_model():
	pass

###############################################################################################################################################
# save final document into file
def kmean_cluster():
	pass

###############################################################################################################################################
# run four separate parts
def run(train_dataset_path,test_dataset_pickle_path,google_pre_trained_word2vec_model_path):

	# train_dataset_list = load_pickle(train_dataset_path)

	# test_dataset_list = load_pickle(test_dataset_pickle_path)
	print "Load Google pre-trained word2vec model"
	model = load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path)
	print "Load finish"


###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_path = "../OMDB/OMDB_train_dataset.txt"
	test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	google_pre_trained_word2vec_model_path = "./word2vec_pre_trained_model/GoogleNews-vectors-negative300.bin"

	run(train_dataset_path,test_dataset_pickle_path,google_pre_trained_word2vec_model_path)
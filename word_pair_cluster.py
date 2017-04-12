#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-11 14:21:03
# @Author  : Ziyi Zhao
# @Version : 1.0
# 1.0 : use word2vec to train the model
#       and use the kmean to do the cluster

import os
import nose
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
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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
# add each word from list into set
def from_list_into_set(train_dataset_path):

	temp_set = set()
	train_dataset= open(train_dataset_path)
	for document in train_dataset:
		for word in document.split():
			temp_set.add(word)
	return temp_set


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
# convert each word in original dictionary into vector format
def convert_dictionary_into_vector(word2vec_model,original_dictionary):
	temp_vector_dictionary = dict()
	temp_vector_list = list()
	for word in original_dictionary:
		if word in word2vec_model.vocab:
			temp_vec = word2vec_model[word]
			# temp_vector_dictionary.update({word:temp_vec})
			temp_vector_list.append(temp_vec)
		# else:
		# 	print word
	# return temp_vector_dictionary
	return temp_vector_list

###############################################################################################################################################
# Kmean processing
def kmean_cluster(word_vector_dictionary):

	np_word_vector_dictionary = np.array(word_vector_dictionary)
	kmeans = KMeans(n_clusters=50, random_state=0).fit(np_word_vector_dictionary)

	print kmeans.labels_

###############################################################################################################################################
# run four separate parts
def run(train_dataset_path,google_pre_trained_word2vec_model_path):

	train_dataset_set = set()

	train_dataset_set = from_list_into_set(train_dataset_path)

	print "The size of the train dataset is :",len(train_dataset_set)

	# test_dataset_list = load_pickle(test_dataset_pickle_path)

	print "Load Google pre-trained word2vec model"
	word2vec_model = load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path)
	print "Load finish"

	print "Convert dictionary into vector"
	# word_vector_dictionary = dict()
	word_vector_dictionary = convert_dictionary_into_vector(word2vec_model,train_dataset_set)
	print "Convert finish"

	print "The size of the word vector dictionary is :",len(word_vector_dictionary)

	print "Start Kmean cluster"
	kmean_cluster(word_vector_dictionary)
	print "Cluster finish"

###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_path = "../OMDB/OMDB_train_dataset.txt"
	# test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	google_pre_trained_word2vec_model_path = "./word2vec_pre_trained_model/GoogleNews-vectors-negative300.bin"

	run(train_dataset_path,google_pre_trained_word2vec_model_path)
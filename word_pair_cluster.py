#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-11 14:21:03
# @Author  : Ziyi Zhao
# @Version : 1.1
# 1.1 : add PCA decomposition
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
# PCA decomposition
from sklearn.decomposition import PCA
# 3D plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy

class autovivify_list(dict):
        '''Pickleable class to replicate the functionality of collections.defaultdict'''
        def __missing__(self, key):
                value = self[key] = []
                return value

        def __add__(self, x):
                '''Override addition for numeric types when self is empty'''
                if not self and isinstance(x, Number):
                        return x
                raise ValueError

        def __sub__(self, x):
                '''Also provide subtraction method'''
                if not self and isinstance(x, Number):
                        return -1 * x
                raise ValueError

###############################################################################################################################################
# load the document from pickle file
def load_pickle(path):

	dataset_list = list()
	dataset_list = pickle.load(open(path, 'rb'))

	return dataset_list

###############################################################################################################################################
# load the document from pickle file into dictionary
def load_pickle_dict(path):

	dataset_dict = dict()
	dataset_dict = pickle.load(open(path, 'rb'))

	return dataset_dict

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
	temp_word_list = list()
	for word in original_dictionary:
		if word in word2vec_model.vocab:
			temp_vec = word2vec_model[word]
			# temp_vector_dictionary.update({word:temp_vec})
			temp_vector_list.append(temp_vec)
			temp_word_list.append(word)
		# else:
		# 	print word
	# return temp_vector_dictionary
	return temp_vector_list,temp_word_list

def find_word_clusters(labels_array, cluster_labels):
        '''Read the labels array and clusters label and return the set of words in each cluster'''
        cluster_to_words = autovivify_list()
        for c, i in enumerate(cluster_labels):
                cluster_to_words[ i ].append( labels_array[c] )
        return cluster_to_words
        
###############################################################################################################################################
# Kmean processing
def kmean_cluster(word_vector_dictionary,word_list):

	group_dictionary = dict()

	numberLoop = np.array([1000])

	np_word_vector_dictionary = np.array(word_vector_dictionary)
	# pca = PCA(n_components=2)
	# newData = pca.fit_transform(np_word_vector_dictionary)
	# #print newData
	# # print len(newData)
	# x = list()
	# y = list()
	# z = list()
	# for vector in newData:
	# 	x.append(vector[0]*100)
	# 	y.append(vector[1]*100)
	# 	# z.append(vector[2]*100)
	# # x = newData[:0]
	# # y = newData[:1]
	# # z = newData[:2]
	# np_x = np.array(x)
	# np_y = np.array(y)
	# # np_z = np.array(z)

	# #print np_z

	# fig = plt.figure()
	# # ax = fig.add_subplot(111, projection='3d')
	# ax = fig.add_subplot(1, 1, 1)
	# # ax.scatter(np_x, np_y,np_z)
	# plt.scatter(np_x, np_y)
	# plt.show()
	
	for n_clusters in numberLoop:
		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np_word_vector_dictionary) #n_clusters
		silhouette_avg = silhouette_score(np_word_vector_dictionary, kmeans.labels_,sample_size=50)
		print('For n_clusters = {0} The average silhouette_score is : {1}'.format(n_clusters, silhouette_avg))

	cluster_to_words  = find_word_clusters(word_list, kmeans.labels_)

	for c in cluster_to_words:
		# print c,cluster_to_words[c],"\n"
		group_dictionary.update({c:list(cluster_to_words[c])})

	with open('./group_dictionary.pkl','w') as f:
	    pickle.dump(group_dictionary,f)


###############################################################################################################################################
# run four separate parts
def run(train_dataset_path,google_pre_trained_word2vec_model_path):
	default_mode = 'load'

	if default_mode == 'train':

		train_dataset_set = set()

		train_dataset_set = from_list_into_set(train_dataset_path)

		print "The size of the train dataset is :",len(train_dataset_set)

		# test_dataset_list = load_pickle(test_dataset_pickle_path)

		print "Load Google pre-trained word2vec model"
		word2vec_model = load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path)
		print "Load finish"

		print "Convert dictionary into vector"
		# word_vector_dictionary = dict()
		word_vector_dictionary,word_list = convert_dictionary_into_vector(word2vec_model,train_dataset_set)
		print "Convert finish"

		print "The size of the word vector dictionary is :",len(word_vector_dictionary)

		print "Start Kmean cluster"
		kmean_cluster(word_vector_dictionary,word_list)
		print "Cluster finish"

	if default_mode == 'load':
		group_dictionary = load_pickle_dict('./group_dictionary.pkl')

		for key,value in group_dictionary.items():
			print(key,value)

###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_path = "../OMDB/OMDB_train_dataset.txt"
	# test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	google_pre_trained_word2vec_model_path = "./word2vec_pre_trained_model/GoogleNews-vectors-negative300.bin"

	run(train_dataset_path,google_pre_trained_word2vec_model_path)
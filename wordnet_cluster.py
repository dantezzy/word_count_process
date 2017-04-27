#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-26 21:13:45
# @Author  : Ziyi Zhao
# @Version : 1.0

import os
import pickle
import operator
import numpy as np
# For wordnet process
from nltk.corpus import wordnet as wn
from itertools import islice
# word2vec process
import gensim
# multi process
from collections import OrderedDict
from multiprocessing import Process, Lock, Manager
# PCA decomposition
from sklearn.decomposition import PCA
# 3D plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# convert each word in original dictionary into vector format
def convert_dictionary_into_vector(word2vec_model,original_dictionary):
	temp_vector_dictionary = dict()
	temp_filter_word = set()
	temp_vector_list = list()
	for word in original_dictionary:
		if word in word2vec_model.vocab:
			temp_vec = word2vec_model[word]
			temp_vector_list.append(temp_vec)
			temp_filter_word.add(word)
		# else:
		# 	print word
	return temp_vector_list,temp_filter_word


###############################################################################################################################################
# wordnet find path and sorting
def wordnet_sort(word_set):

	hyper = lambda s: s.hypernyms()
	count = 0
	# test_list = set()
	path_dict = dict()

# iterate all word set
	for word in word_set:
		temp_list = list()
# generate all synsets for this word
		temp_list = wn.synsets(word)
# iterate all synset word in the synsets list
		minimum_path = 9223372036854775807
		minimum_word = ''
		for elem in temp_list:
# use int max initialize minimum path 
			temp_word = str(elem)[8:-2]
			if temp_word.find(word) != -1 and '.n.' in temp_word: # noum and contain its own meaning
# get the path from this word to wordnet root
				temp = list(elem.closure(hyper))
				length = len(temp)
				if length != 0 and length < minimum_path:
					minimum_path = length
					minimum_word = temp_word
					count += 1
		if minimum_word != '':
			print word
			print minimum_path
			print "\n"
			# test_list.add(word)
			path_dict.update({word:minimum_path})

	# print "Total number is :",len(test_list)
	# print "Total number is :",len(path_dict)
	sorted_x = sorted(path_dict.items(), key=operator.itemgetter(1))

	return sorted_x


###############################################################################################################################################
# cut default number of word from path dictionary
def dictionary_filter(sorted_x,default_number_of_word):

	top_rank_list = list()

	current_distance = 0

	for pair in sorted_x:
		if len(top_rank_list) < default_number_of_word :
			top_rank_list.append(pair[0])
		# if len(top_rank_list) == default_number_of_word and current_distance == pair[1]:
		# 	top_rank_list.append(pair[0])
		current_distance = pair[1]

	print "The length of list :",len(top_rank_list)

	return top_rank_list

###############################################################################################################################################
# vector visualizaiton
def visualize_vector(top_rank_list,word2vec_model):

	word_vector = list()

	for word in top_rank_list:
		temp_vec = word2vec_model[word]
		word_vector.append(temp_vec)

	np_word_vector_dictionary = np.array(word_vector)
	pca = PCA(n_components=2)
	newData = pca.fit_transform(np_word_vector_dictionary)
	#print newData
	# print len(newData)
	x = list()
	y = list()
	z = list()
	for vector in newData:
		x.append(vector[0]*100)
		y.append(vector[1]*100)
		# z.append(vector[2]*100)
	# x = newData[:0]
	# y = newData[:1]
	# z = newData[:2]
	np_x = np.array(x)
	np_y = np.array(y)
	# np_z = np.array(z)

	#print np_z

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	ax = fig.add_subplot(1, 1, 1)
	# ax.scatter(np_x, np_y,np_z)
	plt.scatter(np_x, np_y)
	plt.show()

###############################################################################################################################################
# run four separate parts
def run(train_dataset_path,google_pre_trained_word2vec_model_path):

	train_dataset_set = set()

	train_dataset_set = from_list_into_set(train_dataset_path)

	default_number_of_word = 1000 

	print "The size of the train dataset is :",len(train_dataset_set)

	print "Load Google pre-trained word2vec model"
	word2vec_model = load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path)
	print "Load finish"

	print "Convert dictionary into vector"
	# word_vector_dictionary = dict()
	word_vector_dictionary,word_set = convert_dictionary_into_vector(word2vec_model,train_dataset_set)
	print "Convert finish"

	print "The size of the word vector dictionary is :",len(word_vector_dictionary)
	print "The size of the word set is :",len(word_set)

	print "Start wordnet processing"
	sorted_word_dict = wordnet_sort(word_set)
	print "Wordnet finish"

	print "Get default number of word"
	top_rank_list = dictionary_filter(sorted_word_dict,default_number_of_word)
	print "Wordnet finish"

	#print resutl_list

	print "Start vector visualization"
	visualize_vector(top_rank_list,word2vec_model)
	print "Visualizaiton finish"

###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_path = "../OMDB/OMDB_train_dataset.txt"
	# test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	google_pre_trained_word2vec_model_path = "./word2vec_pre_trained_model/GoogleNews-vectors-negative300.bin"

	run(train_dataset_path,google_pre_trained_word2vec_model_path)
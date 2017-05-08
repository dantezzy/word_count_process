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
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# calculate distance
from scipy.spatial import distance

###############################################################################################################################################
# load the document from pickle file into list
def load_pickle_list(path):

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
# convert each word in original dictionary into vector format
def convert_dictionary_into_vector(word2vec_model,original_dictionary):
	temp_vector_dictionary = dict()
	temp_filter_word = set()
	temp_vector_list = list()
	for word in original_dictionary:
		if word in word2vec_model.vocab:
			temp_vec = word2vec_model[word]
			temp_vector_dictionary.update({word:temp_vec})
			temp_filter_word.add(word)
		# else:
		# 	print word
	return temp_vector_dictionary,temp_filter_word


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
		maximum_path = -9223372036854775807
		minimum_word = ''
		for elem in temp_list:
# use int max initialize minimum path 
			temp_word = str(elem)[8:-2] ## shoule be "str(elem)[8:-2]" # wordnet: "temp_word = str(elem)"
			if temp_word.find(word) != -1 and '.n.' in temp_word: # noum and contain its own meaning
# get the path from this word to wordnet root
				temp = list(elem.closure(hyper))
				length = len(temp)
				if length != 0 and length > maximum_path:
					maximum_path = length
					minimum_word = temp_word ### should be "minimum_word = temp_word" # wordnet: "minimum_word = temp_word"
					count += 1
		if minimum_word != '':
			# print word
			# print minimum_path
			# print "\n"
			# test_list.add(word)
			path_dict.update({word:maximum_path}) # should be "path_dict.update({word:minimum_path})" # wordnet: "path_dict.update({minimum_word:minimum_path})"

	# print "Total number is :",len(test_list)
	# print "Total number is :",len(path_dict)
	sorted_x = sorted(path_dict.items(), key=operator.itemgetter(1))

	return sorted_x


###############################################################################################################################################
# cut default number of word from path dictionary
def dictionary_filter(sorted_x,default_number_of_word):

	top_rank_list = list()
	distribution_dict = dict()

	current_distance = 0

	manually_set = ['handel','then','jabberwocky','batiste','maxwell','camorra','ido','drey','neve',
	                'bengali','n','ro','s','varna','alabama','weber','wagner','en','tammy','mackintosh','frankenstein']

	for pair in sorted_x:
		if len(top_rank_list) < default_number_of_word :
			if pair[0] not in manually_set: #should be "if pair[0] not in manually_set:" # wordnet:"str(pair[0])[8:-2] not in manually_set:""
				top_rank_list.append(pair[0])
				distribution_dict.update({pair[0]:0})
		# if len(top_rank_list) == default_number_of_word and current_distance == pair[1]:
		# 	top_rank_list.append(pair[0])
		current_distance = pair[1]

	print "The length of list :",len(top_rank_list)

	return top_rank_list,distribution_dict

###############################################################################################################################################
# vector visualizaiton
def visualize_vector(top_rank_list,word2vec_model,word_set):

	key_word_vector = list()
	normal_word_vector = list()

	for key_word in top_rank_list:
		key_temp_vec = word2vec_model[key_word]
		key_word_vector.append(key_temp_vec)

	for normal_word in word_set:
		if normal_word not in top_rank_list:
			normal_temp_vec = word2vec_model[normal_word]
			normal_word_vector.append(normal_temp_vec)

	np_key_word_vector_dictionary = np.array(key_word_vector)
	np_normal_word_vector_dictionary = np.array(normal_word_vector)
	pca = PCA(n_components=2)
	key_newData = pca.fit_transform(np_key_word_vector_dictionary)
	normal_newData = pca.fit_transform(np_normal_word_vector_dictionary)
	#print newData
	# print len(newData)
	key_x = list()
	key_y = list()
	key_z = list()
	for key_vector in key_newData:
		key_x.append(key_vector[0]*100)
		key_y.append(key_vector[1]*100)
		# z.append(vector[2]*100)
	# x = newData[:0]
	# y = newData[:1]
	# z = newData[:2]
	key_np_x = np.array(key_x)
	key_np_y = np.array(key_y)
	# np_z = np.array(z)

	#print np_z

	normal_x = list()
	normal_y = list()
	normal_z = list()
	for normal_vector in normal_newData:
		normal_x.append(normal_vector[0]*100)
		normal_y.append(normal_vector[1]*100)
		# z.append(vector[2]*100)
	# x = newData[:0]
	# y = newData[:1]
	# z = newData[:2]
	normal_np_x = np.array(normal_x)
	normal_np_y = np.array(normal_y)
	# np_z = np.array(z)

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	ax = fig.add_subplot(1, 1, 1)
	# ax.scatter(np_x, np_y,np_z)
	#plt.scatter(normal_np_x, normal_np_y,color='blue')
	plt.scatter(key_np_x, key_np_y,color='blue')
	plt.show()

###############################################################################################################################################
# Calculate and sort all word in dictionary with the represented word
def calculate_similarity_and_sort(top_rank_list,word_vector_dictionary,distribution_dict):

	dict_word_count = 0
	sorted_word_dict = dict()
	distribution = dict()
	distribution = distribution_dict.copy()

	for key,value in word_vector_dictionary.items():
# if not the repesented word
		if key not in top_rank_list:
# create a rank list to store all distance between current word and each representd word
			rank_list = dict()
			for import_word in top_rank_list:
# calculate the L2 distance
				dst = distance.euclidean(value,word_vector_dictionary[import_word]) # word2vec based similarity calculation
				# dst = wordnet_similarity(key,import_word)
# store represented word : distance pair into the rank list dictionary # wordnet based similarity calculation
				rank_list.update({import_word:dst})
# sort rank list based on the distancer	
			rank_list=OrderedDict(sorted(rank_list.items(),key=lambda t:t[1], reverse=False)) # "reverse=True" only suitable for wordnet similarity
			# print rank_list
			represented_word = list(rank_list)[:1]
			sorted_word_dict.update({key:represented_word[0]})
			# print represented_word[0]
			distribution[represented_word[0]] += 1
			dict_word_count += 1
		print "Process word:",dict_word_count

	pickle.dump(sorted_word_dict, open('./represented_word_corresponding_relationship.pkl','wb'))


	return sorted_word_dict,distribution


###############################################################################################################################################
# Calculate and sort all word in dictionary with the represented word
def wordnet_similarity(dictionary_word,import_word):

	temp_list = list()
# generate all synsets for this word
	temp_list = wn.synsets(dictionary_word)
# iterate all synset word in the synsets list
	minimum_path = 9223372036854775807
	maximum_path = -9223372036854775807
	minimum_word = ''
	for elem in temp_list:
# use int max initialize minimum path 
		temp_word = str(elem)[8:-2] ## shoule be "str(elem)[8:-2]"
		if temp_word.find(dictionary_word) != -1: # noum and contain its own meaning #"and '.n.' in temp_word"
# get the path from this word to wordnet root
			#temp = list(elem.closure(hyper))
			length = wn.synset(temp_word).path_similarity(wn.synset(str(import_word)[8:-2]))
			if length !="None" and length < minimum_path:
				minimum_path = length


	if minimum_path == None:
		minimum_path = 0
	if minimum_path == 9223372036854775807:
		minimum_path = 0

	return minimum_path


###############################################################################################################################################
# run four separate parts
def run(train_dataset_path,google_pre_trained_word2vec_model_path):
	default_mode = 'train'

	if default_mode == 'train':

		train_dataset_set = set()

		train_dataset_set = from_list_into_set(train_dataset_path)

		default_number_of_word = 1000 

		print "The size of the train dataset is :",len(train_dataset_set),"\n"

		print "Load Google pre-trained word2vec model"
		word2vec_model = load_pre_trained_word2vec_model(google_pre_trained_word2vec_model_path)
		print "Load finish\n"

		print "Convert dictionary into vector"
		# word_vector_dictionary = dict()
		word_vector_dictionary,word_set = convert_dictionary_into_vector(word2vec_model,train_dataset_set)
		print "Convert finish\n"

		print "The size of the word vector dictionary is :",len(word_vector_dictionary)
		print "The size of the word set is :",len(word_set),"\n"

		print "Start wordnet processing"
		sorted_word_dict = wordnet_sort(word_set)
		print "Wordnet finish\n"

		print "Get default number of word"
		top_rank_list,distribution_dict = dictionary_filter(sorted_word_dict,default_number_of_word)
		print "Wordnet finish\n"
		# for word in top_rank_list:
		# 	print word

		# for key,value in distribution_dict.items():
		# 	print(key,value)

		print "Start vector visualization"
		# visualize_vector(top_rank_list,word2vec_model,word_set)
		print "Visualizaiton finish"

		print "Calculate and sort all word in dictionary"
		sorted_word_dict,word_dictribution = calculate_similarity_and_sort(top_rank_list,word_vector_dictionary,distribution_dict)
		print "Calculate and sort finish\n"

		print "The size of the sorted word dictionary is :",len(sorted_word_dict),"\n"

		word_dictribution=OrderedDict(sorted(word_dictribution.items(),key=lambda t:t[1], reverse=True))
		outputFile = open('./represented_distribution.txt', 'w')
		pickle.dump(word_dictribution, open('./represented_distribution.pkl','wb'))

		for key,value in word_dictribution.items():
			print(key,value)
			temp = '(\'' + str(key) + '\',' + str(value) + '\''+')'+'\n'
			outputFile.write(temp)

		# X = np.arange(len(word_dictribution))
		# pl.bar(X, word_dictribution.values(), align='center', width=0.5)
		# pl.xticks(X, word_dictribution.keys())
		# ymax = max(word_dictribution.values()) + 1
		# pl.ylim(0, ymax)
		# pl.show()
	

	if default_mode == 'load':

		temp_dict = dict()
		temp = set()
		temp_dict = load_pickle_dict("./represented_word_corresponding_relationship.pkl")

		for key,value in temp_dict.items():
			print(key,value)
			temp.update(value)
		#print temp
		# outputFile = open('./wordnet.txt', 'w')

		# for key,value in temp_dict.items():
		# 	temp = '(\'' + str(key) + '\',[' + str(value)[9:-9] + '\''+'])'+'\n'
		# 	outputFile.write(temp)

		print "The size of the sorted word dictionary is :",len(temp_dict),"\n"


###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_path = "../OMDB/OMDB_train_dataset.txt"
	# test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	google_pre_trained_word2vec_model_path = "./word2vec_pre_trained_model/GoogleNews-vectors-negative300.bin"

	run(train_dataset_path,google_pre_trained_word2vec_model_path)
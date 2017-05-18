#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-27 10:27:56
# @Author  : Ziyi Zhao
# @Version : 1.1
# 1.2: use represented word replace real word
# 1.1: batch processing for the train dataset
# 1.0: implement all basic functions

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
import itertools
# multi process
from collections import OrderedDict
from multiprocessing import Process, Lock, Manager


###############################################################################################################################################
# load the test document from pickle file
def load_pickle(path):

	train_dataset_list = list()
	train_dataset_list = pickle.load(open(path, 'rb'))

	return train_dataset_list

###############################################################################################################################################
# load the document from pickle file into dictionary
def load_pickle_dict(path):

	group_dict = dict()
	group_dict = pickle.load(open(path, 'rb'))

	return group_dict

###############################################################################################################################################
# single process convert real word into  group word
def single_process_group_word_converter(dataset,group_word_collection,start,end,group_dict):

	mode = "wordnet"

	if mode == "kmean":
		for word_pair_index in xrange(start,end):
# iterate each word pair in training dictionary
			group1 = ''
			group2 = ''
# iterate each word group in represented word relationships
			for index,group_word in group_dict.items():
# if a word in specific word group
				if dataset[word_pair_index][0] in group_word:
					group1 = index
				if dataset[word_pair_index][1] in group_word:
					group2 = index
			pair = list()
			pair.append(group1)
			pair.append(group2)
			print "Process word pair:",word_pair_index

			group_word_collection.append(pair)

	if mode == "wordnet":
		for word_pair_index in xrange(start,end):
# iterate each word pair in training dictionary
			group1 = dataset[word_pair_index][0]
			group2 = dataset[word_pair_index][1]
# if a word in specific word group
			if group_dict.has_key(dataset[word_pair_index][0]):
				group1 = group_dict[dataset[word_pair_index][0]]
			if group_dict.has_key(dataset[word_pair_index][1]):
				group2 = group_dict[dataset[word_pair_index][1]]
			pair = list()
			pair.append(group1)
			pair.append(group2)
			print "Process word pair:",word_pair_index

			group_word_collection.append(pair)

###############################################################################################################################################
# multi process convert real word into  group word
def multi_process_group_word_converter(dataset,number_of_process,group_dict):

	manager = Manager()
	group_word_collection = manager.list()
	segment = len(dataset)/number_of_process

	all_processes = [Process(target=single_process_group_word_converter, args=(dataset, group_word_collection,x*segment, (x+1)*segment,group_dict)) for x in xrange(0,number_of_process)]

	for p in all_processes:
		p.start()

	P_last=Process(target=single_process_group_word_converter, args=(dataset, group_word_collection,number_of_process*segment, len(dataset),group_dict))
	if number_of_process*segment < len(dataset):
		P_last.start()

	for p in all_processes:
		  p.join()

	if number_of_process*segment < len(dataset):
		P_last.join()

	return group_word_collection

###############################################################################################################################################
# single process parser
def single_process_word_count(dataset,word_pair_collection,start,end,word_dict,group_dict):

	mode = "wordnet"

	if mode == "kmean":
		for doc_count in xrange(start,end):

			temp_dataset_list = list()
			for word_pair_in_dataset in dataset[doc_count]:
				group1 = ''
				group2 = ''
				for index,group_word in group_dict.items():
	# if a word in specific word group
					if word_pair_in_dataset[0] in group_word:
						group1 = index
					if word_pair_in_dataset[1] in group_word:
						group2 = index
				pair = list()
				pair.append(group1)
				pair.append(group2)
				temp_dataset_list.append(pair)

			document_word_count_temp = list()
			# print temp_dataset_list
			for word_pair in word_dict:
				frequency = temp_dataset_list.count(word_pair)
				# if frequency  != 0:
					# print frequency
				document_word_count_temp.append(frequency)
			print "Doc:",doc_count
			#print document_word_count_temp
			word_pair_collection.update({doc_count:document_word_count_temp})

	if mode == "wordnet":
		for doc_count in xrange(start,end):

			temp_dataset_list = list()
			for word_pair_in_dataset in dataset[doc_count]:
				group1 = ''
				group2 = ''
				for index,group_word in group_dict.items():
	# if a word in specific word group
					if word_pair_in_dataset[0] in group_word:
						group1 = index
					if word_pair_in_dataset[1] in group_word:
						group2 = index
				pair = list()
				pair.append(group1)
				pair.append(group2)
				temp_dataset_list.append(pair)

			document_word_count_temp = list()
			# print temp_dataset_list
			for word_pair in word_dict:
				frequency = temp_dataset_list.count(word_pair)
				# if frequency  != 0:
					# print frequency
				document_word_count_temp.append(frequency)
			print "Doc:",doc_count
			#print document_word_count_temp
			word_pair_collection.update({doc_count:document_word_count_temp})

###############################################################################################################################################
# multi process parser
def multi_process_word_count(dataset,number_of_process,word_dict,group_dict):

	manager = Manager()
	word_pair_collection = manager.dict()
	word_pair_collection_dict = dict()
	segment = len(dataset)/number_of_process
	word_count_list = list()

	all_processes = [Process(target=single_process_word_count, args=(dataset, word_pair_collection,x*segment, (x+1)*segment,word_dict,group_dict)) for x in xrange(0,number_of_process)]

	for p in all_processes:
		p.start()

	P_last=Process(target=single_process_word_count, args=(dataset, word_pair_collection,number_of_process*segment, len(dataset),word_dict,group_dict))
	if number_of_process*segment < len(dataset):
		P_last.start()

	for p in all_processes:
		  p.join()

	if number_of_process*segment < len(dataset):
		P_last.join()

	word_pair_collection_dict = word_pair_collection

	word_pair_collection_dict=OrderedDict(sorted(word_pair_collection_dict.items(),key=lambda t:t[0]))

	for key, elem in word_pair_collection_dict.items():
		word_count_list.append(elem)

	return word_count_list

###############################################################################################################################################
# run separate parts
def run(train_dataset_pickle_path,train_dictionary_pickle_path,test_dataset_pickle_path):

	mode = "wordnet"

	if mode == "kmean":
		group_dict = load_pickle_dict("./group_dictionary.pkl")
	if mode == "wordnet":
		group_dict = load_pickle_dict("./represented_word_corresponding_relationship.pkl")

	# for key,value in group_dict.items():
	# 	print (key,value)

# process dictionary, convert real word into group index and remove duplicate
	train_dictionary = load_pickle(train_dictionary_pickle_path)
	group_word_dictionary = multi_process_group_word_converter(train_dictionary,8,group_dict)
	print "The length of original dict is:",len(train_dictionary)
	print "The length of group dict is:",len(group_word_dictionary)
	group_word_dictionary.sort()
	group_word_dictionary_without_duplicate = list(group_word_dictionary for group_word_dictionary,_ in itertools.groupby(group_word_dictionary))
	# print group_word_dictionary_without_duplicate
	print "The length of group dict set is:",len(group_word_dictionary_without_duplicate)
  	with open('./group_word_dictionary.pkl','w') as f:
	    pickle.dump(group_word_dictionary_without_duplicate,f)
######################################################################################################################

# process training and testing dataset
	train_dataset = load_pickle(train_dataset_pickle_path)

	test_dataset = load_pickle(test_dataset_pickle_path)

	train_dataset_word_count_path = open('./train_dataset_word_count.txt','w')
	train_dataset_word_count_pickle_path = open('./train_dataset_word_count.pkl','w')
	train_count = 0
	test_count = 0

	segment = len(train_dataset)/5
	result = list()

# batch processing
	for index in xrange(0,5):
		temp_train_dataset = list()
		train_dataset_word_count_list = list()
		for x in xrange(index*segment, (index+1)*segment):
			temp_train_dataset.append(train_dataset[x])
		train_dataset_word_count_list = multi_process_word_count(temp_train_dataset,8,group_word_dictionary_without_duplicate,group_dict)
		result.extend(train_dataset_word_count_list)
		train_count += len(train_dataset_word_count_list)

		dictionary_result='./train_dataset_word_count_'
		dictionary_result+=str(index+1)
		dictionary_result+='.pkl'

	  	with open(dictionary_result,'w') as f:
		    pickle.dump(train_dataset_word_count_list,f)

	if 5*segment < len(train_dataset):
		temp_train_dataset = list()
		train_dataset_word_count_list = list()
		for x in xrange(5*segment, len(train_dataset)):
			temp_train_dataset.append(train_dataset[x])
		train_dataset_word_count_list = multi_process_word_count(temp_train_dataset,8,train_dictionary,group_dict)
		result.extend(train_dataset_word_count_list)
		train_count += len(train_dataset_word_count_list)

		dictionary_result='./train_dataset_word_count_last.pkl'
		dictionary_result+='.pkl'

	  	with open(dictionary_result,'w') as f:
		    pickle.dump(train_dataset_word_count_list,f)

# save all in one file, txt and pickle
	train_dataset_word_count_path = open('./train_dataset_word_count.txt','a')
	for item in result:
		train_dataset_word_count_path.write("%s\n" % item)
  	# with open('./train_dataset_word_count.pkl','w') as f:
	  #   pickle.dump(result,f)


	test_dataset_word_count_list = multi_process_word_count(test_dataset,8,group_word_dictionary_without_duplicate,group_dict)
	test_count = len(test_dataset_word_count_list)
	test_dataset_word_count_path = open('./test_dataset_word_count.txt','w')
	for item in test_dataset_word_count_list:
		test_dataset_word_count_path.write("%s\n" % item)
  	with open('./test_dataset_word_count.pkl','w') as f:
	    pickle.dump(test_dataset_word_count_list,f)

	print "Number of train document: ",len(result)
	print "Number of test document: ",test_count

	print "Number of words in dictionary: ", len(group_word_dictionary_without_duplicate)
	print len(train_dataset)
	print len(test_dataset)

###############################################################################################################################################
if __name__ == '__main__':

	train_dictionary_pickle_path = "./dictionary_word_pair_with_2TFIDF.pkl"

	train_dataset_pickle_path = "./OMDB_train_dataset_with_word_pair_based_2TFIDF.pkl"

	test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_without_TFIDF.pkl"

	run(train_dataset_pickle_path,train_dictionary_pickle_path,test_dataset_pickle_path)
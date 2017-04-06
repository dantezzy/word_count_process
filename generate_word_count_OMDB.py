#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-27 10:27:56
# @Author  : Ziyi Zhao
# @Version : 1.1
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
# single process parser
def single_process_word_count(dataset,word_pair_collection,start,end,word_dict):

	for doc_count in xrange(start,end):
		document_word_count_temp = list()
		for word_pair in word_dict:
			frequency = dataset[doc_count].count(word_pair)
			# if frequency  != 0:
				# print frequency
			document_word_count_temp.append(frequency)	
		print "Doc:",doc_count	
		#print document_word_count_temp
		word_pair_collection.update({doc_count:document_word_count_temp})

###############################################################################################################################################
# multi process parser
def multi_process_word_count(dataset,number_of_process,word_dict): 

	manager = Manager()
	word_pair_collection = manager.dict()
	word_pair_collection_dict = dict()
	segment = len(dataset)/number_of_process
	word_count_list = list()

	all_processes = [Process(target=single_process_word_count, args=(dataset, word_pair_collection,x*segment, (x+1)*segment,word_dict)) for x in xrange(0,number_of_process)]

	for p in all_processes:
		p.start()

	P_last=Process(target=single_process_word_count, args=(dataset, word_pair_collection,number_of_process*segment, len(dataset),word_dict))
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

	train_dictionary = load_pickle(train_dictionary_pickle_path)

	train_dataset = load_pickle(train_dataset_pickle_path)

	test_dataset = load_pickle(test_dataset_pickle_path)

	train_dataset_word_count_path = open('./train_dataset_word_count.txt','w')
	train_dataset_word_count_pickle_path = open('./train_dataset_word_count.pkl','w')
	train_count = 0
	test_count = 0

	segment = len(train_dataset)/5
	result = list()

	for index in xrange(0,5):
		temp_train_dataset = list()
		train_dataset_word_count_list = list()
		for x in xrange(index*segment, (index+1)*segment):
			temp_train_dataset.append(train_dataset[x])
		train_dataset_word_count_list = multi_process_word_count(temp_train_dataset,8,train_dictionary)
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
		train_dataset_word_count_list = multi_process_word_count(temp_train_dataset,8,train_dictionary)
		result.extend(train_dataset_word_count_list)
		train_count += len(train_dataset_word_count_list)

		dictionary_result='./train_dataset_word_count_last.pkl'
		dictionary_result+='.pkl'

	  	with open(dictionary_result,'w') as f:
		    pickle.dump(train_dataset_word_count_list,f)

	train_dataset_word_count_path = open('./train_dataset_word_count.txt','a')
	for item in result:
		train_dataset_word_count_path.write("%s\n" % item)
  	with open('./train_dataset_word_count.pkl','w') as f:
	    pickle.dump(result,f)


	test_dataset_word_count_list = multi_process_word_count(test_dataset,8,train_dictionary)
	test_count = len(test_dataset_word_count_list)
	test_dataset_word_count_path = open('./test_dataset_word_count.txt','w')
	for item in test_dataset_word_count_list:
		test_dataset_word_count_path.write("%s\n" % item)
  	with open('./test_dataset_word_count.pkl','w') as f:
	    pickle.dump(test_dataset_word_count_list,f)

	print "Number of train document: ",len(result)
	print "Number of test document: ",test_count
	
	print "Number of words in dictionary: ", len(train_dictionary)
	# print len(train_dataset)
	# print len(test_dataset)

###############################################################################################################################################
if __name__ == '__main__':

	train_dictionary_pickle_path = "./dictionary_word_pair_with_2TFIDF.pkl"

	train_dataset_pickle_path = "./OMDB_train_dataset_with_word_pair_based_2TFIDF.pkl"

	test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_without_TFIDF.pkl"

	run(train_dataset_pickle_path,train_dictionary_pickle_path,test_dataset_pickle_path)
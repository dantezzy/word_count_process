#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-07 15:35:36
# @Author  : Ziyi Zhao
# @Version : 1.0
# 1.0 : process test dataset
#       add each document from test dataset into 7000 train dataset
#       implement word pair based TF-IDF processing
#       set default score as 0.01
#       use "aaa" be the index to combine a word pair
#       remove duplicate items in dictionary

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
# process stopset
from nltk.corpus import stopwords
# for TF-IDF
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

###############################################################################################################################################
# load the document from pickle file
def load_pickle(path):

	dataset_list = list()
	dataset_list = pickle.load(open(path, 'rb'))

	return dataset_list

###############################################################################################################################################
# combine word pair and save it into a temporal file
def save_into_temporal_file(train_dataset_list,test_document):

	temp_dataset_for_tfidf = []
# process train dataset
	for document in train_dataset_list:
		train_temp_str = ''
		for index in xrange(0,len(document)-1):
			# print document[index]
			temp_combination = document[index][0] + "aaa" + document[index][1]
			train_temp_str += temp_combination
			train_temp_str += ' '
		train_temp_str += document[len(document)-1][0] + "aaa" + document[len(document)-1][1]
		temp_dataset_for_tfidf.append(train_temp_str)

# process test document and add it into train dataset
	test_temp_str = ''
	for index in xrange(0,len(test_document)-1):
		# print document[index]
		temp_combination = test_document[index][0] + "aaa" + test_document[index][1]
		test_temp_str += temp_combination
		test_temp_str += ' '
	test_temp_str += test_document[len(test_document)-1][0] + "aaa" + test_document[len(test_document)-1][1]
	temp_dataset_for_tfidf.append(test_temp_str)

	# thefile = open('./temporal_train_file.txt','w')
	# for item in temp_dataset_for_tfidf:
	# 	thefile.write("%s\n" % item)

	return temp_dataset_for_tfidf

###############################################################################################################################################
# tfidf processing
def tfidf_process(temp_dataset_for_tfidf,test_document):

# TF-IDF process
	vectorizer = CountVectorizer(decode_error="replace")
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
	word = vectorizer.get_feature_names()

	word_dict = dict()
	word_count = 0
	for element in word:
		word_dict.update({element:word_count})
		word_count += 1

	weight = tfidf.toarray()

# remove the word with low TF-IDF score
	texts_after_tfidf = list()
	# print('Remove word based on TF-IDF')

	# print "Process DOC:",len(temp_dataset_for_tfidf)
	# print document
	temp_in_tfidf = list()
	for word_pair in test_document:
		temp_combination = word_pair[0] + "aaa" + word_pair[1]
		if check_tfidf(len(temp_dataset_for_tfidf)-1,word_dict,weight,temp_combination,0.1):
			temp_in_tfidf.append(word_pair)
		# print temp_in_tfidf
	texts_after_tfidf.append(temp_in_tfidf)

	return texts_after_tfidf

###############################################################################################################################################
# check the TF-IDF score
def check_tfidf(currentdoc,worddict,tfidfdict,currentword,score):
	try:
		index = worddict.get(currentword,None)
		if index != None:
			if tfidfdict[currentdoc][index]>score:
				return True
			return False
	except ValueError:
		return False
	return False

###############################################################################################################################################
# save final document into file
def save_into_file(final_test_result):

	test_dataset_file = open('./OMDB_test_dataset_with_word_pair_based_2TFIDF.txt','w')
	for item in final_test_result:
		test_dataset_file.write("%s\n" % item)

	with open('./OMDB_test_dataset_with_word_pair_based_2TFIDF.pkl','w') as f:
	    pickle.dump(final_test_result,f)

###############################################################################################################################################
# run four separate parts
def run(train_dataset_pickle_path,test_dataset_pickle_path):

	train_dataset_list = load_pickle(train_dataset_pickle_path)

	test_dataset_list = load_pickle(test_dataset_pickle_path)

	final_test_result = list()

	count = 0

	for test_document in test_dataset_list:
		temp_dataset_for_tfidf = save_into_temporal_file(train_dataset_list,test_document)
		texts_after_tfidf = tfidf_process(temp_dataset_for_tfidf,test_document) 
		final_test_result.append(texts_after_tfidf)
		print "process document :",count
		count += 1

	print "Total size is :",len(final_test_result)

	save_into_file(final_test_result)

###############################################################################################################################################
if __name__ == '__main__':

	train_dataset_pickle_path = "../OMDB/OMDB_train_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	test_dataset_pickle_path = "../OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl"
	run(train_dataset_pickle_path,test_dataset_pickle_path)

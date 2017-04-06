#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-02-21 20:55:05
# @Author  : Ziyi Zhao
# @Version : 2.7
# 2.7 : add enable/disable control
# 2.6 : remove the word pair which contains the stop word; remove the word pair 
#       by using TF-IDF
# 2.5 : use multi-process to improve the processing efficiency, and generate 
#       the dictionary set
# 2.0 : divide a pargraph into each sentence and parse each sentence to genreate 
#       the word pair. Combine  each unique word together to build the dictionary
# 1.0 : implement the basic function, run in test sentence, generate the word pair

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
# process stopset
from nltk.corpus import stopwords
# for TF-IDF
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# stanford parser
from nltk import tokenize
from nltk.parse import stanford
from nltk.parse import DependencyGraph
from nltk.parse.stanford import StanfordDependencyParser
# multi process
from collections import OrderedDict
from multiprocessing import Process, Lock, Manager

os.environ['STANFORD_PARSER'] = '/home/ziyizhao/stanford-parser-full-2016-10-31/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/ziyizhao/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'

path_to_jar = '/home/ziyizhao/stanford-parser-full-2016-10-31/stanford-parser.jar'

path_to_models_jar = '/home/ziyizhao/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'

tfidf_flag = "ENABLE_TFIDFG"

#parser = stanford.StanfordParser(model_path="/home/ziyizhao/stanford-parser-full-2016-10-31/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

###############################################################################################################################################
# generate the word pairs
def generate_word_pairs(train_dataset,test_dataset,new_train_path,new_test_path,new_dictionary_path,new_train_path_without_tfidf,new_dictionary_path_without_tfidf,new_test_path_without_tfidf):

# list for storing the word pairs for each document. Format will be [[a,a],[b,b],[c,c]],[[d,d],[e,e],[f,f]]
	documents = list()
# list for storing all word pairs to be the dictionary. Format will be [[a,a],[b,b],[c,c],[d,d],[e,e],[f,f]]
	train_dictionary_list = list()
# list for storing all word pairs to be the dictionary. Format will be [[a,a],[b,b],[c,c],[d,d],[e,e],[f,f]]
	test_dictionary_list = list()
# open three files for storing the new train, new test and dictionary data
	new_train_file = open(new_train_path, 'w')
	new_train_file_without_tfidf = open(new_train_path_without_tfidf, 'w')

	new_test_file = open(new_test_path, 'w')
	new_test_file_without_tfidf = open(new_test_path_without_tfidf, 'w')

	train_dictionary_file = open(new_dictionary_path, 'w')
	train_dictionary_file_without_tfidf = open(new_dictionary_path_without_tfidf, 'w')

# open original train and test file
	original_train_file = open(train_dataset)
	original_test_file = open(test_dataset)

# initialize the Stanford Dependency Parset	
	dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

# document number count
	count = 1
# default number of process
	default_number_of_process = 20
# train document word pair dictionary 
	train_word_pair_dictionary = dict()
# test document word pair dictionary 
	test_word_pair_dictionary = dict()
	
# load each document from train file into a dictionary
	train_dataset_dict = dict()
	train_count = 0
	for each_line_train_file in original_train_file:
		train_dataset_dict.update({train_count:each_line_train_file})
		train_count += 1

# load each document from test file into a dictionary
	test_dataset_dict = dict()
	test_count = 0
	for each_line_test_file in original_test_file:
		test_dataset_dict.update({test_count:each_line_test_file})
		test_count += 1

	if tfidf_flag == "ENABLE_TFIDFG":
		word_dict,weight = generate_tfidf(train_dataset)
		train_word_pair_dictionary,train_dictionary_list = multi_process_parser_tfidf(dependency_parser,train_dataset_dict,default_number_of_process,word_dict,weight)
	if tfidf_flag == "DISABLE_TFIDFG":
		train_word_pair_dictionary,train_dictionary_list = multi_process_parser(dependency_parser,train_dataset_dict,default_number_of_process)

	test_word_pair_dictionary,test_dictionary_list = multi_process_parser(dependency_parser,test_dataset_dict,default_number_of_process)

# if enable tfidf processing
	if tfidf_flag == "ENABLE_TFIDFG":
# save each document word pair into the file
		train_document_list = list()
		for index,document in train_word_pair_dictionary.items():
			new_train_file.write("%s\n" % document)
			train_document_list.append(document)
		with open('./OMDB/OMDB_train_dataset_word_pair_without_stopword_with_TFIDF.pkl','w') as f:
		    pickle.dump(train_document_list,f)
		print "The total number of document in the train dataset is ", len(train_word_pair_dictionary)

# save the dictionary into file
		train_dictionary = list(train_dictionary_list)
		train_dictionary_file.write("%s\n" % train_dictionary)
# save the dictionary into pickle
		with open('./OMDB/dictionary_word_pair_without_stopword_with_TFIDF.pkl','w') as f:
		    pickle.dump(train_dictionary,f)
		print "The total number of word pair in the dictionary is ", len(train_dictionary)

# save each document word pair into the file
		test_document_list = list()
		for index,document in test_word_pair_dictionary.items():
			new_test_file.write("%s\n" % document)
			test_document_list.append(document)
		with open('./OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.pkl','w') as f:
		    pickle.dump(test_document_list,f)
		print "The total number of document in the test dataset is ", len(test_word_pair_dictionary)

# if disable tfidf processing
	if tfidf_flag == "DISABLE_TFIDFG":
# save each document word pair into the file
		train_document_list = list()
		for index,document in train_word_pair_dictionary.items():
			new_train_file_without_tfidf.write("%s\n" % document)
			train_document_list.append(document)
		with open('./OMDB/OMDB_train_dataset_word_pair_without_stopword_without_TFIDF.pkl','w') as f:
		    pickle.dump(train_document_list,f)
		print "The total number of document in the train dataset is ", len(train_word_pair_dictionary)

# save the dictionary into file
		train_dictionary = list(train_dictionary_list)
		train_dictionary_file_without_tfidf.write("%s\n" % train_dictionary)
# save the dictionary into pickle
		with open('./OMDB/dictionary_word_pair_without_stopword_without_TFIDF.pkl','w') as f:
		    pickle.dump(train_dictionary,f)
		print "The total number of word pair in the dictionary is ", len(train_dictionary)

# save each document word pair into the file
		test_document_list = list()
		for index,document in test_word_pair_dictionary.items():
			new_test_file_without_tfidf.write("%s\n" % document)
			test_document_list.append(document)
		with open('./OMDB/OMDB_test_dataset_word_pair_without_stopword_without_TFIDF.pkl','w') as f:
		    pickle.dump(test_document_list,f)
		print "The total number of document in the test dataset is ", len(test_word_pair_dictionary)

###############################################################################################################################################
# single process parser
def single_process_parser_tfidf(dependency_parser,dataset,word_pair_dictionary,dictionary,start,end,word_dict,weight):

# Create English stop words
	stopset = stopwords.words('english')

	for doc_count in xrange(start,end):
		each_document = list()
# seperate each pargraph into sentences
		sentences = tokenize.sent_tokenize(dataset[doc_count])
		# iterate each sentence
		for sentence in sentences:
# parse each sentence
			result = dependency_parser.raw_parse(sentence)
			dep = result.next()
			output = list(dep.triples())
# if number of result > 0
			if len(output)>0:
				for elem in output:
# remove the word pair which contains the stop word
					if elem[0][0] not in stopset and elem[2][0] not in stopset:
# remove the word pair by TF-IDF score
						if check_tfidf(doc_count,word_dict,weight,elem[0][0],0.01) and check_tfidf(doc_count,word_dict,weight,elem[2][0],0.01):
							pair = list()
							pair.append(elem[0][0])
							pair.append(elem[2][0])
	# add word pair into each document collection
							each_document.append(pair)
							dictionary.append((elem[0][0],elem[2][0]))
		print "Doc:",doc_count	
		word_pair_dictionary.update({doc_count:each_document})

###############################################################################################################################################
# multi process parser
def multi_process_parser_tfidf(dependency_parser, dataset, number_of_process,word_dict,weight): 

	manager = Manager()
	word_pair_dictionary = manager.dict()
	dictionary = manager.list()
	segment = len(dataset)/number_of_process

	all_processes = [Process(target=single_process_parser_tfidf, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,x*segment, (x+1)*segment,word_dict,weight)) for x in xrange(0,number_of_process)]

	for p in all_processes:
		p.start()

	P_last=Process(target=single_process_parser_tfidf, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,number_of_process*segment, len(dataset),word_dict,weight))
	if number_of_process*segment < len(dataset):
		P_last.start()

	for p in all_processes:
		  p.join()

	if number_of_process*segment < len(dataset):	  
		P_last.join()
		
	word_pair_dictionary=OrderedDict(sorted(word_pair_dictionary.items(),key=lambda t:t[0]))

	return word_pair_dictionary,dictionary

###############################################################################################################################################
# single process parser
def single_process_parser(dependency_parser,dataset,word_pair_dictionary,dictionary,start,end):

# Create English stop words
	stopset = stopwords.words('english')

	for doc_count in xrange(start,end):
		each_document = list()
# seperate each pargraph into sentences
		sentences = tokenize.sent_tokenize(dataset[doc_count])
		# iterate each sentence
		for sentence in sentences:
# parse each sentence
			result = dependency_parser.raw_parse(sentence)
			dep = result.next()
			output = list(dep.triples())
# if number of result > 0
			if len(output)>0:
				for elem in output:
# remove the word pair which contains the stop word
					if elem[0][0] not in stopset and elem[2][0] not in stopset:
						pair = list()
						pair.append(elem[0][0])
						pair.append(elem[2][0])
# add word pair into each document collection
						each_document.append(pair)
						dictionary.append((elem[0][0],elem[2][0]))
		print "Doc:",doc_count	
		word_pair_dictionary.update({doc_count:each_document})

###############################################################################################################################################
# multi process parser
def multi_process_parser(dependency_parser, dataset, number_of_process): 

	manager = Manager()
	word_pair_dictionary = manager.dict()
	dictionary = manager.list()
	segment = len(dataset)/number_of_process

	all_processes = [Process(target=single_process_parser, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,x*segment, (x+1)*segment)) for x in xrange(0,number_of_process)]

	for p in all_processes:
		p.start()

	P_last=Process(target=single_process_parser, args=(dependency_parser, dataset, word_pair_dictionary,dictionary,number_of_process*segment, len(dataset)))
	if number_of_process*segment < len(dataset):
		P_last.start()

	for p in all_processes:
		  p.join()

	if number_of_process*segment < len(dataset):	  
		P_last.join()
		
	word_pair_dictionary=OrderedDict(sorted(word_pair_dictionary.items(),key=lambda t:t[0]))

	return word_pair_dictionary,dictionary

###############################################################################################################################################
# generate TF-IDF
def generate_tfidf(train_dataset):
# initialize logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	traindataset=[]

	with open(train_dataset) as infile:
		for line in infile:
			traindataset.append(line)

	count = 1

# remove the stop words
	print('Process stop words')
# restore into a set to improve the speed
	stop_word_set = set(stopwords.words('english'))
	texts_without_stop = [[word_with_stop for word_with_stop in document.lower().split() if word_with_stop not in stop_word_set] for document in traindataset]
	#print(texts_without_stop)

# remove the punctuation
	exclude = set(string.punctuation)
	texts_without_punctuation = [[word_with_punctuation for word_with_punctuation in text_without_stop if word_with_punctuation not in (exclude)] for text_without_stop in texts_without_stop]
	print('Process punctuation')
	#print(texts_without_punctuation)	

# remove the number
	print('Process digit')
	texts_without_digit = [[word_with_digit for word_with_digit in text_without_punctuation if not word_with_digit.isdigit()] for text_without_punctuation in texts_without_punctuation]
    #print(texts_without_digit)

# remove word tense
	print('Process tense and plural')
	texts_without_tense = []
	for text_without_digit in texts_without_digit:
		temp = []
		for word in text_without_digit:
			temp.append(PorterStemmer().stem(word.decode('utf-8')))
		texts_without_tense.append(temp)
	#print(texts_without_tense)

# prepare dataset for TF-IDF processing
	
	print('Enable TF-IDF process')
	temp_dataset_for_tfidf = []
	#temp_count=0
	for text_without_tense in texts_without_digit:
		if len(text_without_tense)!=0:
			temp_str = ''
			for i in xrange(0,len(text_without_tense)-1):
				temp_str += text_without_tense[i]
				temp_str += ' '
			temp_str += text_without_tense[len(text_without_tense)-1]
			#print(temp_count)
			temp_dataset_for_tfidf.append(temp_str)
			#temp_count+=1
	#print temp_dataset_for_tfidf

# TF-IDF process
	vectorizer = CountVectorizer(decode_error="replace")
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(temp_dataset_for_tfidf))
	word = vectorizer.get_feature_names()
	#pickle.dump(vectorizer.vocabulary_,open("feature7.pkl","wb"))

	word_dict = dict()
	word_count = 0
	for element in word:
		#print element,":",word_count
		#print('\n')
		word_dict.update({element:word_count})
		word_count += 1

	weight = tfidf.toarray()

	return word_dict,weight


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
# main function
if __name__ == '__main__' :

	train_dataset = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_train_dataset.txt"
	test_dataset = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_test_dataset.txt"

	new_test_path = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_test_dataset_word_pair_without_stopword_with_TFIDF.txt"
	new_test_path_without_tfidf = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_test_dataset_word_pair_without_stopword_without_TFIDF.txt"

	new_train_path = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_train_dataset_word_pair_without_stopword_with_TFIDF.txt"
	new_train_path_without_tfidf = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/OMDB_train_dataset_word_pair_without_stopword_without_TFIDF.txt"

	new_dictionary_path = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/dictionary_word_pair_without_stopword_with_TFIDF.txt"
	new_dictionary_path_without_tfidf = "/home/ziyizhao/lda/data/small_sentence_DataSet/OMDB/dictionary_word_pair_without_stopword_without_TFIDF.txt"

	generate_word_pairs(train_dataset,test_dataset,new_train_path,new_test_path,new_dictionary_path,new_train_path_without_tfidf,new_dictionary_path_without_tfidf,new_test_path_without_tfidf)

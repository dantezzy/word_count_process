#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 14:08:16
# @Author  : Ziyi Zhao
# @Version : 1.0

import os
import pickle

###############################################################################################################################################
# evaluation function
def generate(key,train_doc_ID_path,user_info_path):
# dictionary to store user and corresponding preference docs	
	movie_group_dict = dict()
# dictionary to store test ID and real citeUlike ID
	real_test_doc_ID = dict()
# dictionary to store train ID and real citeUlike ID
	real_train_doc_ID = dict()
# dictionary to store test ID and real citeUlike ID
	real_train_tag_list = list()
# dictionary to store train ID and real citeUlike ID
	real_test_tag_list = list()
# open user info file
	movie_group_info_file = open(user_info_path)
# iterator each user
	for movie_froup_info in movie_group_info_file:
# get user ID and corresponding preference
		movieID,groupID = movie_froup_info.split('::',1)
# save user ID and preference list into dictionary
		movie_group_dict.update({int(movieID):int(groupID)})
	# print len(movie_group_dict)

# load train doc list and corresponding doc ID
	train_doc_ID = open(train_doc_ID_path)
	train_doc_ID_count = 0
	for doc_ID in train_doc_ID:
		real_train_doc_ID.update({int(train_doc_ID_count):int(doc_ID)})
		train_doc_ID_count += 1
	# print len(real_train_doc_ID)

# load test doc list and corresponding doc ID
	test_doc_ID = open(test_doc_ID_path)
	test_doc_ID_count = 0
	for doc_ID in test_doc_ID:
		real_test_doc_ID.update({int(test_doc_ID_count):int(doc_ID)})
		test_doc_ID_count += 1
	# print len(real_test_doc_ID)

	for key,value in real_train_doc_ID.items():
		real_train_tag_list.append(movie_group_dict[value])

	for key,value in real_test_doc_ID.items():
		real_test_tag_list.append(movie_group_dict[value])

	# print real_train_tag_list
	# print real_test_tag_list

  	with open('./train_dataset_tag_20.pkl','w') as f:
	    pickle.dump(real_train_tag_list,f)

  	with open('./test_dataset_tag_20.pkl','w') as f:
	    pickle.dump(real_test_tag_list,f)


	print "Number of train tag: ",len(real_train_tag_list)
	print "Number of test tag: ",len(real_test_tag_list)
	


###############################################################################################################################################
# main function
if __name__ == '__main__':

	user_info_path = './OMDB/MovieKMeansCat20.dat'
	test_doc_ID_path = './OMDB/OMDB_test_dataset_ID.txt'
	train_doc_ID_path = './OMDB/OMDB_train_dataset_ID.txt'

	generate(test_doc_ID_path,train_doc_ID_path,user_info_path)
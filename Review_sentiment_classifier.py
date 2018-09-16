#
# Statistical machine learning
# Assignment : 1
# Question : 6
# Author : Sarvesh Patil
# ASU ID : 1213353386
#
###########################################

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
import sys
import glob
import re
import string
import random
import os


def remove_punctuations(txtstr):
	table = str.maketrans({key: None for key in string.punctuation})
	new_sen = txtstr.translate(table)
	return new_sen 

	
def remove_stop_words(txtstr):
	stop_words = set(stopwords.words('english'))  
	word_tokens = word_tokenize(txtstr) 
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
	filtered_sentence = [] 
	for w in word_tokens: 
	    if w not in stop_words: 
	        filtered_sentence.append(w.lower()) 
	return (sorted(filtered_sentence))

def build_vocab(sentences):
	words = []
	for sentence in sentences:
		w = remove_stop_words(remove_punctuations(sentence))
		words.extend(w)

	words = sorted(list(set(words)))
	return words


def bag_of_words(sentence, vocab):
    sentence_words = sorted(remove_stop_words(remove_punctuations(sentence)))
    bag = [0 for i in range(len(vocab))]
    for sw in sentence_words:
        for i,word in enumerate(vocab):
            if word == sw: 
                bag[i] += 1
    return (np.array(bag))


def splitdirs(files, dir1, dir2, ratio):
    shuffled = files[:]
    random.shuffle(shuffled)
    num = round(len(shuffled) * ratio)
    to_dir1, to_dir2 = shuffled[:num], shuffled[num:]
    for d in dir1, dir2:
        if not os.path.exists(d):
            os.mkdir(d)
    for file in to_dir1:
        os.symlink(file, os.path.join(dir1, os.path.basename(file)))
    for file in to_dir2:
        os.symlink(file, os.path.join(dir2, os.path.basename(file)))


def file_random_split(path, ratio):
	filenames = glob.glob(path)
	random.shuffle(filenames) 

	split = int(ratio * len(filenames))
	train_filenames = filenames[:split]
	test_filenames = filenames[split:]
	
	return train_filenames,test_filenames



# Splitting training/Testing and Reading files
# ratio = [0:1; 0:3; 0:5; 0:7; 0:8; 0:9]
ratio = 0.8
posrev_path = r'C:\Users\Sarvesh007\Documents\GRE\Applications\App_ASU\MS\3. SML\Assignment_1\pos\*.txt'
negrev_path = r'C:\Users\Sarvesh007\Documents\GRE\Applications\App_ASU\MS\3. SML\Assignment_1\neg\*.txt'


pos_reviews = []
neg_reviews = []
pos_train_files, pos_test_files = file_random_split(posrev_path, ratio)
neg_train_files, neg_test_files = file_random_split(negrev_path, ratio)

for name in pos_train_files:
	with open(name, encoding="utf8") as f:
		new_posreview = f.read()
		pos_reviews.append(new_posreview)

pos_vocab = build_vocab(pos_reviews) 
print("Unique words of positive reviews :",len(pos_vocab))


for name in neg_train_files:
	with open(name, encoding="utf8") as f:
		new_negreview = f.read()
		neg_reviews.append(new_negreview)

neg_vocab = build_vocab(neg_reviews)
print("Unique words of negative reviews :",len(neg_vocab))



complete_vocab = pos_vocab.copy()
for w in neg_vocab:
	if w not in complete_vocab:
		complete_vocab.append(w)

print("Unique words of all reviews :",len(complete_vocab))
# print(complete_vocab)



# Generating 2-D matrix of frequencies
pos_data_matrix = np.empty([len(complete_vocab),])
new_review = []  
for name in pos_train_files:
	with open(name, encoding="utf8") as f:
		new_review = f.read()
		review_inst = bag_of_words(new_review,complete_vocab)
		pos_data_matrix = np.column_stack((pos_data_matrix,review_inst))

neg_data_matrix = np.empty([len(complete_vocab),])
new_review = []   
for name in neg_train_files:
	with open(name, encoding="utf8") as f:
		new_review = f.read()
		review_inst = bag_of_words(new_review,complete_vocab)
		neg_data_matrix = np.column_stack((neg_data_matrix,review_inst))


pos_data_matrix = np.delete(pos_data_matrix,0,axis=1)
neg_data_matrix = np.delete(neg_data_matrix,0,axis=1)
data_matrix =  np.concatenate((pos_data_matrix,neg_data_matrix),axis=1)
# print(np.shape(pos_data_matrix))
# print(np.shape(neg_data_matrix))
# print(np.shape(data_matrix))

prob_pos = pos_data_matrix.sum(axis=1, dtype='int')
prob_neg = neg_data_matrix.sum(axis=1, dtype='int')
# print("Frequency of word in positive reviews :",prob_pos)
# print("Frequency of word in negative reviews :",prob_neg)


pos_inst_samples = np.size(pos_data_matrix,1)
neg_inst_samples = np.size(neg_data_matrix,1)
# print(pos_inst_samples)
# print(neg_inst_samples)


# Generating Triplet file
pos_triplet_mat = np.zeros((len(pos_vocab),pos_inst_samples))
neg_triplet_mat = np.zeros((len(neg_vocab),neg_inst_samples))
triplet_mat = np.chararray((len(complete_vocab),(pos_inst_samples+neg_inst_samples)), itemsize=50)

r=0
for w in complete_vocab:
	c=0
	while(c<(pos_inst_samples+neg_inst_samples)):
		v=data_matrix[r,c]
		triplet = "("+ str(w)+",Review #"+str(c+1)+","+str(int(v))+")"
		# print(triplet)
		triplet_mat[r,c] = triplet.encode('utf8')
		c=c+1
	r=r+1
# print(triplet_mat)
np.savetxt("data_matrix.txt",triplet_mat,delimiter=" ",fmt='%s')
print("Data matrix file created.")


P_Y_1 = len(pos_vocab)/(len(pos_vocab)+len(neg_vocab))
P_Y_2 = len(neg_vocab)/(len(pos_vocab)+len(neg_vocab))
P_X_Y_1 = prob_pos/(pos_inst_samples+len(pos_vocab))
P_X_Y_2 = prob_neg/(neg_inst_samples+len(neg_vocab))

# print("P(Y=1) =", P_Y_1)
# print("P(Y=2) =", P_Y_2)
# print("P(X|Y=1) =", P_X_Y_1)
# print("P(X|Y=2) =", P_X_Y_2)



# Testing
positive_revs = 0
negative_revs = 0
def get_Y_1_prob(review_inst):
	j=0
	for i in np.nditer(review_inst):
		if i>0:
			if P_X_Y_1[j]==0:
				P_X_Y_1[j]=(i+1)/(pos_inst_samples+len(complete_vocab))
		j=j+1


	test_P_X_Y_1 = review_inst*P_X_Y_1
	return ((np.prod(test_P_X_Y_1[test_P_X_Y_1!=0]))*P_Y_1)

def get_Y_2_prob(review_inst):
	j=0
	for i in np.nditer(review_inst):
		if i>0:
			if P_X_Y_2[j]==0:
				P_X_Y_2[j]=(i+1)/(neg_inst_samples+len(complete_vocab))
		j=j+1
		
	test_P_X_Y_2 = review_inst*P_X_Y_2
	return ((np.prod(test_P_X_Y_2[test_P_X_Y_2!=0]))*P_Y_2)


for name in pos_test_files:
	with open(name, encoding="utf8") as f:
		new_review = f.read()
		review_inst = bag_of_words(new_review,complete_vocab)
		review_inst[review_inst>0] = 1

		Y_1_prob = get_Y_1_prob(review_inst)
		Y_2_prob = get_Y_2_prob(review_inst)
		if Y_1_prob > Y_2_prob:
			positive_revs = positive_revs + 1
		elif Y_1_prob < Y_2_prob:
			negative_revs = negative_revs + 1

print("Positive Reviews accuracy:",(positive_revs*100)/(len(pos_test_files)))
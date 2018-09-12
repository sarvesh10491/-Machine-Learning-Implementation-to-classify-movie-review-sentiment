import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np
import re
import string


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


def bag_of_words(sentence, words):
    sentence_words = remove_stop_words(remove_punctuations(sentence))
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)


sentences = ["Machine learning is great","Natural Language Processing is a complex field","Natural Language Processing is used in machine learning"]
vocab = build_vocab(sentences)
print(vocab)

test_sentence = "Machine learning is complex but great for learning"
print(bag_of_words(test_sentence,vocab))

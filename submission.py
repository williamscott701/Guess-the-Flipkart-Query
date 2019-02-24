#!/usr/bin/env python
# coding: utf-8
import numpy as np
import copy
import nltk

from nltk import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

stop_words = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]



f = open('training.txt','r')
lines = int(f.readline())
text = []
labels = []
for i in range(lines):
    inp = str(f.readline())
    ind = inp.index('\t')
    labels.append(inp[ind+1:])
    text.append(inp[:ind])


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    words = str(data).split(' ')
    new_text = ""
    for w in words:
        if w.isdigit():
            new_text = new_text + " number"
        else:
            if w not in stop_words:
                new_text = new_text + " " + w
                pass
    return np.char.strip(new_text)


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")



def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    data = remove_apostrophe(data)
    return data


def processes_arr(text):
    preprocessed_text = []
    for t in text:
        preprocessed_text.append(preprocess(t))
    return preprocessed_text


def make_grams(data, n = 1):
    grammed_data = []
    for i in data:
        k = copy.deepcopy(str(i))
        for r in range(2,n+1):
            sixgrams = ngrams(str(i).split(), r)
            for grams in sixgrams:
                g = ""
                for p in grams:
                    g = g+p
                k = k+" "+g
        grammed_data.append(k)
    return grammed_data


def gen_grammed(text):
    preprocessed_text = processes_arr(text)
    grammed_data = make_grams(preprocessed_text)
    return grammed_data


grammed_data = gen_grammed(text)

vectorizer = CountVectorizer() #TfidfVectorizer()
vectorizer.fit(grammed_data)
vector = vectorizer.transform(grammed_data)
feature_vector = vector.toarray()


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(feature_vector, labels)

clf.score(feature_vector, labels)

c = clf.classes_

C = gen_grammed(c)

lines = int(input())
test_text = []
for i in range(lines):
    inp = str(input())
    test_text.append(inp)

test_grammed = gen_grammed(test_text)

vector = vectorizer.transform(test_grammed)
test_feature_vector = vector.toarray()

prob = clf.predict_proba(test_feature_vector)

lis2 = []
for i in test_grammed:
    lis1 = []
    for j in C:
        p = str(j).split()
        words = str(i).split()
        lens = len(p)
        count = 0
        for m in p:
            if m in words:
                count += 1
        lis1.append(count)
    lis2.append(lis1)

q = np.array(lis2)

k = prob

outs = q+k

vals = []
for i in outs:
    inde = np.argmax(i)
    vals.append(c[inde])

for i in vals:
    print(i[:len(i)-1])
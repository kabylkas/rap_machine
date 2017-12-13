import csv
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import math
import re
import pickle

class lyrics:
  def __init__(self, filename, dict_size = 8000):
    self.dict_size = dict_size
    self.corpus = []
    self.tokenized_corpus = []
    self.df = {}
    self.df["string"] = []
    self.d = self.loadLyrics(filename)
    self.df["as_numbers"] = self.tokenize_corpus()
    self.df["length"] = self.getLength()
    self.df = pd.DataFrame(data=self.df)

  #helper functions
  def process(self, sentence):
    count=0
    sen = sentence.lower()

    sen = sen.replace(".", " **period** ")
    sen = sen.replace("-", " **dash** ")
    sen = sen.replace(",", " **comma** ")
    sen = sen.replace("!", " **exclamation** ")
    sen = sen.replace("¡", "")
    sen = sen.replace("?", " **question** ")
    sen = sen.replace(":", " **colon** ")
    sen = sen.replace(";", " **semicolon** ")
    sen = sen.replace("\"", " **quote** ")
    sen = sen.replace("(", " **bracketsopen** ")
    sen = sen.replace(")", " **bracketsclose** ")
    sen = sen.replace("[", " **sqbracketsopen** ")
    sen = sen.replace("]", " **sqbracketsclose** ")
    sen = sen.replace("{", " **curbracketsopen** ")
    sen = sen.replace("}", " **curbracketsclose** ")
    sen = sen.replace("'", "")
    sen = sen.replace("\n", " **newline** ")

    sen_num = ""
    for word in sen.split():
      temp_word = word.replace(" ", "")

      if temp_word.isdigit():
        temp_word = "**number**"
      elif 'co/' in temp_word:
        temp_word = ""

      sen_num+=temp_word+" "

    return sen_num

  def loadLyrics(self, fileName):
    #read training set
    d = {}
    with open(fileName, 'r') as train_file:
      for line in train_file.readlines():
        processed_line = self.process(line)
        self.df["string"].append(processed_line)
        #count the frequencies of the words
        for word in processed_line.split():
          if word not in d:
            d[word] = 1
          else:
            d[word] += 1

    # One of the ways to improve training is to mark
    # chunk of least frequent wordsrds as "unknowns".
    # This will give better results on testing when
    # program encounters words that were not in the 
    # training set. The following several lines:
    # 1) sort the by frequency
    # 2) removes some chunk at the end
    l = []      
    for word, freq in d.items():
      l.append((word, freq))
    l_sorted_by_freq = sorted(l, key=lambda tup: tup[1], reverse = True)
    l_sorted_by_freq = l_sorted_by_freq[:self.dict_size-1]
    d = [word[0] for word in l_sorted_by_freq]
    d.append("**unknown**")
    d.append("'0")
    d.append("'1")
    return sorted(d)

  def tokenize_corpus(self):
    tokenized_corpus = []
    for sentence in self.df["string"]:
      token = []
      for word in sentence.split():
        if word in self.d:
          token.append(self.d.index(word))
        else:
          token.append(self.d.index("**unknown**"))
      
      tokenized_corpus.append(token)
          
    return tokenized_corpus

  def tokenize(self, sentence): 
    token = []
    sen = self.process(sentence)
    for word in sen.split():
      if word in self.d:
        token.append(self.d.index(word))
      else:
        token.append(self.d.index("**unknown**"))

    return token

  def tokens_to_sentence(self, tokens):
    sentence = ""
    for token in tokens:
      if token==1:
        break
      if token==0:
        continue

      word = self.d[token]
      sentence += word+" "

    sen = sentence
    sen = sen.replace("**period**", ".")
    sen = sen.replace("**dash**", "-")
    sen = sen.replace("**comma**", ",")
    sen = sen.replace("**exclamation**", "!")
    sen = sen.replace("**question**", "?")
    sen = sen.replace("**colon**", ":")
    sen = sen.replace("**semicolon**", ";")
    sen = sen.replace("**quote**", "\"")
    sen = sen.replace("**bracketsopen**", "(")
    sen = sen.replace("**bracketsclose**", ")")
    sen = sen.replace("**sqbracketsopen**", "[")
    sen = sen.replace("**sqbracketsclose**", "]")
    sen = sen.replace("**curbracketsopen**", "{")
    sen = sen.replace("**curbracketsclose**", "}")
    sen = sen.replace("**newline**", "\n")

    return sen

  def getLength(self):
    length_list = []
    for token in self.df["as_numbers"]:
      length_list.append(len(token))

    return length_list

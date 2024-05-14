import sys, os
from dotenv import load_dotenv 
load_dotenv()
sys.path.append(os.environ.get("FOLDER_PATH"))

import tensorflow as tf

import pandas as pd
import numpy as np

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

class Process:
    def __init__(self, vocab_size : int = 5000, sent_len : int  = 20):
        self.ps = PorterStemmer()
        self.vocab_size = vocab_size
        self.sent_len = sent_len
        
    def process_train_data(self, data_path : str):
        data = pd.read_csv(data_path)
        data = data.dropna()
        
        label = data['label']
        label = np.array(label)
        
        data = data.drop('label', axis = 1)
        
        dt = data.fillna(0)
        dt_cp = dt.copy()
        dt_cp.reset_index(inplace = True)
        
        corpus = []

        for i in tqdm(range(0, len(dt_cp))):
            review = re.sub('[^a-zA-Z]', ' ', str(dt_cp['title'][i]))
            review = review.lower()
            review = review.split()

            review = [self.ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)

        onehot_repr = [one_hot(words, self.vocab_size) for words in corpus]

        # Embedding Representation
        # making all sentences of same length
        embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = self.sent_len)

        dt_cp = np.array(embedded_docs)
            
        return dt_cp, label
    
    def process_test_data(self, data_path):
        # data = pd.read_csv(data_path)
        
        data = data_path
        
        dt = data.fillna(0)
        dt_cp = dt.copy()
        dt_cp.reset_index(inplace = True)
        
        corpus = []

        for i in tqdm(range(0, len(dt_cp))):
            review = re.sub('[^a-zA-Z]', ' ', str(dt_cp['text'][i]))
            review = review.lower()
            review = review.split()

            review = [self.ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)

        onehot_repr = [one_hot(words, self.vocab_size) for words in corpus]

        # Embedding Representation
        # making all sentences of same length
        embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = self.sent_len)

        dt_cp = np.array(embedded_docs)
            
        return dt_cp, dt

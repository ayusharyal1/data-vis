
# coding: utf-8

# In[2]:

import pandas as pd
from pandas import *
from numpy import *
import numpy as np
import os
from pandas import DataFrame
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import nltk.stem
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import string
from collections import Counter
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer


# In[3]:

class LemmaTokenizer(object):
    
    def __init__(self):
        
        self.wnl = WordNetLemmatizer()
        
    def __call__(self, doc):
        
        lowers = doc.lower()
        doc = lowers.translate(None,string.punctuation) ##remove the punctuation using the character
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# In[4]:

'''Computes eculidean distance between two normalized vectors v1 and v2'''
def dist_norm(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta= v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray()) #norm() calculates the Eculidean norm i.e. shortest distance"

def best_match(column_vectorizer, fmatrix,text_to_compare):
    n_samples = 100 # fmatrix.shape[0]
    best_dist = sys.maxint
    vect_to_compare = column_vectorizer.transform(text_to_compare)
    for i in range(0, n_samples):
        text_in_column = col_cat[i]
        if text_in_column == text_to_compare[0]:
            continue
        vector_for_column_text = fmatrix.getrow(i)
        #d = dist_raw(post_vec, new_post_vec)
        d = dist_norm(vector_for_column_text, vect_to_compare)
        print "===Category of app- %i with dist = %.2f: %s"%(i,d,text_in_column)
        if d < best_dist:
            best_dist = d
            best_i = i
    print "Best text in category is %i with dist = %.4f"%(best_i,best_dist)


# In[5]:

'''Use StemmedCountVectorizer to do:
1. lower casing the raw post in the preprossing step done in parent calss.
2. Extracting all individual words in the tokenization step in parent class.
3. Converting each word into its stemmed version.'''

import enchant
import re

class StemmedCountVectorizer(CountVectorizer):
    
    eng_dict = enchant.Dict("en_US")
    filter_option=['OnlyEng','AllLang'] 
    filter_by =""
    no_of_dialect = 0
    regex = r'[0-9_=*.-+]+' #Unicode strings are immune to regular expression.
    
    def setfilter_option(self, filter_by, count_dialect):
        self.filter_by = filter_by
        self.count_dialect = count_dialect
        self.eng_dict = enchant.Dict("en_US")
        self.regex = r'[0-9]+' #Unicode strings are immune to regular expression.

     
    ##overiding the analyzer of CountVectorizer
    def build_analyzer(self):
        #english_stemmer = nltk.stem.SnowballStemmer('english')
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc: self.analyzeddf(analyzer,doc)#(english_stemmer.stem(w.strip(regex)) for w in analyzer(doc) if w.isdigit() is False)               

    #For each document i.e. sentence/row this function is called.
    def analyzeddf(self,analyzer,doc):
        
        english_stemmer = nltk.stem.SnowballStemmer('english')
        tokens= analyzer(doc)
        
        if self.filter_by=='OnlyEng':
            tokens=[token for token in  analyzer(doc) if self.eng_dict.check(token) is True]
            
        if self.filter_by =='AllLang':
            tokens = [re.sub(self.regex,'',token) for token in tokens]
            tokens = [token.strip("_") for token in tokens if token !='']
            return (english_stemmer.stem(w) for w in tokens if w.isdigit() is False)
        
        return (english_stemmer.stem(w) for w in tokens if w.isdigit() is False)
        
def stat_vectorized_matrix(vectorized_array,vectorizer_type=None):
   
    #count the number of features generated,
    m, n = vectorized_array.shape
    count_non_zero_cells = np.count_nonzero(vectorized_array) #vectorized_array.nnz
    #print("vectorizer_type:"),type(vectorizer_type)
    #print("Sparse matrix shape: "), vectorized_array.shape
    #count the number of non-zero entries,
    #print("Sparsity(%%of non-zero values): %.6f %%" %(count_non_zero_cells/float(m*n) * 100))
    sparsity = (count_non_zero_cells/float(m*n) * 100)
    vectorizer_type = type(vectorizer_type)
    vector_shape = vectorized_array.shape
    record = [vectorizer_type,vector_shape, sparsity]
    return record


# In[6]:

import string

'''Returns a vectorized ND dataframe, vectorized ndarray, 
and an instance of the vectorizer Class used to transform.'''

def vectorize_column(dataframe,column_name,vectorizer=None):
    if vectorizer is None:
        print("No Vectorizer is explicitly specified. Using CountVectorizer as default one. ")
        column_vectorizer = CountVectorizer(min_df=1)
    else:
        column_vectorizer = vectorizer
    
    if column_name in dataframe.columns.values.tolist():
        column_df = dataframe[column_name] #select all the samples from the column passed as param.
        fmatrix = column_vectorizer.fit_transform(column_df) #convert text features to numerical vectors
        dataframe_f = pd.DataFrame(fmatrix.toarray(), columns=column_vectorizer.get_feature_names())
        print("Dataframe shape :("),dataframe_f.index.max()+1,",", dataframe_f.head(1).shape[1],")"
        
        return dataframe_f, fmatrix, column_vectorizer
    else:
        print("No column found")

#Custom tokenizer used by CountVectorizer.
def custom_tokenizer(doc):
    
    lowers = doc.lower() #lower-casing
    lowers = lowers.translate(string.punctuation) #remove punctuation
    tokens = nltk.word_tokenize(lowers)
    return tokens
       
'''Returns a vectorized (n_samples,n_features) dataframe, matrix and vectorizing object.
Parameters:
dataframe: pandas dataframe object
column_name: name of the column you want to vectorize (a column in above dataframe object)
vectorizer= Vectorizer Object, if none then CountVectorizer is used as default. 
n_samples: number of rows you want to vectorize
tf_idf: if True then TF-IDF matrix is returned, else only matrix of term frequency is return.

USAGE:
stem_vectorizer = StemmedCountVectorizer(encoding='utf-8',
                                         min_df =min_df,
                                         max_df =max_df,
                                         stop_words='english',
                                         analyzer='word',
                                         lowercase = lowercase)
                                         filter_by = ['OnlyEng','AllLang']
##set filterparameter to your vectorizer
filter_by=["OnlyEng", "AllLang"] #two options are available
count_dialect = True 
n_samples = n_samples #as u choose it.
stem_vectorizer.setfilter_option(filter_by[0],count_dialect)

dfx, matrixX, sv = vectorize_columnTfIdf(df, 'my_column',vectorizer=stem_vectorizer, n_samples=100, tf_idf=True)
'''

def vectorize_columnTfIdf(dataframe,column_name,vectorizer=None, n_samples=None, tf_idf=False):
    
    more_stopwords = []
    more_stopwords  = ['00','000','0000','0003','0004','0004','0005'] 
    more_stopwords += stopwords.words('english')
    more_stopwords += stopwords.words('japanese') 
    more_stopwords += stopwords.words('chinese')
    more_stopwords += stopwords.words('arabic')
    more_stopwords += stopwords.words('korean')
    more_stopwords += stopwords.words('russian')    
    
    if vectorizer is None:
        print("No Vectorizer is explicitly specified. Using CountVectorizer as default one. ")
        column_vectorizer = CountVectorizer(min_df=1, 
                                            max_df= 0.99, 
                                            stop_words=more_stopwords) #default vectorizer
    else:
        column_vectorizer = vectorizer
        column_vectorizer.stop_words = more_stopwords
    
    if column_name in dataframe.columns.values.tolist():
        
        if n_samples is None:
            column_df = dataframe[column_name] #select all the samples from the column passed as param. 
            print len(column_df)
        else:
            #column_df = dataframe[column_name].iloc[:n_samples] #select all the samples from the column passed as param.
            column_df = dataframe[column_name].iloc[:n_samples] 
        fmatrix = column_vectorizer.fit_transform(column_df)   
        
        if(tf_idf is True):
            
            tfidf_transformer  = TfidfTransformer(norm='l2').fit(fmatrix)
            tfidfNormalzedmatrix = tfidf_transformer.transform(fmatrix)
            fmatrix = tfidfNormalzedmatrix
            
        dataframe_f = pd.DataFrame(fmatrix.todense(), columns=column_vectorizer.get_feature_names())
        print("formed dataframe of size:("),dataframe_f.index.max()+1,",", dataframe_f.head(1).shape[1],")"
        
        return dataframe_f, fmatrix, column_vectorizer
    else:
        print("No column found")


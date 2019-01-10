#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:29:24 2019

@author: Ankit
"""
from keras.preprocessing.text import Tokenizer
import pandas as pd
import os
class Createdictionary:
    # creating diction and saving into dictionary folder
    def __init__(self,datapath='./data'):
        self.train_question=pd.read_csv(datapath+'/train_question.iob',header=None)
        self.test_question=pd.read_csv(datapath+'/test_question.iob',header=None)
        self.train_intent=pd.read_csv(datapath+'/train_intent.iob',header=None)
        self.test_intent=pd.read_csv(datapath+'/test_intent.iob',header=None)
        self.slot_train=pd.read_csv(datapath+'/slot_train.iob',header=None)
        self.slot_test=pd.read_csv(datapath+'/slot_test.iob',header=None)
    
    def build_chardictionary(self):
        #char_dictionary
        chartk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        chartk.fit_on_texts(self.train_question[0])
        chartk.fit_on_texts(self.test_question[0])
        chardict=chartk.word_index
        return chardict
    def create_save_char_seq(self,char_dict):
        
        
cr=Createdictionary()
print(cr.build_chardictionary())
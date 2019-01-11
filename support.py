#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:29:24 2019

@author: Ankit
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import numpy as np
from keras.utils import np_utils
class Support:
    # creating diction and saving into dictionary folder
    def __init__(self,datapath='./data'):
        self.train_question=pd.read_csv(datapath+'/train_question.iob',header=None)
        self.test_question=pd.read_csv(datapath+'/test_question.iob',header=None)
        self.train_intent=pd.read_csv(datapath+'/train_intent.iob',header=None)
        self.test_intent=pd.read_csv(datapath+'/test_intent.iob',header=None)
        self.slot_train=pd.read_csv(datapath+'/slot_train.iob',header=None)
        self.slot_test=pd.read_csv(datapath+'/slot_test.iob',header=None)
        self.MAXLENGTH=0
        for i in self.train_question[0]:
            if(len(i.split())>self.MAXLENGTH):
                self.MAXLENGTH=len(i.split())
        for i in self.test_question[0]:
            if(len(i.split())>self.MAXLENGTH):
                self.MAXLENGTH=len(i.split())
        self.MAXLENGTH=self.MAXLENGTH+1
    def build_dictionary(self):
        #char_dictionary
        wordtk = Tokenizer(num_words=None, oov_token='UNK')
        wordtk.fit_on_texts(self.train_question[0])
        wordtk.fit_on_texts(self.test_question[0])
        return wordtk
    
    def encode_sentence(self):
        wordtokenizer=self.build_dictionary()
        train_sentence=wordtokenizer.texts_to_sequences(self.train_question[0])
        train_sentence=pad_sequences(train_sentence,maxlen=self.MAXLENGTH,padding='post')
        test_sentence=wordtokenizer.texts_to_sequences(self.test_question[0])
        test_sentence=pad_sequences(test_sentence,maxlen=self.MAXLENGTH,padding='post')
        id2word={}
        for word,index in wordtokenizer.word_index.items():
            id2word[index]=word
        return np.asarray(train_sentence),np.asarray(test_sentence),self.MAXLENGTH,id2word
    
    
    def encode_outside_sentence(self,wordtokenizer,sentence):
        sentence=wordtokenizer.texts_to_sequences(sentence)
        sentence=pad_sequences(sentence,maxlen=self.MAXLENGTH,padding='post')
    def create_intentdict(self):
        label2id={}
        id2label={}
        index=0
        for label in self.train_intent[0]:
            if(label not in label2id):
                label2id[label]=index
                id2label[index]=label
                index=index+1
        for label in self.test_intent[0]:
            if(label not in label2id):
                label2id[label]=index
                id2label[index]=label
                index=index+1
        return label2id,id2label;
    def intent(self):
        label2id,id2label=self.create_intentdict()
        train_labels=[]
        for label in self.train_intent[0]:
            train_labels.append(label2id[label])
        test_labels=[]
        for label in self.test_intent[0]:
            test_labels.append(label2id[label])
        return np_utils.to_categorical(np.asarray(train_labels), num_classes=len(id2label)),np_utils.to_categorical(np.asarray(test_labels), num_classes=len(id2label))
    def create_slotdict(self):
        slot2id={}
        id2slot={}
        index=0
        for slots in self.slot_train[0]:
            slot=slots.split(' ')
            for sl in slot:
                if(sl not in slot2id):
                    slot2id[sl]=index
                    id2slot[index]=sl
                    index=index+1
        for slots in self.slot_test[0]:
            slot=slots.split(' ')
            for sl in slot:
                if(sl not in slot2id):
                    slot2id[sl]=index
                    id2slot[index]=sl
                    index=index+1
        return slot2id,id2slot
    def maximum_slot(self):
        max_slot_length=0
        for slots in self.slot_train[0]:
            if(len(slots.split())>max_slot_length):
                max_slot_length=len(slots.split())
        for slots in self.slot_test[0]:
            if(len(slots.split())>max_slot_length):
                max_slot_length=len(slots.split())
        return max_slot_length
    def slot(self):
        slot2id,id2slot=self.create_slotdict()
        maxium_slot_length=self.maximum_slot()
        train_slot=list()
        for slots in self.slot_train[0]:
            out=np.zeros((maxium_slot_length,len(slot2id)))
            out=list(out)
            slot=slots.split(' ')
            for st in range(len(slot)):
                out[st][slot2id[slot[st]]]=1
            train_slot.append(out)
        test_slot=list()
        for slots in self.slot_test[0]:
            out=np.zeros((maxium_slot_length,len(slot2id)))
            out=list(out)
            slot=slots.split(' ')
            for st in range(len(slot)):
                out[st][slot2id[slot[st]]]=1
            test_slot.append(out)
        return np.asarray(train_slot),np.asarray(test_slot),maxium_slot_length
    def word2vec(self):
        if(os.path.exists('./embedding/word2vec.npy')):
            print('Word2vec embedding present')
        else:
            wordtk=self.build_dictionary()
            embeddingsdict = {}
            f = open('./embedding/glove.6B.100d.txt')  
            return "exist"
            for line in f:
                values = line.split()
                word = values[0]
                value_list=[float(i) if i!='.' else 0 for i in values[1:]]    
                coefs = np.asarray(value_list, dtype='float32')
                embeddingsdict[word] = coefs
            f.close()
            word2vec = []  
            for word, i in wordtk.word_index.items(): 
                if(word in embeddingsdict):
                    word2vec.append(embeddingsdict[word])        
                else:
                    empty = np.zeros(100)
                    word2vec.append(empty) 
            empty = np.zeros(100)
            word2vec.append(empty) 
            word2vec = np.array(word2vec)
            np.save('./embedding/word2vec.npy',word2vec)
        return word2vec
            
        
        

#wordtokenizer=cr.build_dictionary()

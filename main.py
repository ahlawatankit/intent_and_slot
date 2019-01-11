#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:44:44 2019

@author: ankit
"""
from model import Customemodel
from support import Support
import numpy as np
from keras.models import Model,load_model
import keras
import os
from sklearn.metrics import accuracy_score,f1_score
import elmo
class Main:
    def __init__(self):
        sp=Support()
        self.train,self.test,self.maxlen,self.id2word=sp.encode_sentence()
        self.train_label,self.test_label=sp.intent()
        self.train_slot,self.test_slot,self.maximum_slot_length=sp.slot()
        if(os.path.exists('./embedding/word2vec.npy')):
             self.word2vec=np.load('./embedding/word2vec.npy')
        else:
             self.word2vec=sp.word2vec()
        self.slot2id,self.id2slot=sp.create_slotdict()
        self.intent2id,self.id2intent=sp.create_intentdict()
        
    def training(self):
        model_file="./model_weight/model2.h5"
        if(os.path.isfile(model_file)):
            model=load_model(model_file,custom_objects={'ELMoEmbedding':elmo.ELMoEmbedding})
            return model
        else:
            ml=Customemodel(self.word2vec,self.maxlen,len(self.slot2id),len(self.intent2id),self.id2word)
            model=ml.build_model()
            saveBestModel =keras.callbacks.ModelCheckpoint(filepath="./model_weight/model2.h5", monitor='intent_out_acc', verbose=2, save_best_only=True, mode='auto')
            history=model.fit(self.train,[self.train_label,self.train_slot],batch_size=64,epochs=50,verbose=1,callbacks=[saveBestModel])
            return model
    def validate(self,model):
        pred=model.predict(self.test)
        print("Intent Accuracy :",accuracy_score(np.argmax(self.test_label, axis=-1),np.argmax(pred[0], axis=-1)))
        print("Intent f1 :",f1_score(np.argmax(self.test_label, axis=-1),np.argmax(prd[0], axis=-1),average='micro'))
        
xyz=Main()
model=xyz.training()
xyz.validate(model)
        
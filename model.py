#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:04:48 2019

@author: ankit
"""
from elmo import ELMoEmbedding
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Lambda, concatenate, TimeDistributed, Bidirectional
from keras.layers import Input,Activation, Flatten, Dense, Reshape
from keras.layers import  Dropout,merge
from keras.models import Model
class Customemodel:
    def __init__(self,wordembedding,maxlength,totalslots,totalintents,id2word):
        self.wordembedding=wordembedding
        self.maxlength=maxlength
        self.totalslots=totalslots
        self.totalintents=totalintents
        self.id2word=id2word
    def build_model(self):
        words_input=Input(shape=(self.maxlength,),dtype='float32',name='words_input')
        word_embedding=Embedding(self.wordembedding.shape[0],self.wordembedding.shape[1],weights=[self.wordembedding],trainable=False)(words_input)
        elmoembedding = ELMoEmbedding(idx2word=self.id2word, output_mode="elmo", trainable=True)(words_input)
        embedding=concatenate([word_embedding,elmoembedding],axis=-1)
        conv1= Conv1D(30, 3,padding='same',activation='relu',kernel_initializer='Orthogonal')(embedding)
        conv2= Conv1D(40, 4,padding='same',activation='relu',kernel_initializer='Orthogonal')(embedding)
        conv3= Conv1D(50, 5,padding='same',activation='relu',kernel_initializer='Orthogonal')(embedding)
        lstm_out=Bidirectional(LSTM(128, return_state=True,return_sequences=True, go_backwards=True, kernel_initializer='Orthogonal'))(embedding)
        #print(lstm_out)
        conv_out=concatenate([conv1,conv2,conv3],axis=-1)
        slot_in=concatenate([conv_out,lstm_out[0]])
        flat=Flatten()(conv_out)
        intent_in=concatenate([flat,lstm_out[1],lstm_out[2],lstm_out[3],lstm_out[4]],axis=-1)
        
        slot_out=LSTM(128, return_sequences=True, go_backwards=False, kernel_initializer='Orthogonal')(slot_in)
        slot_out = Dense(units=self.totalslots, activation='softmax', kernel_initializer='Orthogonal',name = 'slot_out')(slot_out)
        
        
        intent_out = Dense(units=32, activation='relu', kernel_initializer='he_normal')(intent_in)
        intent_out=Dropout(0.3)(intent_out)
        intent_out = Dense(units=self.totalintents, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(intent_out)
        model = Model(inputs=words_input, outputs=[intent_out,slot_out])
        model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics=['accuracy'])
        return model



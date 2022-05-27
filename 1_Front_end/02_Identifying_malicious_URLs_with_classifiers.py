#-- coding:UTF-8 --
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"##Set the currently used GPU device as device #1
G = 1 # GPU number

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
# from keras.models import load_model
from tensorflow.keras.models import load_model
from evaluate import *
import time

##predefine Strat##
MAX_SEQUENCE_LENGTH = 200 # Fixed url length
EMBEDDING_DIM = 100 # 100d
BATCH_SIZE = 64 * G
token_path = 'model/tokenizer.pkl'
##predefine end##

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
tf.compat.v1.keras.backend.set_session(session)#Setting up a global session

time_start0=time.time() 

test = []
for line in open('data/Test_url.txt',encoding='utf-8', errors='ignore'):
    test.append(line.strip('\n'))
    
#load tokenizer
token_path = 'model/tokenizer.pkl'
tokenizer = pickle.load(open(token_path, 'rb'))

# sequences
sequences = tokenizer.texts_to_sequences(test)# Convert each line to a vector of word subscripts

# padding
test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

time_end0=time.time() 
print('totally cost0: ',time_end0-time_start0)

print('Shape of data tensor:', test.shape)

url=test
np.savez('data/save_url',url=url) 

npzfile=np.load('data/save_url.npz') 
url=npzfile['url']

##########****** BLSTM classifier *****#########
model=load_model("model/model-blstm.h5",custom_objects={'precision': precision,'recall':recall,'f1':f1})

time_start1=time.time() 
label=model.predict(url,verbose=1, batch_size=BATCH_SIZE)
time_end1=time.time() 
print('totally cost1: ',time_end1-time_start1)

dflabel=pd.DataFrame(label)
dflabel.columns = ['label']
dflabel.to_csv("Results/ex_label&model-blstm.csv",index=0)

##########****** BLSTM-CNN classifier *****#########
model=load_model("model/model-blstm-cnn.h5",custom_objects={'precision': precision,'recall':recall,'f1':f1})

time_start2=time.time() 
label=model.predict(url,verbose=1, batch_size=BATCH_SIZE)
time_end2=time.time() 
print('totally cost2: ',time_end2-time_start2)

dflabel=pd.DataFrame(label)
dflabel.columns = ['label']
dflabel.to_csv("Results/ex_label&model-blstm-cnn.csv",index=0)

##########****** CNN classifier *****#########
model=load_model("model/model-cnn.h5",custom_objects={'precision': precision,'recall':recall,'f1':f1})

time_start3=time.time() 
label=model.predict(url,verbose=1, batch_size=BATCH_SIZE)
time_end3=time.time() 
print('totally cost3: ',time_end3-time_start3)

dflabel=pd.DataFrame(label)
dflabel.columns = ['label']
dflabel.to_csv("Results/ex_label&model-cnn.csv",index=0)

##########****** CNN-BLSTM classifier *****#########
model=load_model("model/model-cnn-blstm.h5",custom_objects={'precision': precision,'recall':recall,'f1':f1})

time_start4=time.time() 
label=model.predict(url,verbose=1, batch_size=BATCH_SIZE)
time_end4=time.time() 
print('totally cost4: ',time_end4-time_start4)

dflabel=pd.DataFrame(label)
dflabel.columns = ['label']
dflabel.to_csv("Results/ex_label&model-cnn-blstm.csv",index=0)
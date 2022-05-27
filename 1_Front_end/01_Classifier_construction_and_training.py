#-- coding:UTF-8 --
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"##Set the currently used GPU device as device #1
G = 1 # GPU number

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization,Flatten
from keras.layers import Dense, LSTM, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard,CSVLogger
from keras.utils import plot_model
from evaluate import *
from history import *
import time

##predefine Strat##
MAX_SEQUENCE_LENGTH = 200 # Fixed url length
EMBEDDING_DIM = 100 # 100d
QA_EMBED_SIZE = 64
DROPOUT_RATE = 0.5#dropout
# EPOCHS = 15
EPOCHS = 7
BATCH_SIZE = 64 * G
VALIDATION_SPLIT = 0.1
token_path = 'model/tokenizer.pkl'
##predefine end##



gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
tf.compat.v1.keras.backend.set_session(session)#Setting up a global session


good = []
bad = []
for line in open('data/URL_goodqueries.txt',encoding='utf-8', errors='ignore'):
    good.append(line.strip('\n'))
for line in open('data/URL_badqueries.txt',encoding='utf-8', errors='ignore'):
    bad.append(line.strip('\n'))

data = []
labels = []
length = len(bad)
scale = 3
data.extend(good[:length * scale])##Undersampling
data.extend(bad)
labels.extend([1] * length * scale)
labels.extend([0] * length)


start0 = time.time() 

# tokenizer
texts = data
tokenizer = Tokenizer(char_level=True) # Word vectors #char_level: If True, each character will be treated as a token
tokenizer.fit_on_texts(texts)
pickle.dump(tokenizer, open(token_path, 'wb'))#Storing the tokenizer
word_index = tokenizer.word_index# A dict that holds the numbered ids of all words, starting from 1

# sequences
sequences = tokenizer.texts_to_sequences(data)#Convert each row of data to a vector form of the word index

# padding
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

end0 = time.time() 
print('totally cost0: ',end0-start0)

# Disrupt the order
index = [i for i in range(len(data))]
random.shuffle(index)
data = np.array(data)[index]
labels = np.array(labels)[index]


TRAIN_SPLIT = 0.9 # Large sample size, 10% as test set
TRAIN_SIZE = int(len(data) * TRAIN_SPLIT)

X_train, X_test = data[0:TRAIN_SIZE], data[TRAIN_SIZE:]
Y_train, Y_test = labels[0:TRAIN_SIZE], labels[TRAIN_SIZE:]


##########****** Building BLSTM classifier *****#########
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

#BLSTM Layer
model.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
model.add(Dense(QA_EMBED_SIZE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))

# model.summary()
plot_model(model, to_file='images/model-blstm.png',show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#monitor: the monitored data  #patience: the number of training rounds without progress. When the monitored data is no longer improving, training is stopped.
model_checkpoint = ModelCheckpoint('model/model-blstm.h5', save_best_only=True, save_weights_only=False)
# Training, and saving the entire model
# tensor_board = TensorBoard('log/tflog-blstm', write_graph=True, write_images=True)


#Configuring the training model
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, f1])

# history = LossHistory()
time_callback = TimeHistory() 
log_csv = CSVLogger(filename='log/log-blstm.csv')

# fit #Train the model with a given number of rounds (iterations on the dataset)
model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
          validation_split=VALIDATION_SPLIT, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, log_csv,time_callback])#Problem with tensor_board

print(time_callback.times)
# print(time_callback.totaltime)
# Plotting acc-loss curves
# history.loss_plot('epoch')

# Testing
model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)

##########****** Building BLSTM-CNN classifier *****#########
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
#BLSTM
model.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
# QA_EMBED_SIZE#hidden dimension,i.e., the output dimension, is 2*QA_EMBED_SIZE in this case
#CNN
model.add(Convolution1D(filters=128, kernel_size=3, padding='valid', activation='relu'))#Convolution1D One-dimensional convolution for text
# The size of each convolution kernel=kernel_size= 3
# model.add(BatchNormalization())#Batch specification layer (normalized)
# model.add(Activation('relu'))
model.add(MaxPooling1D(4))#4,4   198-4+1  /4  =49
model.add(Flatten())#49*128=6272
model.add(Dense(QA_EMBED_SIZE))#QA_EMBED_SIZE-Output Size
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))#1-Output Size
model.add(BatchNormalization())
model.add(Activation("sigmoid"))#Classification function-sigmoid
plot_model(model, to_file='images/model-blstm-cnn.png',show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('model/model-blstm-cnn.h5', save_best_only=True, save_weights_only=False)

model.compile(loss='binary_crossentropy',#binary_crossentropy--Cross-entropy loss function, generally used for binary classification
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, f1])

time_callback = TimeHistory() 
log_csv = CSVLogger(filename='log/log-blstm-cnn.csv')


model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
          validation_split=VALIDATION_SPLIT, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, log_csv,time_callback])
print(time_callback.times)

# Testing
model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)

##########****** Building CNN classifier *****#########
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
#CNN
model.add(Convolution1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(QA_EMBED_SIZE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
plot_model(model, to_file='images/model-cnn.png',show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('model/model-cnn.h5', save_best_only=True, save_weights_only=False)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, f1])

time_callback = TimeHistory() 
log_csv = CSVLogger(filename='log/log-cnn.csv')


model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
          validation_split=VALIDATION_SPLIT, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, log_csv,time_callback])
print(time_callback.times)

# Testing
model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)

##########****** Building CNN-BLSTM classifier *****#########
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

#CNN
model.add(Convolution1D(filters=128, kernel_size=3, padding='valid', activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(MaxPooling1D(4))
#BLSTM
model.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=False, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE)))
model.add(Dense(QA_EMBED_SIZE))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation("sigmoid"))
plot_model(model, to_file='images/model-cnn-blstm.png',show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('model/model-cnn-blstm.h5', save_best_only=True, save_weights_only=False)

model.compile(loss='binary_crossentropy',#binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy', precision, recall, f1])

time_callback = TimeHistory() 
log_csv = CSVLogger(filename='log/log-cnn-blstm.csv')

model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
          validation_split=VALIDATION_SPLIT, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, log_csv,time_callback])
print(time_callback.times)

# Testing
model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)

#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/12/06
# Code:CL191206
# Purpose:Simple LSTM for Sequence Classification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#-----------------------------------------------------
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(7) # fix random seed for reproducibility

# imdb_data = np.load('data/imdb.npz')
# print(imdb_data.files)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="E:/nlpTest/deepLearning/data/imdb.npz", num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
model.add(LSTM(100))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:%.2f%%" % (scores[1]*100))


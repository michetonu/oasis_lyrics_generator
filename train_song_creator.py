# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Input, Dropout, GRU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
import sys
import os
import pickle
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen, build_opener
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import textmining

# Load text
text = open('oasis_lyrics.txt').read()

#Replace confusing punctuation
text = text.replace(".", "\n")
text = text.replace("?", "?\n")
text = text.replace("!", "!\n")
text = text.replace(" ?", "?")
text = text.replace(" !", "!")
text = text.replace("\n\n", "\n")
text = text.replace('"', "")

text = re.sub(r'(?<=[a-z])(?=[A-Z])', '\n', text).lower()
text = re.sub(r'x(?<=[0-9])', '\n', text)
text = re.sub(r'\[.*?\]', '', text)
text = text.replace('chorus', "")

#Create character dictionary
chars = sorted(list(set(text)))
print('Number of characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Create character dictionary
chars = sorted(list(set(text)))
print('Number of characters:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Create sequences
max_len = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])
print('Number of sequences:', len(sentences))

X = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Building model...')

input_shape = (max_len, len(chars))
input_text = Input(shape=input_shape)
lstm_1 = LSTM(128, return_sequences=True)(input_text)
lstm_1 = Dropout(0.2)(lstm_1)
lstm_2 = GRU(128)(lstm_1)
output = Dense((len(chars)), activation='softmax')(lstm_2)

model = Model(inputs=input_text, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="weights-dante.hdf5", monitor="loss", verbose=1, save_best_only=True,
                               mode='min')

loadweights = False
if loadweights:
    model.load_weights("weights-oasis.hdf5")

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# train the model, output generated text after each iteration
num_iters = 600
for iteration in range(1, num_iters):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=[checkpointer])

    model.save('dante_model.h5')

    start_index = random.randint(0, len(text) - max_len - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + max_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()
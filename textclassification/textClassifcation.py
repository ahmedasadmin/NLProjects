
# Title:        Text classification {spam, ham}
# Author:       Ahmed Muhammed Abdelgaber
# Data:         Fri 8, 2024 
# email:         ahmedmuhammedza1998@gmail.com 
# code version: 0.0.0
#
# Copyright 2024 Ahmed Muhammed 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pandas as pd 

embedding_dim = 254

# preprocessed data stored in out.csv but the raw data in spam.csv
# this data has been preprocessed using regular 're' expression and not the original dataset 


df = pd.read_csv("out.csv")
max_len =  100

labels = {
    "spam": 0,
    "ham": 1
}

df["labels"] = df['Category'].map(labels)
# print(df["labels"])

str = []
for i in df["preMessage"]:
    str.append("".join(i))

tokenizer = Tokenizer()

tokens= tokenizer.fit_on_texts(str)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(str)

padded =  pad_sequences(sequences, padding='post')

training_padded = np.array(padded)
training_labels = np.array(df['labels'])


print(training_labels)
print(type(training_labels))

print(len(padded[0]))
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 25
history = model.fit(training_padded, training_labels, epochs=num_epochs)


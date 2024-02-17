# bag of words


import pandas as pd
import numpy as np


df = pd.read_csv("spam.csv")

print(df.head())

print(df.Category.value_counts())

df['spam'] = df.Category.apply(lambda x : 1 if x=='spam' else 0)

print(df)
from sklearn.model_selection import train_test_split

x_train, X_test, y_train, Y_test = train_test_split(df.Message, df.spam, test_size=0.15)
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()
x_train_cv  = v.fit_transform(x_train.values)
x_train_np = x_train_cv.toarray()

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_cv, y_train)

X_test_cv = v.transform(X_test)

from  sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
print(classification_report(Y_test, y_pred))


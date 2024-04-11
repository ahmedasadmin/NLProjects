import numpy as np # linear algebra
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import nltk
import os 
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

nltk.download('punkt')
#################################################################
def embedding():

    # its dimension number of words , size of vector
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, emb_dimension))#np.zeros(vocab_len, emb_dimension)
   
    # loop voer the words in word_index
    for word,i in tokenizer.word_index.items():
    
        # get embedding vector from embedding dictionary that has all words representations
        embedding_vector = w2vEmbedding_index.get(word)
        # check that embedding vector is not equal to None
        if embedding_vector is not None:
        # assign embedding vector to martix 
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
#################################################################
# import emoji
def clean_data (text):
    # clean from symbols 
    cleaned =re.sub(r'[^\w\s]',' ',text)
#     cleaned=''.join(list(map(lambda x: x.strip(), cleaned.split())))
#     print(1,cleaned.strip())
#     cleaned=re.sub(r'[^a-zA-Z0-9\s]', '', text) # Special Character Removal
#     print(2,cleaned)
#     cleaned = re.sub(r'\d+', '', text) # Handling Numbers
#     cleaned = emoji.get_emoji_regexp().sub(r'', cleaned) # Handling Emojis and Special Characters
#     print(3,cleaned)
    cleaned =re.sub(r'http\S+|www\S+|https\S+', ' ', cleaned) # Handling URLs and HTML tags
#     print(4,cleaned)
    # Handling Negations
#     cleaned = re.sub(r'\b(?:not|no|never)\b[\w\s]+', lambda match: match.group().replace(" ", "_"), cleaned)
    #Remove, punctuation
#     print(4,cleaned)
#     clean_text = text.translate(str.maketrans('', '', string.punctuation))
#     print(5,clean_text)
#     #Remove html commands
#     soup = BeautifulSoup(text, "html.parser")
#     print(6,soup)
#     plain_text = soup.get_text()
#     print(7,plain_text)
#     cleaned_text = text.lower()
#     print(8,cleaned_text)
    bad_chars = [';', ':', '!', "*", "  ","_"]
 
    # remove bad_chars
    for i in bad_chars:
        test_string = cleaned.replace(i, ' ')
        
    
    return test_string
##########################################################
def stemming(cleaned_data):
    
    tokens = nltk.word_tokenize(cleaned_data) # work with ara diff (test)

    stemmer = ISRIStemmer()
    for num,word in enumerate(tokens): 
        # there is some words stemming break it
        # test for this problem if stemming , token
        # بوظتها لانها بتعملها استيم غلط زى كلمه فالعين خلتها فالع
        tokens[num] = stemmer.stem(word)
      
    return tokens    

#####################################################################
def remove_stop_words(stmmed_data):
    
    stop_words = set(stopwords.words('arabic'))

    # Remove stopwords from the tokenized text
    filtered_tokens = [token for token in stmmed_data if token not in stop_words]

    preprocessed_data = ' '.join(filtered_tokens)

    return preprocessed_data

#######################################################################
def text_embadding(data):
    
    data=data.split()
    for word_num ,word_tweet in enumerate(data):
        if word_tweet in t_model.wv:
            most_similar = t_model.wv.most_similar(word_tweet , topn=1 )
            for term, score in most_similar:
                if term != word_tweet :
                    data[word_num]=score
        else :
            data[word_num]=0
#######################################################################
def index_word(data):
    # Create an instance of Tokenizer
    tokenizer = Tokenizer()

    # Fit the tokenizer on the text data
    tokenizer.fit_on_texts(data)

    return(len(tokenizer.word_index)),tokenizer
########################################################################
# Define a function to check if a string contains English characters
def contains_english(text):
    for char in text:
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            print(text)
            return True
        
    return False
###############################################################################
#Define a function to check if a string contains English characters
def contains_english(text):
    for char in text:
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                    return True
    return False
#################################################################################
def convert_to_squence(data,tokenizer):
    sequences = tokenizer.texts_to_sequences(data)
    if not sequences:
        return None  # Return None if the sequence is empty
    else : return sequences         
#################################################################################


filenames = os.listdir('/home/ahmed/coding/NLPprojects/TransformerClassification/dataset/tweets_review')
files=[]
for path in filenames:
    files.append(os.path.join('/home/ahmed/coding/NLPprojects/TransformerClassification/dataset/tweets_review',path))

print(files)

train_pos=pd.read_csv(files[4],sep='\t' , header=None)
train_neg=pd.read_csv(files[2],sep='\t' , header=None)
test_pos=pd.read_csv(files[3],sep='\t' , header=None)
test_neg=pd.read_csv(files[1],sep='\t' , header=None)

print(train_neg)
train = pd.concat([train_pos, train_neg], axis=0, ignore_index=True, 
                  keys=['positive', 'negative'], names=['sentiment', 'index'])
train.columns = ['target', 'tweets']

test = pd.concat([test_pos, test_neg], axis=0, ignore_index=True, 
                  keys=['positive', 'negative'], names=['sentiment', 'index'])
test.columns = ['target', 'tweets']

print(train)
print(train.tweets[3])
print(clean_data(train.tweets[3]))
print(remove_stop_words(stemming(clean_data(train.tweets[9]))))

## Apply data processing techniques
## on the test and train data

train.tweets = train.tweets.apply(clean_data)
train.tweets = train.tweets.apply(stemming)
train.tweets = train.tweets.apply(remove_stop_words)


test.tweets = test.tweets.apply(clean_data)
test.tweets = test.tweets.apply(stemming)
test.tweets = test.tweets.apply(remove_stop_words)


train.target=train.target.replace("pos",1)
train.target=train.target.replace("neg",0)

print(train.target)

test.target=test.target.replace("pos",1)
test.target=test.target.replace("neg",0)

X=train.tweets
y=train.target

vectorize=TfidfVectorizer(max_features=2000)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0,shuffle=True)

X_train=vectorize.fit_transform(X_train)
X_test=vectorize.transform(X_test)

#########  Using naive baise model for classification ############

# model_naive=MultinomialNB()
# model_naive.fit(X_train,y_train)
# y_pred_naive=model_naive.predict(X_test)
# acc_naive=accuracy_score(y_test,y_pred_naive)

# print(f"accuracy:{acc_naive}")

##############################################################
t_model = gensim.models.Word2Vec.load('full_grams_sg_100_wiki.mdl')



# embedding dictionary to save all words and its embedding vector in it
w2vEmbedding_index = {}
# the size of embedding vector
emb_dimension = 100
# the path of embedding file
embeddings_file= 'full_grams_sg_100_wiki.mdl'             #"../Embedding/tweets_cbow_300"
# load word2vec model with its pretrained embedding 
w2v_model = gensim.models.Word2Vec.load(embeddings_file)  #KeyedVectors.load(embeddings_file)
# loop over the words in word2vec vocab
for word in w2v_model.wv.index_to_key:
# assign the word as a key and its vector as a value in dictionary
    w2vEmbedding_index[word] = w2v_model.wv[word]



vocab_len,_=index_word(train.tweets)
# Filter the DataFrame to find rows containing English words
english_rows = train[train['tweets'].apply(contains_english)]

# Count the number of rows containing English words
num_english_rows = len(english_rows)

print("Number of rows containing English words:", num_english_rows)

# Filter the DataFrame to find rows containing English words
english_rows = train[train['tweets'].apply(contains_english)]

train = train.drop(english_rows.index)


# Filter the DataFrame to find rows containing English words
english_rows = train[train['tweets'].apply(contains_english)]


# Count the number of rows containing English words
num_english_rows = len(english_rows)

print("Number of rows containing English words:", num_english_rows)
vocab_len,tokenizer=index_word(train.tweets)
print(vocab_len)

sequences = convert_to_squence(X,tokenizer)
print(len(sequences))

embedding_matrix=embedding()
print(embedding_matrix[1])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)


X_train_final =convert_to_squence(X_train,tokenizer)
X_test_final =convert_to_squence(X_test,tokenizer)

X_train_final =pad_sequences(X_train_final, padding='post', maxlen=900)
X_test_final =pad_sequences(X_test_final, padding='post', maxlen=900)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_len+ 1, emb_dimension, input_length = 900,weights=[embedding_matrix],
                              trainable=False),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1,  activation='sigmoid')

])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model -
model.fit(X_train_final, y_train, validation_data=(X_test_final, y_test), epochs=10, batch_size=250)



X_train_of_test =convert_to_squence(test.tweets,tokenizer)
X_train_final_test =pad_sequences(X_train_of_test, padding='post', maxlen=900)

y_test_of_test=test.target

loss, accuracy = model.evaluate(X_train_final_test, y_test_of_test)
print('Test accuracy:', accuracy)

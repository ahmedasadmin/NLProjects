########################################################
# 'Company Review Classification using BERT'
########################################################
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import pyarabic.araby as araby
import tensorflow_datasets as tfds
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt # for visualizations purpose  
import seaborn as sn
import os
import pandas as pd 
import numpy as np 
import random
from collections import Counter
import string


seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

#######################################################
# Function: 'text_preprocessing'
# Description: this function performs 'tokenization','Stop Wrod Removing', and 'Punctuation cleaning'
# Return: 'string'
#######################################################
def text_preprocessing(text):
    processed_text = text
    tokens = araby.tokenize(text)
    stop_words = set(stopwords.words('arabic'))

    # Removing stop words
    tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(tokens)

    # Removing punctuation
    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))

    return processed_text
###############################################################
# Function: 'label_to_numbers'
# Description: convert labels to integers {0, 1, 2, 3, 4, 5, 6}
# Return: 'integer'
################################################################
def label_to_numbers(category):
   
    if category == 1:
        return 0
    elif category == 2:
        return 1
    elif category == 3:
        return 2
    elif category == 4:
        return 3
    elif category == 5:
        return 4
    elif category == 6:
        return 5
    elif category == 7:
        return 6
###############################################################
# Function: 'plot_confusion_matrix'
# Description: plotting heatmap for test step 
# Note: we disabled this function because we change number of labels and also work only in test 
# Return: 'Graph'
###############################################################
def plot_confusion_matrix(actual, pred):
    sn.heatmap(confusion_matrix(actual, pred), annot=True, yticklabels=['price', 'app', 'delivery', 'pros', 'general','service', 'positive'],
               xticklabels=['price', 'app', 'delivery', 'pros', 'general','service', 'positive'], fmt="g", cmap='Greens')
    plt.tight_layout()
    plt.show()
################################################################
# Function: 'get_model'
# Description: 
#            The model is already trained, and we only need to fine-tune it on our dataset and task. 
#            To achieve this, we need to add one trainable layer while freezing the others.
#            By doing so, we avoid training the entire model from scratch.
#            we need to set the output layer to produce the desired number of classes, which in this case is two.
# Return: 'tf.model'
#################################################################
def get_model(max_len = 0):
    
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids') # that will be the numric represention of the data will be the input for the model
    
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask') # that will be the attention mask that will tell the model which is import word to fouce on it 1 fouce 0 for padding token 
    
    # Obtain the last hidden state from the BERT model
    output = bert_model([input_ids,input_mask])[0] # the last hidden state 
    
    # Apply dropout regularization

    output = tf.keras.layers.Dropout(rate = 0.15)(output) 
    
    
    # Output layer with softmax activation for the desired number of classes (which is two in this case)

    y = tf.keras.layers.Dense(num_labels,activation='softmax')(output) # output layer
        
    # Create the model with input and output layers
    model = tf.keras.models.Model(inputs=[input_ids, input_mask], outputs=y)
    
    return model
##########################################################################
# Function: 'encoding'
# Description: encoding sentence into array of numbers to be used by model
# Return: 'Numpy array'
##########################################################################
def encoding(X_ds):
    
    return tokenizer.batch_encode_plus(X_ds,
                                       max_length =max_len, #set the maximum laength of sentenece
                                       add_special_tokens=True,# add the [CLS] & [SEP] tokens
                                       return_attention_mask=True,
                                       return_token_type_ids= False,
                                       padding = 'max_length', # add the [PAD] token
                                       truncation = True, # cut if the sentence exceced the max length
                                       return_tensors= 'tf'
                                      )

#######################################################################
# Function: 'predict_category'
# Description: take raw text, then preprocessing it and encoding it. 
#              finally use model to predict output
# Return: 'string'
#######################################################################
def predict_category(text,loaded_model = False):
    # Tokenize the input text
    text = text_preprocessing(text)
    encoded_text = encoding([text])

    # Get the model predictions
    if loaded_model:
        print("Using saved model....")
        predictions = saved_model.predict([encoded_text['input_ids'], encoded_text['attention_mask']])
    else:
        predictions = model.predict([encoded_text['input_ids'], encoded_text['attention_mask']])

    # Get the predicted category index
    predicted_index = np.argmax(predictions[0])

    # Convert the index to the actual category label
    categories = ['price','app','delivery','pros','general','service','postive']
    predicted_category = categories[predicted_index]

    return predicted_category

##########################################################################

# Reading the file using the pandas library
arabic_ds = pd.read_excel('/home/ahmed/coding/NLPprojects/textClassification UingBERT/main/dataset.xlsx')
arabic_ds = arabic_ds[arabic_ds['category'].notnull()]

# Counting the occurrences of each class
count = Counter(arabic_ds['category'])

# Printing the class counts and the total number of examples
print(count)
print(len(arabic_ds))

# Printing the first five examples in the dataset
print(arabic_ds.head())

# Split the dataset into X (examples) and Y (labels)
X = arabic_ds['text']
Y = arabic_ds['category']

# Check whether the three classes are in a balanced state or not
count = Counter(Y)  # Counter for the three classes
print(count)

__price, __app, __delivery, __pros, __general, __service, __positive= count[1.0], count[2.0], count[3.0], count[4.0], count[5.0], count[6.0], count[7.0]
print("price label number is ",__price)
# uncomment the following lines of code  for ploting count of labels 

plt.figure(figsize=(5, 3))
sn.barplot(x=[0, 1, 2, 3, 4, 5, 6], y=[__price, __app, __delivery, __pros, __general, __service, __positive])
plt.title('Number of Examples', fontsize=14)
plt.xlabel('Case Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(7), ['price','app','delivery','pros','general','service','postive'])
plt.show()



print("Before text preprocessing",arabic_ds['text'][1])
print("Before encoding",arabic_ds['category'][1])
arabic_ds['text'] = arabic_ds['text'].apply(str)
# apply the text preprocessing on the examples & encoding the labels
arabic_ds['text'] = arabic_ds['text'].apply(text_preprocessing)

arabic_ds['category'] = arabic_ds['category'].apply(label_to_numbers)



print("\n\nAfter text preprocessing",arabic_ds['text'][1])
print("After encoding",arabic_ds['category'][1])

# Split the data into three sets: train, test, and validation
# The split ratio is 70% for training, 15% for testing, and 15% for validation

x_train, x_test, y_train, y_test = train_test_split(arabic_ds['text'].to_list(), arabic_ds['category'].to_list(), test_size=0.3, random_state=seed, shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=seed)

# Print the size of each data set
print("Training data:", len(x_train))
print("Test data:", len(x_test))
print("Validation data:", len(x_val))

# Hyperparameters for the model, tokenizer, and BERT model
max_len = 64  # Maximum length for sentences to ensure consistent sentence length
BATCH_SIZE = 16  # Batch size for training
LR = 2e-5  # Learning rate (e.g., 2e-5, 3e-5, or 5e-5)
EPOCHS = 7  # Number of epochs for training
num_labels = 7  # Number of classes

loss_fn = 'categorical_crossentropy'  # Loss function for the model
metric = ['accuracy']  # Evaluation metric for the model

model_name = "aubmindlab/bert-base-arabertv2"  # Name of the BERT model to use
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize the tokenizer for the BERT model

bert_model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
# Initialize the BERT model for sequence classification with the specified number of labels
# creating the model 
model = get_model(max_len=max_len)
model.summary()


# opt = tf.keras.optimizers.Adam(learning_rate=LR) # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
# model.compile(optimizer=opt, loss=loss_fn, metrics=metric)

print('Starting to encode the training data')
x_train_encodings = encoding(x_train)
print('Finished !! \n')

print('Starting to encode the validation data')
x_val_encodings = encoding(x_val)
print('Finished !! \n')

print('Starting to encode the testing data')
x_test_encodings = encoding(x_test)
print('Finished !! \n')

# now is the time to train the model

# bert_history = model.fit(
#     x=[x_train_encodings['input_ids'], x_train_encodings['attention_mask']],
#     y=tf.keras.utils.to_categorical(y_train, num_classes=num_labels),
#     validation_data=([x_val_encodings['input_ids'], x_val_encodings['attention_mask']],
#                      tf.keras.utils.to_categorical(y_val, num_classes=num_labels)),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE
# )

# predicted = model.predict({'input_ids': x_test_encodings['input_ids'], 'attention_mask': x_test_encodings['attention_mask']})
# y_predicted = np.argmax(predicted, axis=1)

# print(classification_report(y_test, y_predicted))

# plot_confusion_matrix(y_test, y_predicted)


# model.save_weights('nlp_project_category_with_positive.h5') # save the trained weights
saved_model = get_model(max_len) # crearte new model
saved_model.load_weights('nlp_project_category_with_positive.h5') # load the weights


predicted_category = predict_category(" لا بأس بالبرنامج إنه جيد",loaded_model = True)

print("Predicted Category:", predicted_category)

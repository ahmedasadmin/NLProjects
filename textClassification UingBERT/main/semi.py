import numpy as np
import pandas as pd
import tarfile
import os
import gc
import random
import pyarabic.araby as araby
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
import os 
import string
from bs4 import BeautifulSoup
nltk.download('stopwords')
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
import gensim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import string
import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
import transformers as trf
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows',5000)
from tqdm.autonotebook import tqdm
import re
nltk.download('punkt')
import warnings
warnings.filterwarnings('ignore')
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
def bert_embedding(txt):
  idx = tokenizer.encode(txt) #creating tokens
  idx = np.array(idx)[None,:] #converting 2d array

  emb = bert(idx) #bert layer
  hidden = np.array(emb[0][0]) #batch output of last_hidden_state

  sent_emb = hidden.mean(0) # creating mean vector
  return sent_emb
#################################################################
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
################################################################################################


price = [',اسعارهم اغلا من المحلات ب كثير و بحطولك توصيل مجاني حكي فاضي التطبيق لا انصح به أصبح غالي جداً ,للأسف الواحد ينصدم بعد زيادة الاسعار و للاسف بعض المطاعم اصبحت معاملتهم رديئةنزلته ومفيش الخصم اللي قلته عليه علي أول 3طلبات اعمل ايه عشان يطبق,الضريبه عالية جداًاسعار أعلى بنسب تتجاوز ال٧٠٪ من المطاعم تجارب توصيل سيئة جدا مشاكل في دفع الفيزا تطبيق لا يستحق حجمه على الموبايل ,برنامج حرامى طلبت من زاكس أوردر ٣ ساندوتش دابل ب ٨٠ جنيه للواحد بعتولى ٣ سندوتشات سنجل ب ٦٥ جنيه الواحد وحاسبونى على أسعار الدابل يعنى ٤٥ جنيه فرق'
]
app = [' برنامج زباله مش عايز يحمل خالص عل 3اجهزه مش راضي يحمل نهءي ,التطبيق لا يغتح دائما بيعطيني لا يوجد اتصال بالشبكة..مع انه النت عندي تمام شو الحل, البرنامج بيظهر كل المطاعم و مغلقه مع انها بتكون فاتحه بقاله كده اكتر من شهر, برنامج توترز توصيل احلى من برنامجكم فاشلفاااااشل جدا البحث ما بعطي فائدة يتم كتابه إسم المطعم ومع ذلك البحث لا يظهر المطعم المطلوب يقوم بإعطاء مطاعم على مزاجه']

delivery=[' التطبيق صار سيء كثير، وتاخير في توصيل الطلبات، الطعام يصل متأخر وبارد جدا، من ١٥ دقيقة يصل الطلب بعد ساعة ونصف ولا يوجد تجاوب من قبل الادارة بشأن تأخير التوصيلأنا حزين جدا بسبب الذي حصل اليوم أوردران يرجعو في نفس اليوم ومكالمات كاذبةعدم احترام وقت توصيل الطلب وبعد الوقت المقدر لتوصيل الطعام 40 دقيقه وبعد انتهاء الوقت تفاجئنا بتمديد الوقت الى ساعتين وعند التواصل معاهم عن طريق الدردشه ماأخذنا منهم غير الكلام الحلو الي لابقدم ولا بأخر المشكله ... هذا ثاني مره اتعرض لنفس المشكله معاهم ولا في حل للشكوى وعندما طلبت التواصل مع الاداره افادوا بأنهم سيتواصلون بعد 60 دقيقه معاي ... الظاهر مشغولين من كثرة الشكاوي هههههه']
pros=['غير متوفر في جميع المتاجرخائب لا يضم مطاعم مهمة وأسعار مرتفعة']
general = ['تطبيق فاشل سيء جداً٨٠٪ من بغداد لا يمكن التوصيل اليه ادري شكو مسوين شركة ومتعبين نفسكم جان خليتو الدلفري مالتكم يتمشون يم الشركه بس شيء بخزي قعدت ساعه أستنى طلب واخر شيء كتبولي طلب إلغى ,!!مت عجبني لانه بن نادر أشوف شي محل في دفع نقد,اسوء تطبيق اول مابعرف ما اعمل تاني في جميع الاعلانات سيء جدا ,غدآ السبت الأحد الأربعاء والخميس'
]
service = [' التطبيق اصبح سيء جدا والتكلفة اصبحت مرتفعة كان التطبيق المفضل بنسبة لي وكنت اتصفحه واستخدمه يوميا لكن الان اصبح مخيب للامال, ازبل واحقر واوسخ برنامج وموظفين وساخه ونصب مش طبيعي, كان افضل تطبيق بعدين سرق من اختي 6 دينار و سكرو خساب اختي عشان ما تشتكي لاحد يحمل عشان ما يصير نفس اختي لحد الان ماردو الفلوس نصيحه مني, عروض كذب, من أسوء التطبيقات فعلاً تندم على تنزيله سيء للغايه وغير صادق.,تم الغاء طلبي بدون اي اشعار']



#loading the model
bert = trf.TFBertModel.from_pretrained('aubmindlab/bert-base-arabertv2')

#creating tokenizer
tokenizer = trf.BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2', max_length=2024)

# txt = "تم الغاء طلبي بدون اي اشعار"
# # ## tokenize
# # idx = tokenizer.encode(txt)
# # print("tokens:", tokenizer.convert_ids_to_tokens(idx))
# # print("ids   :", tokenizer.encode(txt))


    


arabic_ds = pd.read_csv('/home/ahmed/coding/NLPprojects/TransformerClassification/company_review/CompanyReviews.csv')

df = arabic_ds[arabic_ds['company'] == "talbat"]
df =df[df['label'] ==-1]
df.to_excel("output.xlsx") 
"""
print(df)
df.text = df['text'].apply(str)
neg = df

neg.text = neg.text.apply(clean_data)
neg.text = neg.text.apply(stemming)
neg.text = neg.text.apply(remove_stop_words)

price = clean_data(price[0])
app = clean_data(app[0])
pros = clean_data(pros[0])
delivery = clean_data(delivery[0])
general = clean_data(general[0])
service = clean_data(service[0])

price = stemming(price)
app = stemming(app)
pros = stemming(pros)
delivery = stemming(delivery)
general = stemming(general)
service = stemming(service)

price = remove_stop_words(price)
app = remove_stop_words(app)
pros = remove_stop_words(pros)
delivery = remove_stop_words(delivery)
general = remove_stop_words(general)
service = remove_stop_words(service)

neg_text = []

for text in neg['text']:
    neg_text.append(text[:500])

print(len(neg_text[0]))

sent_matrix = np.array([bert_embedding(text) for text in tqdm(neg_text)])
np.save('embeddings.npy',sent_matrix)
# sent_matrix = np.load('embeddings.npy')


price_emb = np.array([bert_embedding(t) for t in price])
app_emb = np.array([bert_embedding(t) for t in app])
delivery_emb = np.array([bert_embedding(t) for t in delivery])
pros_emb = np.array([bert_embedding(t) for t in pros])
general_emb = np.array([bert_embedding(t) for t in general])
service_emb = np.array([bert_embedding(t) for t in service])


cosine_score = pd.DataFrame(columns=['id','price','app','delivery','pros','general','service'])
cosine_score['id'] = range(len(sent_matrix))
cosine_score['price'] = cosine_similarity(sent_matrix,price_emb.mean(0)[None,:])
cosine_score['app'] = cosine_similarity(sent_matrix,app_emb.mean(0)[None,:])
cosine_score['delivery'] = cosine_similarity(sent_matrix,delivery_emb.mean(0)[None,:])
cosine_score['pros'] = cosine_similarity(sent_matrix,pros_emb.mean(0)[None,:])
cosine_score['general'] = cosine_similarity(sent_matrix,general_emb.mean(0)[None,:])
cosine_score['service'] = cosine_similarity(sent_matrix,service_emb.mean(0)[None,:])

print(cosine_score.head())

cosine_score['label'] = cosine_score[['price','app','delivery','pros','general','service']].idxmax(axis=1) #finding the column which has maximum value and retunrning the column name (this becomes the label for the text)
label_df = cosine_score


del cosine_score #deleting as we dont need this df anymore
gc.collect()
print(label_df.head())

label_df['text'] = neg['text']#Earlier we have defined num_sentences
print(label_df['text'])
label_df.drop(['price','app','delivery','pros','general','service'],axis=1,inplace =True) 
label_df = label_df[['id','label']]
label_df.loc[:, "text"] = neg['text']
df.loc[:, "category"] = list(label_df['label'])
df.to_csv('out.csv', index=False)
"""
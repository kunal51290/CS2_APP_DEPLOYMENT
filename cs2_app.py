import re
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer,PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')
import joblib
from joblib import load
import streamlit as st
from tensorflow.keras.models import load_model

st.title("IDENTIFICATION OF TWEETS RELATED TO DISASTER AND THOSE NOT RELATED TO DISASTER")

st.write("You can enter your tweet below and the model trained on labeled tweet data provided by Kaggle can predict with above 80% accuracy whether the tweet is related to disaster or not")

st.image(image="https://landerapp.com/blog/wp-content/uploads/2018/08/twitter_cover-1024x559.jpg", caption="SOURCE:https://landerapp.com/blog/6-promoting-tweets-tips/")

tweet = st.text_input(label="Enter the tweet you want to identify below:",value="", max_chars=None,placeholder="Enter your tweet here")

import tensorflow.keras.backend as K
class dot_attention(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(dot_attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)      
        super(dot_attention, self).build(input_shape)
 
    def call(self,x):
        e = K.dot(x,self.W)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def auc_score(y_true,y_pred):
    return tf.py_function(roc_auc_score,(y_true,y_pred),tf.double)

def tweet_classifier(tweet):
    
    print("Analysing tweet...")
    
    def remove_tags(html):
  
        # parse html content
        soup = BeautifulSoup(html, "html.parser")
  
        for data in soup(['style', 'script']):
        # Remove tags
            data.decompose()
  
        # return data by retrieving the tag content
        return ' '.join(soup.stripped_strings)

    def text_pp(text):
        #remove HTML tags
        txt = remove_tags(text)
    
        #split text into tokens
        tokens = txt.split()
    
        #define set of stopwords
        sw = set(stopwords.words('english'))
    
        for i,tok in enumerate(tokens):
            #remove URLs
            tokens[i] = re.sub(r'http?\:\/\/\S*','',tok)
            tokens[i] = re.sub(r'https?\:\/\/\S*','',tokens[i])
            #remove @tags because the person or entity tagged doesnt contribute much to determine the tweet class
            tokens[i] = re.sub(r'^@\S*','',tokens[i])
            #expand word contractions
            tokens[i] = re.sub(r"won\'t", "will not", tokens[i])
            tokens[i] = re.sub(r"can\'t", "can not", tokens[i])
            tokens[i] = re.sub(r"won\’t", "will not", tokens[i])
            tokens[i] = re.sub(r"can\’t", "can not", tokens[i])
            tokens[i] = re.sub(r"ain\’t", "is not", tokens[i])
            tokens[i] = re.sub(r"\’tis", "is", tokens[i])
            tokens[i] = re.sub(r"y\’all", "you all", tokens[i])
            tokens[i] = re.sub(r"n\'t", " not", tokens[i])
            tokens[i] = re.sub(r"\'re", " are", tokens[i])
            tokens[i] = re.sub(r"\'s", " is", tokens[i])
            tokens[i] = re.sub(r"\'d", " would", tokens[i])
            tokens[i] = re.sub(r"\'ll", " will", tokens[i])
            tokens[i] = re.sub(r"\'t", " not", tokens[i])
            tokens[i] = re.sub(r"\'ve", " have", tokens[i])
            tokens[i] = re.sub(r"\'m", " am", tokens[i])
            tokens[i] = re.sub(r"n\’t", " not", tokens[i])
            tokens[i] = re.sub(r"\’re", " are", tokens[i])
            tokens[i] = re.sub(r"\’s", " is", tokens[i])
            tokens[i] = re.sub(r"\’d", " would", tokens[i])
            tokens[i] = re.sub(r"\’ll", " will", tokens[i])
            tokens[i] = re.sub(r"\’t", " not", tokens[i])
            tokens[i] = re.sub(r"\’ve", " have", tokens[i])
            tokens[i] = re.sub(r"\’m", " am", tokens[i])
            #remove punctuations, special characters & digits
            tokens[i] = re.sub('[^A-Za-z\s]','',tokens[i])
            
        #remove empty string tokens and tokens of length less than 2 from token list
        #remove stopwords
        tokens = [tok for tok in tokens if tok != '' and len(tok) >= 2 and tok not in sw]
    
        #convert to lowercase
        for i,tok in enumerate(tokens):
            tokens[i] = tok.lower()
        
        #perform lemmatization of words
        for i,tok in enumerate(tokens):
            tokens[i] = WordNetLemmatizer().lemmatize(tok)
        
        tokens = [tok for tok in tokens if len(tok) > 2]
    
        #remove duplicate words
        tokens = sorted(set(tokens), key=tokens.index)
    
        pptxt = ' '.join(tokens)
    
        return pptxt
    
    pp_tweet = text_pp(tweet)
    
    #load the pretrained tokenizer
    tok = joblib.load("tok.sav")
    
    enc = tok.texts_to_sequences([pp_tweet])
    
    enc_p = pad_sequences(enc, maxlen=20, padding='post')
    
    model = load_model('CNN_BIDIR_DOT.h5',custom_objects={'dot_attention': dot_attention,'auc_score':auc_score})
    
    pred_prob = model.predict(enc_p)
    
    if pred_prob>=0.5:
        st.write("Tweet is related to disaster")
    else:
        st.write("Tweet is not related to disaster")

if st.button("Predict"):
       tweet_classifier(tweet)

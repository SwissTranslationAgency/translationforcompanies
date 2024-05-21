from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import matplotlib.pyplot as plt
import numpy as np
from langdetect import detect

import scipy
import json
import requests
from tqdm import tqdm_notebook as tqdm
from absl import logging

#tqdm().pandas()  # Enable tracking of progress in dataframe `apply` calls
#tweets = pd.read_csv('twcs.csv',encoding='utf-8')
#print(tweets.shape)
#tweets.head()
#first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
#QnR = pd.merge(first_inbound, tweets, left_on='tweet_id', 
#                                      right_on='in_response_to_tweet_id')
#        
#        # Filter to only outbound replies (from companies)
#QnR = QnR[QnR.inbound_y ^ True]
#print(f'Data shape: {QnR.shape}')
#QnR.head()
#        
#QnR = QnR[["author_id_x","created_at_x","text_x","author_id_y","created_at_y","text_y"]]
#QnR.head(5)
#        
#        #count = QnR.groupby("author_id_y")["text_x"].count()
#        #c = count[count>15000].plot(kind='barh',figsize=(10, 8), color='#619CFF', zorder=2, width=width,)
#        #c.set_ylabel('')
#        #plt.show()
#        
#amazonQnR = QnR[QnR["author_id_y"]=="AmazonHelp"]
#        
#amazonQnR = amazonQnR[:10000]
#        
#module_url = "/Users/berhandiclepolat/bookingberhan/tensorflow3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#embed = hub.Module(module_url)
#        
#        
#messages = amazonQnR
#custmessages = amazonQnR.loc[:,"text_x"]
#custmessages = custmessages.values.tolist()
#suppomessages = amazonQnR.loc[:,"text_y"]
#suppomessages = suppomessages.values.tolist()
#
#suppomessages = np.array(suppomessages)
#np.savez("suppomessages", suppomessages)


app = Flask(__name__)

#embed = hub.Module(module_url)
questionsdeembedded= []
questionsfrembedded= []
questionsitembedded= []
questionsenembedded= []

answersdeembedded= []
answersenembedded= []
answersfrembedded= []
answersitembedded= []

questionsdeembedded= np.load("questionsdeembedded.npy")
questionsenembedded= np.load("questionsdeembedded.npy")
questionsfrembedded= np.load("questionsdeembedded.npy")
questionsitembedded= np.load("questionsdeembedded.npy")

answersdeembedded = np.load("answersdeembedded.npy")
answersenembedded = np.load("answersenembedded.npy")
answersfrembedded = np.load("answersfrembedded.npy")
answersitembedded = np.load("answersitembedded.npy")

df = pd.read_excel("Q&A_DE.xlsx") 
answersde = pd.DataFrame(df, columns= ['answers']).values.tolist()
answersde = [item for sublist in answersde for item in sublist]

df_it = pd.read_excel("Q&A_ITA.xlsx") 
answersit = pd.DataFrame(df_it, columns= ['answers']).values.tolist()
answersit = [item for sublist in answersit for item in sublist]


df_fr = pd.read_excel("Q&A_FRA.xlsx") 
answersfr = pd.DataFrame(df_fr, columns= ['answers']).values.tolist()
answersfr = [item for sublist in answersfr for item in sublist]

df_en = pd.read_excel("Q&A_EN.xlsx") 
answersen = pd.DataFrame(df_en, columns= ['answers']).values.tolist()
answersen = [item for sublist in answersen for item in sublist]



@app.route('/')
def home():
	return render_template('index.html')


ko =[]
my_prediction= []
my_prediction2 = []

        
#cust_embeddings = np.load('custmessagesall.npy')
#suppo_embeddings = np.load('suppomessagesall.npy')

##embed_fn = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
#        select = request.form.get('comp_select')
#        if select in ["1"]:

@app.route('/',methods=['POST', 'GET'])
def predict(): 
    if request.method == 'POST':
            messages = request.form['message']
            messages = [messages]
            def listtostring(s):  
                str1 = " " 
                return (str1.join(s))
            messagesdet = listtostring(messages)
            if not messagesdet:
                lan = detect(messagesdet)
                print(lan)
                print(messages)   
                embed = hub.load("/Users/berhandiclepolat/bookingberhan/sentenceencoderlargemulti")
                message_embeddings=embed(messages)
                message_embeddings = message_embeddings.numpy()
                print(message_embeddings)

        
            
            if lan in ['de']:
                corr=np.inner(message_embeddings,questionsdeembedded)
                suppomessages = answersde
            if lan in ['en']:  
                corr=np.inner(message_embeddings,questionsenembedded)
                suppomessages = answersen
            if lan in ['it']:
                corr=np.inner(message_embeddings,questionsitembedded)
                suppomessages = answersit
            if lan in ['fr']:
                corr=np.inner(message_embeddings,questionsfrembedded)
                suppomessages = answersfr
                
            
    
            corrsortindex = np.argsort(corr)
            corrsortindex = corrsortindex.tolist()
            flat_list = []
            for sublist in corrsortindex:
                for item in sublist:
                    flat_list.append(item)
                    ko = flat_list[-5:]
                    ko.reverse()
            my_prediction = suppomessages[ko[0]]
            messages = (', '.join(messages))
            my_prediction = my_prediction.replace('"','')
        #    my_prediction2 = '\n' + "Second Prediction:" +suppomessages[ko[1]]
        #    my_prediction2 = my_prediction2.replace('"','')
            print (my_prediction)
            return render_template('autocomplete.html',prediction = my_prediction, message = messages)

#        data = [message]
#		vect = cv.transform(data).toarray()
#
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, debug=True)

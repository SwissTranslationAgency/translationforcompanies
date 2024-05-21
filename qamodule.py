import tensorflow_text
from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

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
module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

#module_url = "/Users/berhandiclepolat/bookingberhan/tensorflow5" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#embed = hub.load(module_url)
#embed = hub.Module(module_url)
custmessages= []
suppomessages = []
custmessages=np.load("custmessages.npz")
custmessages = custmessages['arr_0']
custmessages = custmessages.tolist()
suppomessages=np.load("suppomessages.npz")
suppomessages = suppomessages['arr_0']
suppomessages = suppomessages.tolist()

@app.route('/')
def home():
	return render_template('autocomplete.html')


#reviewbody = []
#reviewresponse = []
#with open('reviewskaggle.csv', encoding='utf-8', errors='ignore') as csv_file:
#    lines = csv_file.readlines()
#    for row in lines:
#        reviewbody.append(row)
#        
#with open('reviewskaggle.csv', encoding='utf-8', errors='ignore') as csv_file:
#    resp = csv_file.readlines()
#    for row in resp:
#        reviewresponse.append(row)
   

#with open('comcastfinalall.csv', encoding='utf-8', errors='ignore') as csv_file:
#    resp = csv_file.readlines()
#    for row in resp:
#        comcast2.append(row)  

ko =[]
my_prediction= []
my_prediction2 = []
        
cust_embeddings = np.load('custmessagesall.npy')
suppo_embeddings = np.load('suppomessagesall.npy')

#embed_fn = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")



@app.route('/',methods=['POST', 'GET'])
def predict(): 
    if request.method == 'POST':
        messages = request.form['message']
        messages = [messages]
        print(messages)   
        
        
#    module_url = ("/Users/berhandiclepolat/bookingberhan/tensorflow3")
#    embed = hub.Module(module_url)
    module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')
    questions = ["What is your age?"]
    responses = ["I am 20 years old.", "good morning"]
    response_contexts = ["I will be 21 next year.", "great day."]

    module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

    question_embeddings = module.signatures['question_encoder'](
            tf.constant(questions))
    response_embeddings = module.signatures['response_encoder'](
        input=tf.constant(responses),
        context=tf.constant(response_contexts))

    np.inner(question_embeddings['outputs'], response_embeddings['outputs'])

#        sendtens = data.toarray()
#    corr = np.inner(message_embeddings,cust_embeddings)
#    corrsortindex = np.argsort(corr)
#    corrsortindex = corrsortindex.tolist()
#    flat_list = []
#    for sublist in corrsortindex:
#        for item in sublist:
#            flat_list.append(item)
#            ko = flat_list[-5:]
#            ko.reverse()
#    my_prediction = "First Prediction:" + suppomessages[ko[0]]
#    messages = (', '.join(messages))
#    my_prediction = my_prediction.replace('"','')
#    my_prediction2 = '\n' + "Second Prediction:" +suppomessages[ko[1]]
#    my_prediction2 = my_prediction2.replace('"','')
#    print (my_prediction)
    return render_template('autocomplete.html',prediction = my_prediction+my_prediction2, message = messages)
    

#        data = [message]
#		vect = cv.transform(data).toarray()
#
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)


questions = ["What is your age?"]
responses = ["I am 20 years old.", "good morning"]
response_contexts = ["I will be 21 next year.", "great day."]

module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

question_embeddings = module.signatures['question_encoder'](
            tf.constant(questions))
response_embeddings = module.signatures['response_encoder'](
        input=tf.constant(responses),
        context=tf.constant(response_contexts))

np.inner(question_embeddings['outputs'], response_embeddings['outputs'])

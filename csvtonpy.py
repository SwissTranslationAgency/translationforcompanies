from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from tempfile import TemporaryFile

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
##module_url = "/Users/berhandiclepolat/bookingberhan/tensorflow3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#
#embed = hub.Module(module_url)
#
#messages = []

#with open('comcast5final.csv', encoding='utf-8', errors='ignore') as csv_file:
#    lines = csv_file.readlines()
#    for row in lines:
#        messages.append(row)


##messages = pd.read_csv('comcast.csv',encoding ='latin1',usecols = [3])


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
#amazonQnR = amazonQnR[:5000]
#        
#module_url = "/Users/berhandiclepolat/bookingberhan/tensorflow3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#embed = hub.Module(module_url)
        
#        
#messages = amazonQnR
#custmessages = amazonQnR.loc[:,"text_x"]
#custmessages = custmessages.values.tolist()
#custmessages1= custmessages[:1000]
#custmessages2= custmessages[1000:2000]
#custmessages3= custmessages[2000:3000]
#custmessages4= custmessages[3000:4000]
#custmessages5= custmessages[4000:5000]
#
#suppomessages = amazonQnR.loc[:,"text_y"]
#suppomessages = suppomessages.values.tolist()
#suppomessages1= suppomessages[:1000]
#suppomessages2= suppomessages[1000:2000]
#suppomessages3= suppomessages[2000:3000]
#suppomessages4= suppomessages[3000:4000]
#suppomessages5= suppomessages[4000:5000]

df = pd.read_excel("Q&A_DE.xlsx") 
questionsde = pd.DataFrame(df, columns= ['questions']).values.tolist()
questionsde = [item for sublist in questionsde for item in sublist]
answersde = pd.DataFrame(df, columns= ['answers']).values.tolist()
answersde = [item for sublist in answersde for item in sublist]

df_it = pd.read_excel("Q&A_ITA.xlsx") 
questionsit = pd.DataFrame(df_it, columns= ['questions']).values.tolist()
questionsit = [item for sublist in questionsit for item in sublist]
answersit = pd.DataFrame(df_it, columns= ['answers']).values.tolist()
answersit = [item for sublist in answersit for item in sublist]


df_fr = pd.read_excel("Q&A_FRA.xlsx") 
questionsfr = pd.DataFrame(df_fr, columns= ['questions']).values.tolist()
questionsfr = [item for sublist in questionsfr for item in sublist]
answersfr = pd.DataFrame(df_fr, columns= ['answers']).values.tolist()
answersfr = [item for sublist in answersfr for item in sublist]

df_en = pd.read_excel("Q&A_EN.xlsx") 
questionsen = pd.DataFrame(df_en, columns= ['questions']).values.tolist()
questionsen = [item for sublist in questionsen for item in sublist]
answersen = pd.DataFrame(df_en, columns= ['answers']).values.tolist()
answersen = [item for sublist in answersen for item in sublist]


#mylist = df['questions'].tolist()

#messages = suppomessages5
questionsdeembedded= []
answersdeembedded= []
questionsdeembedded=embed(questionsde)
answersdeembedded=embed(answersde)

questionsdeembedded=questionsdeembedded.numpy()
answersdeembedded = answersdeembedded.numpy()



questionsitembedded= []
answersitembedded= []
questionsitembedded=embed(questionsit)
answersitembedded=embed(answersit)

questionsitembedded=questionsitembedded.numpy()
answersitembedded = answersitembedded.numpy()



questionsfrembedded= []
answersfrembedded= []
questionsfrembedded=embed(questionsit)
answersfrembedded=embed(answersit)

questionsfrembedded=questionsfrembedded.numpy()
answersfrembedded = answersfrembedded.numpy()


questionsenembedded= []
answersenembedded= []
questionsenembedded=embed(questionsen)
answersenembedded=embed(answersen)

questionsenembedded=questionsenembedded.numpy()
answersenembedded = answersenembedded.numpy()

#outfile = TemporaryFile()
#np.save(outfile, questionsdeembedded)
#logging.set_verbosity(logging.ERROR)
#
#with tf.Session() as session:
#  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#  message_embeddings = session.run(embed(questionsde))
#
#  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#    print("Message: {}".format(messages[i]))
#    print("Embedding size: {}".format(len(message_embedding)))
#    message_embedding_snippet = ", ".join(
#        (str(x) for x in message_embedding[:3]))
#    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

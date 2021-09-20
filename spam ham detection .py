#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required packages.
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer 


# In[2]:


#loading the data set 
email_data = pd.read_csv("sms_raw_NB.csv")


# In[4]:


#looking at the data 
email_data


# In[5]:


#removing all the stop words from the data 
from nltk.corpus import stopwords


# In[6]:


stop_words = set(stopwords.words('english'))


# In[8]:


#cleaning the data
import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


# In[33]:


email_data.text = email_data.text.apply(cleaning_text)


# In[34]:


# removing empty rows
email_data = email_data.loc[email_data.text != " ",:]


# In[35]:


#looking at the data after cleaning 
email_data


# In[36]:


# splitting the data into train and test data sets 
from sklearn.model_selection import train_test_split

email_train, email_test = train_test_split(email_data, test_size = 0.2)


# In[37]:


# creation of matrix for the entire document 
def split_into_words(i):
    return [word for word in i.split(" ")]


# In[38]:


# Bag of Words
emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)


# In[39]:


#BOW for all messages
all_emails_matrix = emails_bow.transform(email_data.text)


# In[40]:


# BOW For training messages
train_emails_matrix = emails_bow.transform(email_train.text)


# In[41]:


# BOW For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)


# In[42]:


# Term weighting and normalizing on ALL emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)


# In[43]:


# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
train_tfidf.shape # (row, column)


# In[44]:


# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape #  (row, column)


# In[45]:


# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB


# In[46]:


# Multinomial Naive Bayes
classifier_mb = MB(alpha=1)
classifier_mb.fit(train_tfidf, email_train.type)


# In[47]:


# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.type)
accuracy_test_m


# In[48]:


from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, email_test.type) 

pd.crosstab(test_pred_m, email_test.type)


# In[49]:


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == email_train.type)
accuracy_train_m


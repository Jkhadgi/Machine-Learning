# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:01:24 2023

@author: jasmita khadgi
"""
#importing the libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

### Text preprocessing: Turn the text content into numerical feature vectors and compute frequencies (10%)#####

#######1. importing the datasets################################
from datasets import datasets_folder
from util import NaiveBayes, pre_processing, accuracy_score  #calling the custom function

data, mytest_data = datasets_folder("D:/Jasmita/Lecture/ML_DL/CA/1/data/data")


dataframe = pd.DataFrame(data, columns=["text","labels"])
mytest_dataframe = pd.DataFrame(mytest_data, columns=["text"])


#datalabelling #Converting the values in the 'labels' column to numerical values
label_map = {'myham': 0, 'myspam': 1}
dataframe['labels'] = dataframe['labels'].map(label_map)
print(dataframe)

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()

# fit the training data
training_data = count_vec.fit_transform(dataframe['text'].values)
print("Shape of the document-term matrix:", training_data.shape)

###frequency
freq = pd.DataFrame(training_data.toarray(), columns=count_vec.get_feature_names())
print(freq)
freq.to_csv("D:/Jasmita/Lecture/ML_DL/CA/frequency.csv", index=False)


# split into train and test (0.8:0.2)
X_train, X_test, y_train, y_test = train_test_split(training_data, dataframe['labels'], test_size=0.2, random_state=5)
print('Number of rows in the total set: {}'.format(dataframe.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# train the Naive Bayes classifier
naive_bayes = NaiveBayes()
naive_bayes.fit(X_train, y_train, alpha=0)

# Predict labels for the test set

y_pred = naive_bayes.predict(X_test)
print("Shape of y_true:", X_test.shape)
print("Shape of y_pred:", y_pred.shape)


# Llodaand preprocess the unseen data
mytest_dataframe = pd.DataFrame(mytest_data, columns=["text"])
X_unseen, _ = pre_processing(mytest_dataframe)

# Make predictions on the unseen data
test_preds = naive_bayes.predict(X_unseen)
#print('Unseen Data Predictions:', unseen_predictions)


if test_preds == 0:
    print('The email is "Ham"')
else:
    print('The email is "Spam"')
    
    

#CALCULATING CLASS PRIOR  AND LIKELIHOOD
for class_label, class_prob in naive_bayes.class_priors.items():
    print(f'Prior probability for class {class_label}: {class_prob}')

'''
for class_label, class_prob in naive_bayes.class_priors.items():
    print(f'Prior probability for class {class_label}: {class_prob}')
'''    



feature_names = count_vec.get_feature_names()


# Access likelihood probabilities

# storing the list of words 
words = count_vec.get_feature_names()


likelihood_df = pd.DataFrame(columns=feature_names)

# Iterating over different features
for word_idx, word in enumerate(words):
    likelihoods = []
    

    for outcome in np.unique(y_train):
        feat_val = count_vec.transform([word]).toarray()[0][word_idx]
        likelihood_key = f'{feat_val}_{outcome}'
        likelihood = naive_bayes.likelihoods[word_idx].get(likelihood_key, 0)
        likelihoods.append(likelihood)
    
   
    likelihood_df.loc[:, word] = likelihoods  # Using loc to ensure consistent column
    
likelihood_df.index = ["Ham", "Spam"]

print(likelihood_df)



#with using laplace smotthing
##########4. Implement Laplace smoothing ####################
alpha = 0.7
naive_bayes_lap = NaiveBayes()  # Create an instance of NaiveBayes
naive_bayes_lap.class_likelihood_cal(alpha)  # Call the method on the instance, passing alpha
naive_bayes_lap.fit(X_train, y_train, alpha)


#for unseen test data like before
test_preds = naive_bayes_lap.predict(X_unseen)
if test_preds == 0:
    print('The email is "Ham"')
else:
    print('The email is "Spam"')


# Access likelihood probabilities
#
words = count_vec.get_feature_names()
likelihood_df_lap = pd.DataFrame(columns=words)

# Iterating over different features
for word_idx, word in enumerate(words):
    likelihoods = []

    for outcome in np.unique(y_train):
        feat_val = count_vec.transform([word]).toarray()[0][word_idx]
        likelihood_key = f'{feat_val}_{outcome}'
        likelihood = naive_bayes_lap.likelihoods[word_idx].get(likelihood_key, 0)
        likelihoods.append(likelihood)
    
    likelihood_df_lap[word] = likelihoods
    
likelihood_df_lap.index = ["Ham", "Spam"]

print(likelihood_df_lap)









########################################Using sklearn#########################################

##2. Training a Naive Bayes classifier ###############
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB


#print (X.shape) #total
print(X_train.shape) #80%
print(X_test.shape) #20%



##########* Load testing data for a validation and Predict a class (
#since its already loaded 
mytest_dataframe = pd.DataFrame(mytest_data, columns=["text"])

# transform the testing data (preprocess)
test_data = count_vec.transform(mytest_dataframe['text'].values)
print("Shape of the document-term matrix:", test_data)
print('Number of rows in the test set: {}'.format(test_data.shape[0]))

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train , y_train)
naive_bayes.score(X_test, y_test) #to evaluate the test



test_preds = naive_bayes.predict(test_data)
if test_preds == 0:
    print('The email is "Ham"')
else:
    print('The email is "Spam"')

#Access prior probabilities
class_priors = naive_bayes.class_log_prior_
prior_probabilities = np.exp(class_priors)
print("Prior probabilities:", prior_probabilities)

# Access likelihood probabilities

feature_names = count_vec.get_feature_names()

# Get the log probabilities of each word being in a spam or ham email
log_probabilities = naive_bayes.feature_log_prob_

# Create a DataFrame to store the results
likelihood_df = pd.DataFrame(np.exp(log_probabilities), columns=feature_names)
likelihood_df.index = ["Ham", "Spam"]

# lilelihood of the each word in the dataframe
print(likelihood_df)
likelihood_df.to_csv("D:/Jasmita/Lecture/ML_DL/CA/likelihood.csv", index=True)



##########4. Implement Laplace smoothing (10%)#####################
alpha = 0.7
naive_bayes_lap = MultinomialNB(alpha = alpha)
naive_bayes_lap.fit(X_train , y_train)
naive_bayes_lap.score(X_test, y_test) #to evaluate the test

#for unseen test data like before
test_preds = naive_bayes_lap.predict(test_data)

test_preds = naive_bayes_lap.predict(test_data)
if test_preds == 0:
    print('The email is "Ham"')
else:
    print('The email is "Spam"')


#Access prior probabilities
class_priors_lap = naive_bayes_lap.class_log_prior_
prior_probabilities_lap = np.exp(class_priors_lap)
print("Prior probabilities:", prior_probabilities_lap)

# access likelihood probabilities
# 
feature_names = count_vec.get_feature_names()

# log probabilities of each word being in a spam or ham email
log_probabilities_lap = naive_bayes_lap.feature_log_prob_

# dataframe to store the results
likelihood_df_lap = pd.DataFrame(np.exp(log_probabilities_lap), columns=feature_names)
likelihood_df_lap.index = ["Ham", "Spam"]

# lilelihood of the each word in the dataframe
print(likelihood_df_lap)
likelihood_df.to_csv("D:/Jasmita/Lecture/ML_DL/CA/likelihood_lap.csv", index=True)





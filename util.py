# -*- coding: utf-8 -*-
"""
@author: Jasmita Khadgi
"""

'''
this util.py file contains the all fucntions necessary to learn custom model 
'''

###########################################################################

######################Naives Bayes###############################

import numpy as np
import pandas as pd


def accuracy_score(y_true, y_pred):
    return round(float(sum(y_pred == y_true)) / float(len(y_true)) * 100, 2)


class NaiveBayes:
    def __init__(self):
        self.features = list()
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.train_size = int
        self.num_feats = 0  #zero intialization 

    def fit(self, X, y, alpha):
        self.num_feats = X.shape[1]
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]

        for feature in range(self.num_feats):
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}

            feat_vals = np.unique(self.X_train[:, feature].toarray())
            for feat_val in feat_vals:
                self.pred_priors[feature].update({str(feat_val): 0})

                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature].update({str(feat_val) + '_' + str(outcome): 0})
                    self.class_priors.update({outcome: 0})

        self.class_prior_cal()
        
        self.predictor_calc_prior()
        self.class_likelihood_cal(alpha)  # Call class_likelihood_cal with alpha

                    
    def class_likelihood_cal(self, alpha):
        for feature in range(self.num_feats):
            self.likelihoods[feature] = {}
            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.X_train[self.y_train == outcome][:, feature].toarray().flatten()
                unique_feat_vals, feat_val_counts = np.unique(feat_likelihood, return_counts=True)
                total_count = feat_val_counts.sum()

                for feat_val, count in zip(unique_feat_vals, feat_val_counts):
                    likelihood = (count + alpha) / (total_count + alpha * len(unique_feat_vals))
                    self.likelihoods[feature][str(feat_val) + '_' + str(outcome)] = likelihood
   
    def class_prior_cal(self):
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size    
    

    def predictor_calc_prior(self):
        for feature in self.features:
            feat_vals = self.X_train[feature].value_counts().to_dict()

            for feat_val, count in feat_vals.items():
                self.pred_priors[feature][feat_val] = count / self.train_size               


    def predict(self, X):
        results = []
        X = np.array(X)

        for query in X.reshape(-1, 1):
            probs_outcome = {}

            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence = 1

                for feat, feat_val in zip(self.features, query):
                    likelihood *= self.likelihoods[feat][feat_val + '_' + outcome]
                    evidence *= self.pred_priors[feat][feat_val]

                posterior = (likelihood * prior) / evidence
                probs_outcome[outcome] = posterior

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)
    


def pre_processing(df):
    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]
    return X, y







########################Irir datasets ####################################################





from sklearn import datasets

def iris_dataset():
    iris = datasets.load_iris()
    x = iris.data  #petal width, petal length, sepal width, septal length...
    y = iris.target  #species name 
    return x, y


'''
from sklearn.model_selection import train_test_split
def iris_dataset():
    # Load the iris dataset
    iris = datasets.load_iris()

    #For the binary classification, use first two features and only [0,1] labels in the dataset
    X = iris.data[iris.target < 2, :2]
    y = iris.target[iris.target < 2]

    # Splitting the dataset
    tr_x, val_x, tr_y, val_y = train_test_split(X, y, test_size=0.2)

    return tr_x, tr_y, val_x, val_y
'''

################################################for logistic regressions and Stochastic Gradient Ascent########################################
import numpy as np
class logistic:
    def __init__(self, max_iter, eta):
        self.max_iter = max_iter
        self.eta = eta
        self.w = None
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def train_GA(self, X, y):   #for Greadient Ascent
        self.w = np.zeros(X.shape[1])
        for i in range(self.max_iter):
            y_pred = self.sigmoid(np.dot(X, self.w))
            error = y - y_pred
            grad = np.dot(X.T, error)
            self.w += self.eta * grad   #gradient ASCENT
            
    def train_SGA(self, X, y): # for stocastic gradient algorithm
        self.w = np.zeros(X.shape[1])
        m = X.shape[0]  # Number of training examples

        for i in range(self.max_iter):
            for j in range(m):
                rand_idx = np.random.randint(0, m)  # Randomly select an index
                X_j = X[rand_idx]
                y_j = y[rand_idx]
                y_pred = self.sigmoid(np.dot(X_j, self.w))
                error = y_j - y_pred
                grad = error * X_j
                self.w += self.eta * grad
        
    def train_SGA_minibatches(self, X, y, batch_size):
        self.w = np.zeros(X.shape[1])
        m = X.shape[0]  # Number of training examples
        num_batches = m // batch_size  # Compute the number of batches

        for i in range(self.max_iter):
            for batch in range(num_batches):
                rand_indices = np.random.choice(m, size=batch_size, replace=False)  # Randomly select indices for the mini-batch
                X_batch = X[rand_indices]
                y_batch = y[rand_indices]
                y_pred = self.sigmoid(np.dot(X_batch, self.w))
                error = y_batch - y_pred
                grad = np.dot(X_batch.T, error)
                self.w += self.eta * grad

    
    def predict(self, X):
        y_pred = np.round(self.sigmoid(np.dot(X, self.w)))
        return y_pred
    
def computeClassificationAcc(y_true, y_pred):
    return np.mean(y_true == y_pred)






########################################for Regularized  Logistic########################################:
    
class RegLogistic:
    def __init__(self, max_iter, eta, lamda):
        self.max_iter = max_iter
        self.eta = eta
        self.lamda = lamda
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def train_reg_SGA(self, X, y): # regularized logistic regression model
        self.w = np.zeros(X.shape[1])   #self.w weight or coefficient vector associated with each feature
        m = X.shape[0]  # Number of training examples

        for i in range(self.max_iter):
            for j in range(m):
                rand_idx = np.random.randint(0, m)  
                X_j = X[rand_idx]
                y_j = y[rand_idx]

                y_pred = self.sigmoid(np.dot(X_j, self.w))
                error = y_j - y_pred
                grad = error * X_j + self.lamda * self.w  # Regularization term added
                self.w += self.eta * grad

    def predict(self, X):
        y_pred = np.round(self.sigmoid(np.dot(X, self.w)))
        return y_pred


#regularized logistic using mni-batches

class RegLogistic_minibatch:
    def __init__(self, max_iter, eta, lamda):
        self.max_iter = max_iter
        self.eta = eta
        self.lamda = lamda
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train_reg_SGA(self, X, y, batch_size):
        self.w = np.zeros(X.shape[1])   # self.w weight or coefficient vector associated with each feature
        m = X.shape[0]  # Number of training examples
        num_batches = int(np.ceil(m / batch_size))

        for i in range(self.max_iter):
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = min(start_idx + batch_size, m)
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                y_pred = self.sigmoid(np.dot(X_batch, self.w))
                error = y_batch - y_pred
                grad = np.mean(error[:, np.newaxis] * X_batch, axis=0) + self.lamda * self.w  # Regularization term added
                self.w += self.eta * grad
                
    def predict(self, X):
          y_pred = np.round(self.sigmoid(np.dot(X, self.w)))
          return y_pred
            
            
#############################Perceptron##########################################################

from sklearn.metrics import accuracy_score

class perceptron:
    def __init__(self, eta, max_iter, threshold, random_state=None):
        self.eta = eta
        self.max_iter = max_iter
        self.threshold = threshold
        self.random_state = random_state
        self.w = None
        self.errors_ = []

    def train(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def decision_function(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        decision_values = self.decision_function(X)
        return np.where(decision_values >= self.threshold, 1, -1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


'''
def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
'''
            
        
        
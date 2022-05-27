#-- coding:UTF-8 --
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"##Set the currently used GPU device as device #1
G = 1 # GPU number

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import urllib.parse
import matplotlib.pyplot as plt
import datetime
from gensim.models import Word2Vec
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle 
import numpy as np
import pandas as pd


good = []
bad = []
for line in open('data/URL_goodqueries.txt',encoding='utf-8', errors='ignore'):
    if len(line)<3:
        continue
    else:
        good.append(line.strip('\n'))
for line in open('data/URL_badqueries.txt',encoding='utf-8', errors='ignore'):
    if len(line)<3:
        continue
    else:
        bad.append(line.strip('\n'))

data = []
labels = []

length = len(bad)
scale = 3
data.extend(good[:length * scale])
data.extend(bad)
labels.extend([1] * length * scale)
labels.extend([0] * length)

start0 = datetime.datetime.now()

model = Word2Vec(sentences=data, 
                 vector_size=200,
                 window=5, 
                 min_count=1,##min_count: enables to truncate the dictionary. 
                 workers=4)#vocab=175, vector_size=100, alpha=0.025
word2vec_model = model
word2vec_model.save("model/w2v/word2vec.model")
def get_contentVector(cutWords, word2vec_model):
    vector_list = [word2vec_model.wv[k] for k in cutWords if k in word2vec_model.wv]
    contentVector = np.array(vector_list,dtype=list).mean(axis=0)
    return contentVector.tolist()

contentVector_list = []
for i in range(len(data)):
    cutWords = data[i]
    contentVector_list.append(get_contentVector(cutWords, word2vec_model))
X = np.array(contentVector_list)

end0 = datetime.datetime.now()
print('totally time0 is', end0 - start0)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=13) #splitting data
badCount = len(bad)
validCount = len(good)

#########################XGboost####################
start1 = datetime.datetime.now()
clf_xgb = XGBClassifier(scale_pos_weight=1/3,##scale_pos_weight can be set to the number of negative samples/number of positive samples in the data
                        learning_rate=0.1,#specifies the learning rate. The default value is 0.3.
#                         colsample_bytree = 0.8,## is used to control the percentage of columns sampled at random per tree
                        subsample = 0.5,## The ratio of random sampling. Typical value: 0.5-1
                        objective='binary:logistic', # Binary logistic regression, output is probability
                        n_estimators=1000,#Number of decision trees #### i.e. maximum number of iterations #######epoch
#                         reg_alpha = 0.3,#The L1 regularization term for the weights. 
                        max_depth=10,#A number between 3 and 10 is commonly used. This value is the maximum depth of the tree.
                        #After max_depth increases, the convergence condition will be reached sooner with the same learning rate.
                       )
        
#Train Model
clf_xgb_model = clf_xgb.fit(X_train,y_train,early_stopping_rounds=20,eval_metric=['logloss','auc','error'],eval_set=[(X_train,y_train),(X_test,y_test)], verbose=False)#True
joblib.dump(clf_xgb_model, "model/w2v/clf_XGB_lr0.1_maxdp10_earstop20.pkl")

end1 = datetime.datetime.now()
print('totally time1 is', end1 - start1)


#####################Random Forest##################
start2 = datetime.datetime.now()

clf_rfc = RandomForestClassifier(n_jobs=-1,#The number of jobs (operations) running in parallel for fitting and prediction.
                                 max_features= 'auto' ,
                                 n_estimators=10,
                                 oob_score = True) 
clf_rfc.fit(X_train, y_train)
joblib.dump(clf_rfc, "model/w2v/clf_rfc_nestimators100.pkl")

end2 = datetime.datetime.now()
print('totally time2 is', end2 - start2)

##################linear SVM#################
start3 = datetime.datetime.now()
LinearSVM=SVC(C=1,#Penalty parameter for error items; default is 1
              #Too large a C will over-fit, too small a C will under-fit
              kernel='linear',#Linear kernel functions
              decision_function_shape='ovr',
              tol=0.001,#Stop training error accuracy, default value is 0.001
              probability=True)#Set the parameter probability to True in order to use proba
LinearSVM.fit(X_train, y_train)
joblib.dump(LinearSVM, "model/w2v/clf_lsvm_default.pkl")

end3 = datetime.datetime.now()
print('totally time3 is', end3 - start3)

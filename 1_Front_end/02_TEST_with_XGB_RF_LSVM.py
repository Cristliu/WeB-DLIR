#-- coding:UTF-8 --
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"##Set the currently used GPU device as device #1
G = 1 # GPU number

from sklearn.feature_extraction.text import TfidfVectorizer
import os
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import urllib.parse
import matplotlib.pyplot as plt
import time
# from sklearn.externals import joblib
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np 
import pickle 
import scipy.sparse as sp
import scipy
import copy

time_start_tf=time.time() 

test = []
for line in open('data/Test_url.txt',encoding='utf-8', errors='ignore'):
    test.append(line.strip('\n'))
print('test len:', len(test))

tfidf_test = copy.deepcopy(test)
vectorizer = joblib.load("model/tfidf/TfidfVectorizer.pkl")
x_test_Tfidf = vectorizer.transform(tfidf_test)

time_end_tf=time.time() 
print('totally cost_tfidf_TEST: ',time_end_tf-time_start_tf)

URL_test_Tfidf = x_test_Tfidf
scipy.sparse.save_npz('data/URL_test_Tfidf.npz', URL_test_Tfidf)#######1.33GB

# ######Good##load
# Load_URL_test_Tfidf = sp.load_npz('data/URL_test_Tfidf.npz')
# Load_URL_test_Tfidf

time_start_wv=time.time() 

test = []
for line in open('data/Test_url.txt',encoding='utf-8', errors='ignore'):
    test.append(line.strip('\n'))#
print('test len:', len(test))

w2v_test = copy.deepcopy(test)

pickleFilePath = 'model/w2v/word2vec.pickle'
with open(pickleFilePath, 'rb') as file:
    word2vec_model = pickle.load(file)
word2vec_model

def get_contentVector(cutWords, word2vec_model):
    vector_list = [word2vec_model.wv[k] for k in cutWords if k in word2vec_model.wv]
    contentVector = np.array(vector_list,dtype=list).mean(axis=0)
    return contentVector.tolist()


contentVector_list = []
for i in range(len(w2v_test)):
    cutWords = w2v_test[i]
    contentVector_list.append(get_contentVector(cutWords, word2vec_model))
X = np.array(contentVector_list)

time_end_wv=time.time() 
print('totally cost_w2v_TEST: ',time_end_wv-time_start_wv)

URL_test_w2v = X
URL_test_w2v.shape
np.savez('data/URL_test_w2v.npz',URL_test_w2v=URL_test_w2v) #To specify the URL_test_w2v name

# npzfile=np.load('data/URL_test_w2v.npz') 
# Load_URL_test_w2v=npzfile['URL_test_w2v']##To specify the URL_test_w2v name
# Load_URL_test_w2v.shape

URL_test_Tfidf = URL_test_Tfidf
URL_test_w2v = URL_test_w2v

#####tfidf
clf_tf_xgb = joblib.load("model/tfidf/clf_XGB_lr0.3_maxdp10_earstop10.pkl")
clf_tf_rf = joblib.load("model/tfidf/clf_rfc_nestimators100.pkl")
clf_tf_lsvm = joblib.load("model/tfidf/clf_lsvm_default.pkl")

#####w2v
clf_wv_xgb = joblib.load("model/w2v/clf_XGB_lr0.1_maxdp10_earstop20.pkl")
clf_wv_rf = joblib.load("model/w2v/clf_rfc_nestimators100.pkl")
clf_wv_lsvm = joblib.load("model/w2v/clf_lsvm_default.pkl")

predict_timelist = []

start1 = time.time()
y_tf_xgb = clf_tf_xgb.predict_proba(URL_test_Tfidf)
end1 = time.time()
predict_timelist.append(end1-start1)#

start2 = time.time()
y_tf_rf = clf_tf_rf.predict_proba(URL_test_Tfidf)
end2 = time.time()
predict_timelist.append(end2-start2)

start3 = time.time()
y_tf_lsvm = clf_tf_lsvm.predict_proba(URL_test_Tfidf)
end3 = time.time()
predict_timelist.append(end3-start3)

start4 = time.time()
y_wv_xgb = clf_wv_xgb.predict_proba(URL_test_w2v)
end4 = time.time()
predict_timelist.append(end4-start4)

start5 = time.time() 
y_wv_rf = clf_wv_rf.predict_proba(URL_test_w2v)
end5 = time.time()
predict_timelist.append(end5-start5)

start6 = time.time()
y_wv_lsvm = clf_wv_lsvm.predict_proba(URL_test_w2v)
end6 = time.time()
predict_timelist.append(end6-start6)

for i in range(len(predict_timelist)):
    print(predict_timelist[i])

label_tf_xgb = pd.DataFrame(y_tf_xgb)
label_tf_xgb.columns = ['label0','label1']
label_tf_xgb.to_csv('Results/ex_label_tf_xgb.csv',index=0)
# label_tf_xgb0 = label_tf_xgb['label0']

label_tf_rf = pd.DataFrame(y_tf_rf)
label_tf_rf.columns = ['label0','label1']
label_tf_rf.to_csv('Results/ex_label_tf_rf.csv',index=0)
# label_tf_rf0 = label_tf_rf['label0']

label_tf_lsvm = pd.DataFrame(y_tf_lsvm)
label_tf_lsvm.columns = ['label0','label1']
label_tf_lsvm.to_csv('Results/ex_label_tf_lsvm.csv',index=0)
# label_tf_lsvm0 = label_tf_lsvm['label0']


label_wv_xgb = pd.DataFrame(y_wv_xgb)
label_wv_xgb.columns = ['label0','label1']
label_wv_xgb.to_csv('Results/ex_label_wv_xgb.csv',index=0)
# label_wv_xgb0 = label_wv_xgb['label0']

label_wv_rf = pd.DataFrame(y_wv_rf)
label_wv_rf.columns = ['label0','label1']
label_wv_rf.to_csv('Results/ex_label_wv_rf.csv',index=0)
# label_wv_rf0 = label_wv_rf['label0']

label_wv_lsvm = pd.DataFrame(y_wv_lsvm)
label_wv_lsvm.columns = ['label0','label1']
label_wv_lsvm.to_csv('Results/ex_label_wv_lsvm.csv',index=0)
# label_wv_lsvm0 = label_wv_lsvm['label0']

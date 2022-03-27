# -*- coding:utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd
import xgboost as xgb
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score, precision_recall_curve, auc, average_precision_score

trainfile='../midfile/S2_train_testlist_8Fe_cos_quanbian_nrl_hin2vec_node_vec_8Fe_dim128_walklen10_window4_event_w_darknet.csv'
testfile='../midfile/S2_test_testlist_8Fe_cos_quanbian_nrl_hin2vec_node_vec_8Fe_dim128_walklen10_window4_event_w_darknet.csv'

train=pd.read_csv(trainfile)
tests=pd.read_csv(testfile)

print(trainfile)
print(testfile)
train=train.loc[(train.sign==0)|(train.sign==1)]


dtrain=train.drop(['sign'],axis=1)
dtest=tests.drop(['sign'],axis=1)
biaoqian='sign'

#%%
trainData = xgb.DMatrix(dtrain, label=train[biaoqian])


watchlist = [(trainData,biaoqian)]
y_test = tests[biaoqian]

testData = xgb.DMatrix(dtest)

params = {'booster': 'gbtree',
          'objective': 'binary:logistic',

          'eval_metric': 'auc',
          'gmma': 0.1,
            'max_depth':4,
          'lamda': 10,
          'subsample': 0.7,
          'colsample_bytrene': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.033,
          'tree_method': 'exact',
          'seed': 0
          }


round_boost=500
model = xgb.train(params, trainData, num_boost_round=round_boost)

preds = model.predict(testData)

roc = roc_curve(y_test, preds)
roc = np.mat(roc)
roc = np.transpose(roc)
thr = 0.000
for i in roc:
    if i[0, 0] > thr:
        print(i)
        thr += 0.0005
    if i[0, 0] > 0.002: break
print(str(round_boost)+'\n')

roc = roc_curve(y_test, preds)
roc = np.mat(roc)
roc = np.transpose(roc)

maxfpr = 0.05
roc_auc = roc_auc_score(y_test, preds, max_fpr=maxfpr)
prc_auc = average_precision_score(y_test, preds)
print('Test ROC-Auc:' + str(roc_auc), 'Test PRC-Auc:' + str(prc_auc))

fprs = [0.0025 * i for i in range(1, 21)]
print('FPR|TPR|Thresholds---1')

for fpr in fprs:
    for i in roc:
        if i[0, 0] > fpr:
            print(i)
            break


##save
y_test=y_test.reset_index(drop=True)
yttt=pd.DataFrame(preds, columns=['pre'])
resultcsv = yttt.join(y_test)
resultcsv.to_csv("../"+testfile[:-4]+'.pre')
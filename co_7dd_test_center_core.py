# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:41:48 2018

@author: harres.tariq
"""

import pandas as pd
import numpy as np
#import lightgbm as lgb
import pickle
#from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import random
# %%
cols=[
'relevant cols go here'
]
# %%
pth=''
print('loading data...')
df=pd.read_csv(pth+'.txt',sep=',',names=cols)
df=df.replace(np.nan,0)
print(df.shape)
# %%
df=df.loc[(df['reference_date']=='2018-09-05')]
df=df.loc[df['DOR_DAYS']>=3]
#df=df.loc[df['eng']==0]
df=df.loc[df['CAT']=='C: CORE BASE']
df=df.drop(['reference_date','W1234_VOC_IC_CHANGE', 'CAG_VOC_IC_CNT_WKLY_W1234'],axis=1)
df=df.drop(['CAT'],axis=1)
print(df.shape)
print(df.columns)
# %%
with open(pth+'/maps/to_drop1.pkl', 'rb') as fin:
    to_drop=pickle.load(fin)
df=df.drop(to_drop,axis=1)
df=df.reset_index(drop=True)
# %%
print('categoricals...')
cat=[]
with open(pth+'/maps/cat.pkl', 'rb') as fin:
    cat=pickle.load(fin)
with open(pth+'/maps/to_drop2.pkl', 'rb') as fin:
    to_drop=pickle.load(fin)
df=df.drop(to_drop,axis=1)
for c in df.columns:
    if(str(df[c].dtype)=='object'):
        if((np.sum(df[c].str.contains('E'))>0) & (c!='CAT')):#('E' in str(df[c][0])):
           df[c]=df[c].apply(lambda x: float(x.replace('E','e').replace(' ','+')))#float(df[c].replace('E','e').replace(' ','+'))
        else:
            try:
                df[c]=df[c].apply(lambda x: float(str(x).replace('.','0.0')))
            except Exception as err:
                print(c,err)
for c in cat:
    print(c)
    mapp={}
    with open(pth+'/maps/'+str(c+'OH')+'.pkl', 'rb') as fin:
        mapp = pickle.load(fin)
    mk=list(mapp.keys())
    uni=df[c].unique()
    diff=list(set(uni)-set(mk))
    mv=list(mapp.values())
    for d in diff:
        mapp[d]=random.choice(mv)
    df[c+'OH']=df[c].apply(lambda x: mapp[x])
df=df.drop(cat,axis=1)
# %%
ID=df['ID'].values
y_train=df['churn'].values
x_train=df.drop(['ID','churn'],axis=1)
with open(pth+'/maps/x_cols.pkl', 'rb') as fin:
    x_cols=pickle.load(fin)
x_train=x_train[x_cols]
# %%
print('Predicting...')
flds=5
p_train=np.zeros((x_train.shape[0],flds))
for count in range(0,flds):
    with open(pth+'/models/mod'+str(count)+'.pkl', 'rb') as fin:
          model=pickle.load(fin)
    pred = model.predict(x_train, num_iteration=model.best_iteration)
    p_test = (np.exp(pred) - 1.0).clip(0,1)
    p_train[:,count] =p_test
    imp=model.feature_importance()
    imp_idx=sorted(range(len(imp)), key=lambda k: imp[k])
    most_imp=x_train.columns[imp_idx[-10:]]
    print(imp[imp_idx[-10:]])
    print(most_imp)
#    pred_bin=(p_test>=0.3)
#    score=precision_score(y_train,pred_bin)
#    print('precision score: ', score)
#    score=recall_score(y_train,pred_bin)
#    print('recall score: ', score)
#    score=accuracy_score(y_train,pred_bin)
#    print('accuracy score: ', score)
#    cm=confusion_matrix(y_train,pred_bin)
#    print(cm)
# %%
print('average results.....')
p_avg=np.mean(p_train,axis=1)
# %%
thr=0.2
sub=pd.DataFrame()
sub['ID']=ID
sub['CHURN']=y_train
sub['P_CHURN']=(p_avg>=thr).astype(int)
sub['PROB']=p_avg
#sub.to_csv(pth+'/data/test_results.txt', index=False)
#print('results saved for thr: ', thr)
# %%
print('threshold.....')
for i in np.arange(0,1,0.05):
       print('threshold: ', i)
       pred_bin=(p_avg>=i)
       score=precision_score(y_train,pred_bin)
       print('precision score: ', score)
       score=recall_score(y_train,pred_bin)
       print('recall score: ', score)
       score=accuracy_score(y_train,pred_bin)
       print('accuracy score: ', score)
       cm=confusion_matrix(y_train,pred_bin)
       print(cm)
# %%
#top_idx=sorted(range(len(p_avg)), key=lambda k: p_avg[k])
#p_avg_2=p_avg[top_idx[-3350000:]]
#y_train_2=y_train[top_idx[-3350000:]]
#for i in np.arange(0,1,0.05):
#       print('threshold: ', i)
#       pred_bin=(p_avg_2>=i)
#       score=precision_score(y_train_2,pred_bin)
#       print('precision score: ', score)
#       score=recall_score(y_train_2,pred_bin)
#       print('recall score: ', score)
#       score=accuracy_score(y_train_2,pred_bin)
#       print('accuracy score: ', score)
#       cm=confusion_matrix(y_train_2,pred_bin)
#       print(cm)
# %%
idx=sorted(range(len(p_avg)), key=lambda j: p_avg[j])
for lim in [1107]:
    print('lower limit: ', lim)
    pred_bin=np.concatenate([np.zeros(len(p_avg)-lim),np.ones(lim)])
    score=precision_score(y_train[idx],pred_bin)
    print('precision score: ', score)
    score=recall_score(y_train[idx],pred_bin)
    print('recall score: ', score)
    score=accuracy_score(y_train[idx],pred_bin)
    print('accuracy score: ', score)
    cm=confusion_matrix(y_train[idx],pred_bin)
    print(cm)
# %%
#print('deciles......')
#idx=sorted(range(len(p_avg)), key=lambda j: p_avg[j])
#for d in np.arange(0,1,0.1):
#    lim=int(np.floor(d*len(p_avg)))
#    dp=p_avg[idx[lim]]
#    print(d,dp)
#    pred_bin=(p_avg>=dp)
#    score=precision_score(y_train,pred_bin)
#    print('precision score: ', score)
#    score=recall_score(y_train,pred_bin)
#    print('recall score: ', score)
#    score=accuracy_score(y_train,pred_bin)
#    print('accuracy score: ', score)
#    cm=confusion_matrix(y_train,pred_bin)
#    print(cm)
#    
#    

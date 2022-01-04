# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:17:37 2018

@author: harres.tariq
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
# %%
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True
# %%
cols=[
'relevant col names go here'
]
# %%
pth=''
print('loading data...')
df=pd.read_csv(pth+'.txt',sep=',',names=cols)
df=df.replace(np.nan,0)
print(df.shape)
# %%
df=df.loc[(df['reference_date']=='2018-07-01')]
df=df.loc[df['DOR_DAYS']>=3]
#df=df.loc[df['eng']==0]
df=df.loc[df['CAT']=='C: CORE BASE']
df=df.drop(['reference_date','W1234_VOC_IC_CHANGE', 'CAG_VOC_IC_CNT_WKLY_W1234'],axis=1)
df=df.drop(['CAT'],axis=1)
print(df.shape)
print(df.columns)
to_drop=[]
count=0
for c in df.columns:
    if(len(df[c].unique())==1):
        to_drop.append(c)
        print(c)
        count+=1
print(count)
with open(pth+'/maps/to_drop1.pkl', 'wb') as fout:
    pickle.dump(to_drop, fout)
df=df.drop(to_drop,axis=1)
df=df.reset_index(drop=True)
# %%a=df.head(10)
print('categoricals...')
#df['ind']=df['ind'].apply(lambda x: str(x))
#(np.sum(df[c].str.contains('E'))>0)
cat=[]
to_drop=[]
for c in df.columns:
    if(str(df[c].dtype)=='object'):
        if((np.sum(df[c].str.contains('E'))>0) & (c!='CAT')):#('E' in str(df[c][0])):
           df[c]=df[c].apply(lambda x: float(x.replace('E','e').replace(' ','+')))#float(df[c].replace('E','e').replace(' ','+'))
        else:
            try:
                #print('here')
                df[c]=df[c].apply(lambda x: float(str(x).replace('.','0.0')))
            except Exception as err:
                print(c)
                cat.append(c)
                uni=df[c].unique()
                mapp={}
                count=0
                for v in uni:
                    mapp[v]=count
                    count+=1
                df[c+'OH']=df[c].apply(lambda x: mapp[x])
                with open(pth+'/maps/'+str(c+'OH')+'.pkl', 'wb') as fout:
                    pickle.dump(mapp, fout)
            if(len(df[c].unique())==1):
                to_drop.append(c)
print('cat: ', cat)
print('drop: ', to_drop)
with open(pth+'/maps/to_drop2.pkl', 'wb') as fout:
    pickle.dump(to_drop, fout)
with open(pth+'/maps/cat.pkl', 'wb') as fout:
    pickle.dump(cat, fout)
df=df.drop(cat,axis=1)
df=df.drop(to_drop,axis=1)
# %%
ID=df['ID'].values
y_train=df['churn'].values
x_train=df.drop(['ID','churn'],axis=1)
x_cols=x_train.columns
with open(pth+'/maps/x_cols.pkl', 'wb') as fout:
    pickle.dump(x_cols, fout)
# %%
print('training...')
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'num_leaves': 128,#128
          'learning_rate': 0.01,#0.1
          'max_bin': 255,
          'reg_alpha': 1, #10
          'reg_lambda': 10,#50
          'min_data_in_leaf':5000,#10000
          #'feature_fraction':0.6,#0.7
          'colsample_bytree':0.6,
          #'bagging_freq':1,
          'subsample_freq':1,
          #'bagging_fraction':0.6#0.7
          'subsample':0.6
          }
print('training model....')
count=0
folds=5
skf=StratifiedKFold(y_train,n_folds=folds,random_state=87,shuffle=False)
g=0
for train_idx, test_idx in skf:#enumerate(skf.split(x_train, y_train)):
    model = lgb.train(params, lgb.Dataset(x_train.iloc[train_idx].copy(), label=y_train[train_idx]), 2000, 
                   lgb.Dataset(x_train.iloc[test_idx].copy(), label=y_train[test_idx]), verbose_eval=10,
                   early_stopping_rounds=100)#,feval=gini_lgb)
    with open(pth+'/models/mod'+str(count)+'.pkl', 'wb') as fout:
          pickle.dump(model, fout)
    pred = model.predict(x_train.iloc[test_idx], num_iteration=model.best_iteration)
    p_test = (np.exp(pred) - 1.0).clip(0,1)
    print('Iteration Gini: ', gini_normalized(y_train[test_idx],p_test))
    g+=gini_normalized(y_train[test_idx],p_test)
    count+=1
    print('Iterations Done: ', count)
    imp=model.feature_importance()
    imp_idx=sorted(range(len(imp)), key=lambda k: imp[k])
    most_imp=x_train.columns[imp_idx[-10:]]
    print(imp[imp_idx[-10:]])
    print(most_imp)
    pred_bin=(p_test>=0.25)
    score=precision_score(y_train[test_idx],pred_bin)
    print('precision score: ', score)
    score=recall_score(y_train[test_idx],pred_bin)
    print('recall_score: ',score)
g_avg=g/folds
print('Avg Gini: ', g_avg)







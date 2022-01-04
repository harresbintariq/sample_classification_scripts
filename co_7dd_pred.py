import pandas as pd
#import lightgbm as lgb
import numpy as np
import pickle
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, r$
import random
from datetime import datetime, timedelta
# %%
path=''
print('loading data...')
df=pd.read_csv(path+'.csv',low_memory=True)
df=df.replace(np.NaN,0)
# %%
col=[]
for c in df.columns:
    col.append(c.split('.')[1])
df.columns=col
# %%
print(df.shape)
df=df.drop(['w1234_voc_ic_change', 'cag_voc_ic_cnt_wkly_w1234','cat'],axis=1)
#df=df.drop(['dor_days_comp','dor_flag','delta_rev_roll','arpu_w_m_delta','arpu_m_bin','arpu_flg','d0_rev_roll_flg','subs_count','region'],axis=1)
df=df.loc[df['dor_days']>=3]
print(df.shape)
print(df.columns)
# %%
with open(path+'/maps/to_drop1.pkl', 'rb') as fin:
    to_drop=pickle.load(fin)
ls=[]
for s in to_drop:
    ls.append(s.lower())
to_drop=ls
df=df.drop(to_drop,axis=1)
df=df.reset_index(drop=True)
# %%
print('categoricals...')
cat=[]
with open(path+'/maps/cat.pkl', 'rb') as fin:
    cat=pickle.load(fin)
with open(path+'/maps/to_drop2.pkl', 'rb') as fin:
    to_drop=pickle.load(fin)
ls=[]
for s in to_drop:
    ls.append(s.lower())
to_drop=ls
df=df.drop(to_drop,axis=1)
for c in df.columns:
    if(str(df[c].dtype)=='object'):
        if((np.sum(df[c].str.contains('E'))>0) & (c!='cat')):#('E' in str(df[c][0])):
           df[c]=df[c].apply(lambda x: float(x.replace('E','e').replace(' ','+')))#float(df[c].replace('E','e').replace(' ','+'))
        else:
            try:
                df[c]=df[c].apply(lambda x: float(str(x).replace('.','0.0')))
            except Exception as err:
                print(c,err)
for c in cat:
    print(c)
    mapp={}
    with open(path+'/maps/'+str(c+'OH')+'.pkl', 'rb') as fin:
        mapp = pickle.load(fin)
    mk=list(mapp.keys())
    uni=df[c.lower()].unique()
    diff=list(set(uni)-set(mk))
    mv=list(mapp.values())
    for d in diff:
        mapp[d]=random.choice(mv)
    df[c.lower()+'oh']=df[c.lower()].apply(lambda x: mapp[x])
df=df.drop([c.lower() for c in cat],axis=1)
# %%
ID=df['id'].values
msisdn=df['no'].values
#y_train=df['churn'].values
x_train=df.drop(['id','no'],axis=1)
#with open(path+'/maps/x_cols.pkl', 'rb') as fin:
#    x_cols=pickle.load(fin)
x_cols=pd.read_pickle(path+'/maps/x_cols.pkl')
ls=[]
for s in x_cols:
    ls.append(s.lower())
x_cols=ls
x_train=x_train[x_cols]
# %%
print('Predicting...')
flds=5
p_train=np.zeros((x_train.shape[0],flds))
for count in range(0,flds):
    with open(path+'/models/mod'+str(count)+'.pkl', 'rb') as fin:
          model=pickle.load(fin)
    pred = model.predict(x_train, num_iteration=model.best_iteration)
    p_test = (np.exp(pred) - 1.0).clip(0,1)
    p_train[:,count] =p_test
    print(count)
# %%
from datetime import datetime, timedelta
print('average results.....')
p_avg=np.mean(p_train,axis=1)
thr=0.15
p_test_bin=(p_avg>=thr).astype(int)
sub=pd.DataFrame()
sub['access_method_id']=ID
msisdn=['92'+str(x).split('.')[0] for x in msisdn]
sub['msisdn']=msisdn
sub['churn']=p_test_bin
sub['prob']=p_avg
sub['insert_datetime']=str(datetime.now()).split(' ')[0]
sub['reference_date']=str(datetime.now()-timedelta(days=2)).split(' ')[0]
sub['id']=1
sub['cat']='cc'
sub=sub.loc[sub['msisdn']!='920']
print(sub.shape)
print(np.sum(p_test_bin),np.mean(p_test_bin))
file_name='jazz_7day_cc_pred.csv'
sub.to_csv(path+'/data/'+file_name, index=False)
# %%
"""
import pyodbc

pyodbc.pooling = False
conn = pyodbc.connect('DRIVER={Teradata};DBCNAME=10.50.27.11;UID=up_talham;PWD=Newpass123', autocommit=True)
conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-8')
#conn.setencoding(str, encoding='utf-8')
conn.setencoding(encoding='utf-8')
cursor = conn.cursor()
cursor.execute("Update dp_ads.model_health_check set scoring = 1 where model_name = 'Jazz_New' and poc = 'Harres' and cast(run_dt as date)= date")
"""


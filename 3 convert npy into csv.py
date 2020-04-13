import numpy as np
import pandas as pd
import operator
list_score=[]
scores = np.load('LSTM18368_score.npy')
print(len(scores))
print(scores[0:10])
df=pd.read_csv('suspect_drug.csv')
label_m=df['sus_name']
label_g=df['drug_name']

for i in range(0,len(df)):
    list_score.append({'label_m':label_m[i],'label_g':label_g[i],'score':scores[i]})


list_score.sort(key=lambda x:x['score'],reverse=True)

L=pd.DataFrame(list_score[0:3000])
L.to_csv('susperct_drug_pairs.csv',index=None)

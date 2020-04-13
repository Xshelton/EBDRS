import pandas as pd
fname='sum_up_result'
#fname='0_sum_up(drug)'
df=pd.read_csv('{}.csv'.format(fname))
A=df['count']
B=df['drug']
C=df['score']
list_mm=[]
for i in range(0,len(df)):
    list_mm.append({'count':A[i],'name':B[i],'score':C[i],'ave':round(C[i]/A[i],3),'rank_T':0,'rank_I':0,'rank_count':0,'rank_F':0})
list_mm = sorted(list_mm, key=lambda k: k['count'],reverse=True)
for i in range(0,len(list_mm)):
    list_mm[i]['rank_count']=i+1
list_mm = sorted(list_mm, key=lambda k: k['score'],reverse=True)
for i in range(0,len(list_mm)):
    list_mm[i]['rank_T']=i+1
list_mm = sorted(list_mm, key=lambda k: k['ave'],reverse=True)
for i in range(0,len(list_mm)):
    list_mm[i]['rank_I']=i+1
for i in range(0,len(list_mm)):
    list_mm[i]['rank_F']=list_mm[i]['rank_T']+list_mm[i]['rank_I']+list_mm[i]['rank_count']
list_mm = sorted(list_mm, key=lambda k: k['rank_F'],reverse=False)
print(list_mm)
gg=pd.DataFrame(list_mm)
gg.to_csv('final_{}_rank.csv'.format(fname),index=None)

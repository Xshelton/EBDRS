import pandas as pd
df=pd.read_csv('susperct_drug_pairs.csv')
drug=df['label_g']
samples=df['label_m']
score=df['score']
list_result=[]
list_result_detail=[]
for i in range(0,len(df)):
  if score[i]>=0:
      if drug[i] not in list_result:#不在这个里面
          list_result.append(drug[i])
          list_result_detail.append({'drug':drug[i],'count':1,'score':score[i]})#初始化
      else: #在里面
          k=list_result.index(drug[i])
          list_result_detail[k]['count']=list_result_detail[k]['count']+1
          list_result_detail[k]['score']=list_result_detail[k]['score']+score[i]
print(list_result_detail)
list_result_detail.sort(key=lambda X:X['score'],reverse=True)
ls=pd.DataFrame(list_result_detail)
ls.to_csv('sum_up_result.csv',index=None)

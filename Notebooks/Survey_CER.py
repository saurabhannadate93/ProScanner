# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:40:07 2019

@author: arpan
"""

import os
import pandas as pd
import editdistance
os.chdir('C:/Users/arpan/OneDrive/Desktop/Acads/Spring/DeepLearning/ProScanner/Notebooks/')


response = pd.read_csv('Responses_Survey.csv',encoding = 'latin')
response = response.iloc[:,0:21]
response2 = response.copy()

num_responses = response2.shape[0]

actual = pd.read_csv('Actual_responses.csv')

ques_list = response2.columns.tolist()[1:]

for i in ques_list:
    act = actual.loc[actual['Key']==i,'Actual'].values[0]
    list_val=[]
    eval_list=response2[i].tolist()
    for val in eval_list:
        list_val.append(editdistance.eval(act,val))
    col_name=i+'_val'
    response2[col_name] = list_val


total_den=0
total_num=0


for i in ques_list:
    act = actual.loc[actual['Key']==i,'Actual'].values[0]
    total_den+=num_responses*len(act)
    
    col_name = i+'_val'
    total_num+=response2[col_name].sum()
    
    
        
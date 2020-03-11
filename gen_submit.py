import os
import random
import pandas as pd
import numpy as np
from collections import defaultdict

ind2label = [ 'N_N', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7',
                '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7',
                '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7',
                '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',
                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7',
                '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7',
                '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7' ]
                
csv_file_lists = os.listdir('pred_csv/')

result = defaultdict(list)
for csv_file in csv_file_lists:
    print(csv_file)
    wt = 1#float(csv_file.split('_')[2])
    file = pd.read_csv('pred_csv/' + csv_file)
    df = pd.DataFrame(file)

    for i in range(len(df)):
        for j in range(len(ind2label)):
            if(len(result[i]) <50):
                result[i].append(df[ind2label[j]][i] * wt)
            else:
                result[i][j] += df[ind2label[j]][i] * wt

result_file = open("predictions.txt", 'w')
for key in result.keys():
    pred = result[key]
    max_ind = pred.index(max(pred))
   # print(pred[max_ind]/len(csv_file_lists))
    pre_label = ind2label[max_ind]
    result_file.write(pre_label + '\r\n')

result_file.close()    
cmd = 'zip predictions.zip predictions.txt'
os.system(cmd)  

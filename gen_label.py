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

result = {}
test_names = []
test_file = open('./faces_224/anns/test_list.txt', 'r')
for line in test_file.readlines():
    data_info = line.strip('\n')
    test_names.append(data_info)
    result[data_info] = []

for csv_file in csv_file_lists:
    print(csv_file)
    wt = 1 #float(csv_file.split('_')[2])
    file = pd.read_csv('pred_csv/' + csv_file)
    df = pd.DataFrame(file)

    for i in range(len(df)):
        for j in range(len(ind2label)):
            if(len(result[test_names[i]]) <50):
                result[test_names[i]].append(df[ind2label[j]][i] * wt)
            else:
                result[test_names[i]][j] += df[ind2label[j]][i] * wt
                
result_tmp = []
for key in result.keys():
    pred = result[key]
    max_ind = pred.index(max(pred))
    result_tmp.append([key,ind2label[max_ind],pred[max_ind]])
    print(key,ind2label[max_ind],pred[max_ind])
    
result_sorted = sorted(result_tmp,key = lambda x:x[2],reverse = True)

set_num = int(0.06 * len(result_sorted))
result_file = open('./faces_224/anns/test_psdo.txt', 'w')
for ind in range(set_num):
    itms = result_sorted[ind]
    print(itms)
    
    result_file.write(itms[0]+' '+itms[1]+ '\r\n')

result_file.close()    


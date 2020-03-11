import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import Baseline
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from datasets import test_collate,TestDataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
   
ind2label = [ 'N_N', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7',
                '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7',
                '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7',
                '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',
                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7',
                '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7',
                '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7' ]
                                    
model_name = 'AlexNet' 

model_paths = ['0_AlexNet_0.22756_8.pth',
               '1_AlexNet_0.18867_13.pth',
               '2_AlexNet_0.16229_46.pth',
               '3_AlexNet_0.19244_24.pth',
               '4_AlexNet_0.21780_15.pth',
               '5_AlexNet_0.19196_21.pth',
               '6_AlexNet_0.23467_18.pth',
               '7_AlexNet_0.25146_19.pth',
               '8_AlexNet_0.18022_7.pth',
               '9_AlexNet_0.19684_34.pth'
               ]
model = Baseline(model='test', model_name = model_name)

test_data = TestDataset('./faces_224/anns/test_ld.txt')
test_loader = DataLoader(dataset=test_data, batch_size=48, shuffle=False, num_workers=2, collate_fn=test_collate )

for train_model in model_paths:
    print('-------------------model name: {}-------------------'.format(train_model))
    model.load_param('models/' + train_model)
    model = model.cuda() 
    model = model.eval()
    
    result = defaultdict(list)
    with torch.no_grad():
        for img, lms in test_loader:
            img = img.cuda()
            lms = lms.cuda()
            
            prds,_ = model(img, lms)
            prds = F.softmax(prds)
            prds = prds.cpu().numpy()

            for pred in prds:
                pred = list(pred)
                for ind, prd in enumerate(pred):
                    result[ind2label[ind]].append(prd)

    dataframe = pd.DataFrame(result)
    dataframe.to_csv('pred_csv/' + train_model[:-4] + ".csv",index=False,sep=',')
    
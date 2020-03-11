import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import Baseline
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
model_path = './models/1_AlexNet_0.19467_23.pth'

model = Baseline(model='test',model_name = model_name)
model.load_param(model_path)
model = model.cuda() 
model = model.eval()
 
test_data = TestDataset('./faces_224/anns/test_ld.txt')
test_loader = DataLoader(dataset=test_data, batch_size=48, shuffle=False, num_workers=2, collate_fn=test_collate )

result_file = open("predictions.txt", 'w')
with torch.no_grad():
    for img, lms in test_loader:
        img = img.cuda()
        lms = lms.cuda()
        
        prds = model(img, lms)
        prds = F.softmax(prds)
        prds = prds.cpu().numpy()

        for pred in prds:
            pred = list(pred)
            max_ind = pred.index(max(pred))
            pre_label = ind2label[max_ind]
            result_file.write(pre_label + '\r\n')

result_file.close()     
cmd = 'zip predictions.zip predictions.txt'
os.system(cmd)   

import os
import logging
import torch
import torch.optim
import numpy as np
from torch.optim import lr_scheduler
from datasets import make_dataloader,RandomSampler,train_collate
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import data_parallel
from model import Baseline
import pandas as pd
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
kd_id = 1
kd_num = 10
batch_size = 64
instance_num = 2
tr_w = 0.5
#device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")

ind2label = [ 'N_N', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7',
                '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7',
                '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7',
                '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',
                '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7',
                '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7',
                '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7' ]

def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        if epoch < 2:
            param_group['lr'] = 0.00001
        elif epoch < 20:
            param_group['lr'] = 0.0001
        else:
            param_group['lr'] = param_group['lr'] * 0.95
    print('Adjust learning rate: {}'.format(param_group['lr']))
    
def train_fuc(model, epoch):
    model = model.train()
  
    step_loss = 0
    correct = 0
    num_all = 0
    for step, (images,lms,labels)in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        lms = lms.cuda()

        prds,feat = model(images, lms)
    #    loss = model.criterion(prds, labels) + tr_w*model.triplet(feat, labels)
        loss = model.xent(prds, labels) + tr_w*model.triplet(feat, labels)   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = torch.argmax(prds.detach(), 1)
        correct += (prediction == labels).sum().float()
        num_all += len(labels)

        step_loss += loss
        if(step%20 == 0):
            print('[{}/{}/{}], Loss:{}, Acc: {}'.format(step, train_length, epoch, 
                        '%.5f' % (step_loss/20), '%.5f' % (correct/num_all)))
            step_loss = 0
            correct = 0
            num_all = 0
            
    print('--------------------------------------------------------------------')
    
def val_fuc(model, epoch): 
    model = model.eval()
    
    correct = 0
    num_all = 0

    with torch.no_grad():
        result = defaultdict(list)
        for step, (images,lms,labels)in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            lms = lms.cuda()
            
            prds,_ = model(images, lms)
     
            prediction = torch.argmax(prds.detach(), 1)
            correct += (prediction == labels).sum().float()
            num_all += len(labels)
            
            prds = F.softmax(prds)
            prds = prds.cpu().numpy()
            
            for pred, label in zip(prds, labels):
                pred = list(pred)
                for ind, prd in enumerate(pred):
                    result[ind2label[ind]].append(prd)
                result['gt'].append(ind2label[label])

        dataframe = pd.DataFrame(result)
        dataframe.to_csv("models/val_" + str(kd_id) + '_' + model_name + '_' + '%.5f' % (correct/num_all) +".csv",index=False,sep=',')

        print('Epoch: {}, Val_Acc: {}'.format(epoch, '%.5f' % (correct/num_all)))
        print('--------------------------------------------------------------------')
        return correct/num_all
    
# model_name = 'mobilfacenet'
# model_path='model_mobilefacenet.pth'

# model_name = 'model_ir_se50'
# model_path='model_ir_se50.pth'

# model_name = 'resnet50_ibn_a'
# model_path = 'resnet50_ibn_a.pth.tar'

# model_name = 'se_resnet50'
# model_path = 'se_resnet50-ce0d4300.pth'

model_name = 'AlexNet'
model_path='model_mobilefacenet.pth'

# model_name = 'MiniXception'
# model_path = ' '

# model_name = 'ConvNet'
# model_path = ' '

# model_name = 'MixNet'
# model_path = ' ' 

model = Baseline(model='train',model_name = model_name, model_path=model_path)
#model.load_param('models/model_1_180000.pth')
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# kd_id = 0
# kd_num = 7
# batch_size = 48
# instance_num = 1
train_data, val_data, trains, vals = make_dataloader(kd_id,kd_num)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=RandomSampler(trains, batch_size, instance_num), shuffle=False, num_workers=2, collate_fn=train_collate)
#train_loader = DataLoader(dataset=train_data, batch_size=48, shuffle=False, num_workers=2, collate_fn=train_collate)
val_loader   = DataLoader(dataset=val_data,   batch_size=64, shuffle=False, num_workers=2, collate_fn=train_collate )
train_length = len(train_loader)
val_length = len(val_loader)

if __name__ == '__main__':
    max_epoch = 50
    max_val_acc = 0

    for epoch in range(0,max_epoch):
        adjust_lr(optimizer, epoch)
        train_fuc(model, epoch) 
        val_acc = val_fuc(model, epoch) 

        torch.save(model.state_dict(), 'models/'+ str(kd_id)+'_'+ model_name + '_'+ '%.5f'%(val_acc) +'_'+ str(epoch) +'.pth')             
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from resnet_ibn_a import resnet50_ibn_a
from senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from facenet import Backbone, Arcface, MobileFaceNet, Am_softmax
from alexnet import AlexNet,weight_init
#from Alexnet import AlexNet
from emcnet import *
from mixnet import MixNet
from triplet_loss import TripletLoss
from pytorch_revgrad import RevGrad
from center_loss import CenterLoss
from resmasking import resmasking_dropout1

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class Baseline(nn.Module):
    def __init__(self,model='test',model_name = 'resnet50_ibn_a', model_path='', last_stride = 1):
        super(Baseline, self).__init__()
        
        if(model_name == 'resnet50_ibn_a'):
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride = last_stride)
            print('Model name = {}'.format(model_name))
        if(model_name == 'mobilfacenet'):
            self.in_planes = 512
            self.base = MobileFaceNet(512)
            print('Model name = {}'.format(model_name))
        if(model_name == 'model_ir_se50'):
            self.in_planes = 512
            self.base = Backbone(50, 'ir_se')
            print('Model name = {}'.format(model_name))
        if(model_name == 'se_resnet50'):
            self.in_planes = 2048
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
            print('Model name = {}'.format(model_name))

        if(model_name == 'AlexNet'):
            self.base = AlexNet()
           # self.base.load_param('alexnet-owt-4df8aa71.pth')
            self.base.apply(weight_init)
        
        if(model_name == 'MiniXception'):
            self.base = MiniXception()
            self.base.apply(weight_init)
         
        if(model_name == 'ConvNet'):
            self.base = ConvNet()
            self.base.apply(weight_init)
        
        if(model_name == 'PretrConvNet'):
            self.base = PretrConvNet()
            self.base.apply(weight_init)
        
        if(model_name == 'MixNet'):
            self.base = MixNet()
            
        # if(model == 'train'):
            # self.base.load_param(model_path)

        self.fc_lms = nn.Linear(136, 256, bias=True)
        self.bn_lms = nn.BatchNorm1d(256)

        self.fc_head = nn.Linear(512, 250, bias=False)
        self.bn_head = nn.BatchNorm1d(250)
        self.bn_head.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(250, 50 , bias=False)

        self.relu = nn.ReLU(inplace=True)
        
        self.fc_lms.apply(weights_init_kaiming)
        self.bn_lms.apply(weights_init_kaiming)
        
        self.fc_head.apply(weights_init_classifier) 
        self.bn_head.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
        self.dropout = nn.Dropout(p=0.6)

        weight = [0.3,0.3,0.3,0.3,0.37,0.43,0.37,0.3,
                     0.3,0.3,0.3,0.3,0.3, 0.3, 0.3,
                     0.3,0.3,0.3,0.3,0.3, 0.3, 0.3,
                     0.3,0.3,0.38,0.3,0.3, 0.3,0.46,
                     0.3,0.3,0.52,0.3,0.3,  0.3,0.3,
                     0.3,0.3,0.3, 0.3,0.47, 0.3,0.3,
                     0.57,0.3,0.48,0.3,0.68,0.3,0.3]
                     
        weight = np.array(weight)*10
        self.weight = torch.tensor(weight, dtype=torch.float32).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.xent = CrossEntropyLabelSmooth(num_classes=50)
        self.triplet = TripletLoss(0.3)
        #self.triplet = CenterLoss( num_classes=50, feat_dim=250)
        
    def forward(self, x, lms):
        x = self.base(x)
        
        lms = self.fc_lms(lms)
      # #  lms = self.bn_lms(lms)
        lms = self.relu(lms)      ##

        x_cat = torch.cat((lms, x), 1)

        feat = self.fc_head(x_cat)
        feat = self.relu(feat)    ##
        feat = self.dropout(feat) ##
        feat = self.bn_head(feat)
        x = self.classifier(feat)
 
        return x, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
                   
# class Baseline(nn.Module):
    # def __init__(self,model='test',model_name = 'resnet50_ibn_a', model_path='', last_stride = 1):
        # super(Baseline, self).__init__()
        
        # if(model_name == 'resnet50_ibn_a'):
            # self.in_planes = 2048
            # self.base = resnet50_ibn_a(last_stride = last_stride)
            # print('Model name = {}'.format(model_name))
        # if(model_name == 'mobilfacenet'):
            # self.in_planes = 512
            # self.base = MobileFaceNet(512)
            # print('Model name = {}'.format(model_name))
        # if(model_name == 'model_ir_se50'):
            # self.in_planes = 512
            # self.base = Backbone(50, 'ir_se')
            # print('Model name = {}'.format(model_name))
        # if(model_name == 'se_resnet50'):
            # self.in_planes = 2048
            # self.base = SENet(block=SEResNetBottleneck, 
                              # layers=[3, 4, 6, 3], 
                              # groups=1, 
                              # reduction=16,
                              # dropout_p=None, 
                              # inplanes=64, 
                              # input_3x3=False,
                              # downsample_kernel_size=1, 
                              # downsample_padding=0,
                              # last_stride=last_stride) 
            # print('Model name = {}'.format(model_name))

        # if(model_name == 'AlexNet'):
            # self.base = AlexNet()
           # # self.base.load_param('alexnet-owt-4df8aa71.pth')
            # self.base.apply(weight_init)
        
        # if(model_name == 'MiniXception'):
            # self.base = MiniXception()
            # self.base.apply(weight_init)
         
        # if(model_name == 'ConvNet'):
            # self.base = ConvNet()
            # self.base.apply(weight_init)
        
        # if(model_name == 'PretrConvNet'):
            # self.base = PretrConvNet()
            # self.base.apply(weight_init)
        
        # if(model_name == 'MixNet'):
            # self.base = MixNet()
            
        # # if(model == 'train'):
            # # self.base.load_param(model_path)

        # self.fc_lms = nn.Linear(136, 136, bias=True)
        # self.bn_lms = nn.BatchNorm1d(136)

        # self.fc_head = nn.Linear(392, 250, bias=False)
        # self.bn_head = nn.BatchNorm1d(250)
        # self.bn_head.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(250, 50 , bias=False)

        # self.relu = nn.ReLU(inplace=True)
        
        # self.fc_lms.apply(weights_init_kaiming)
        # self.bn_lms.apply(weights_init_kaiming)
        
        # self.fc_head.apply(weights_init_classifier) 
        # self.bn_head.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        
        # self.dropout = nn.Dropout(p=0.8)

        # weight = [0.3,0.3,0.3,0.3,0.37,0.43,0.37,0.3,
                     # 0.3,0.3,0.3,0.3,0.3, 0.3, 0.3,
                     # 0.3,0.3,0.3,0.3,0.3, 0.3, 0.3,
                     # 0.3,0.3,0.38,0.3,0.3, 0.3,0.46,
                     # 0.3,0.3,0.52,0.3,0.3,  0.3,0.3,
                     # 0.3,0.3,0.3, 0.3,0.47, 0.3,0.3,
                     # 0.57,0.3,0.48,0.3,0.68,0.3,0.3]
                     
        # weight = np.array(weight)*10
        # self.weight = torch.tensor(weight, dtype=torch.float32).cuda()

        # self.criterion = nn.CrossEntropyLoss()
        # self.xent = CrossEntropyLabelSmooth(num_classes=50)
        # self.triplet = TripletLoss(0.4)
        # #self.triplet = CenterLoss( num_classes=50, feat_dim=250)
        
    # def forward(self, x, lms):
        # x = self.base(x)
        
        # lms = self.fc_lms(lms)
      # # #  lms = self.bn_lms(lms)
        # lms = self.relu(lms)      ##

        # x_cat = torch.cat((lms, x), 1)

        # feat = self.fc_head(x_cat)
        # feat = self.relu(feat)    ##
        # feat = self.dropout(feat) ##
        # feat = self.bn_head(feat)
        # x = self.classifier(feat)
 
        # return x, feat

    # def load_param(self, trained_path):
        # param_dict = torch.load(trained_path)
        # for i in param_dict:
            # self.state_dict()[i].copy_(param_dict[i])

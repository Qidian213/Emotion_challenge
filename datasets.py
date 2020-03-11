from __future__ import print_function
from PIL import Image
import numpy as np
import math 
import copy
import random
import json
import torch 
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class2label = { 'N_N':  0, '1_1':  1, '1_2':  2, '1_3':  3, '1_4':  4, '1_5':  5, '1_6':  6, '1_7': 7,
                '2_1':  8, '2_2':  9, '2_3': 10, '2_4': 11, '2_5': 12, '2_6': 13, '2_7': 14,
                '3_1': 15, '3_2': 16, '3_3': 17, '3_4': 18, '3_5': 19, '3_6': 20, '3_7': 21,
                '4_1': 22, '4_2': 23, '4_3': 24, '4_4': 25, '4_5': 26, '4_6': 27, '4_7': 28,
                '5_1': 29, '5_2': 30, '5_3': 31, '5_4': 32, '5_5': 33, '5_6': 34, '5_7': 35,
                '6_1': 36, '6_2': 37, '6_3': 38, '6_4': 39, '6_5': 40, '6_6': 41, '6_7': 42,
                '7_1': 43, '7_2': 44, '7_3': 45, '7_4': 46, '7_5': 47, '7_6': 48, '7_7': 49 }

def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    lms = []

    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.append(batch[b][0])
            lms.append(batch[b][1])
            labels.append(batch[b][2])
            
    images = torch.stack(images, 0)
    lms = torch.stack(lms, 0)
    labels = torch.from_numpy(np.array(labels))
    
    return images,lms,labels

def test_collate(batch):
    batch_size = len(batch)
    images = []
    lms = []

    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.append(batch[b][0])
            lms.append(batch[b][1])
            
    images = torch.stack(images, 0)
    lms = torch.stack(lms, 0)
    
    return images,lms
    
class RandomSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        
        for index, sample in enumerate(self.data_source):
            data_info = sample.strip('\n').split()
            label = data_info[1]
            self.index_dic[label].append(index)
            
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
        
class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
        
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(50):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
        
class Dataset(Dataset):
    def __init__(self,data_dir, dataset, mode):
        self.dataset = dataset
        self.data_dir = data_dir
        self.mode = mode 
        self.transform_train = T.Compose([
                                        T.Resize([224,224]),
                                        #T.RandomHorizontalFlip(),
                                       # T.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                        T.ToTensor(),
                                       # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        #RandomErasing(probability=0.4, mean=[0.5, 0.5, 0.5])
                                    ])

        self.transform_test = T.Compose([
                                        T.Resize([224,224]),
                                        T.ToTensor(),
                                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])  
    def __getitem__(self, index):
        data_info = self.dataset[index]
        data_info = data_info.strip('\n').split()
        
        img_path = data_info[0]
        label = data_info[1]
        landmarks = data_info[2:]
        
        img = Image.open(img_path).convert('RGB')
        #img = Image.open(img_path).convert('L').convert('RGB')
        if(self.mode == 'train'):
            img = self.transform_train(img)
        elif(self.mode == 'val'):
            img = self.transform_test(img)
        else:
            print('Please use in Train or Val mode !')
         
        label = class2label[label]

        landmarks = np.array(list(map(float, landmarks)), dtype=np.float32)
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return img, landmarks, label
        
    def __len__(self):
        return len(self.dataset)

class TestDataset(Dataset):
    def __init__(self,data_file):
        file = open(data_file, 'r')
        self.dataset = []
        for line in file.readlines():
            self.dataset.append(line)
        
        self.transform_test = T.Compose([
                                        T.Resize([224,224]),
                                        T.ToTensor(),
                                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])  
    def __getitem__(self, index):
        data_info = self.dataset[index]
        data_info = data_info.strip('\n').split()
        
        img_path = data_info[0]
        landmarks = data_info[1:]
        
        img = Image.open(img_path).convert('RGB')
        #img = Image.open(img_path).convert('L').convert('RGB')
        img = self.transform_test(img)
        
        landmarks = np.array(list(map(float, landmarks)), dtype=np.float32)
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return img, landmarks
        
    def __len__(self):
        return len(self.dataset)
        
def make_dataloader(kd_id, kd_num):
    data_dir = './faces_224'
    file = open('./faces_224/anns/train_ld_shuffle.txt', 'r')
    class_map = defaultdict(list)
    for line in file.readlines():
        data_info = line.split()
        im_path = data_info[0]
        img_name = im_path.split('/')[-1]
        class_map[img_name.split('_')[1]].append(line)
        
    person_names = []
    for key in class_map.keys():
        person_names.append(key)

    length = len(person_names)
    all_name_kds = []
    for i in range(kd_num):
        all_name_kds.append(person_names[math.floor(i / kd_num * length): math.floor((i + 1) / kd_num * length)])

    val_kd = all_name_kds[kd_id]
    
    trains = []
    vals = []
    
    for key in class_map.keys():
        if(key in val_kd):
            vals.extend(class_map[key])
        else:
            trains.extend(class_map[key])

### pusdo label
    # file = open('./faces_224/anns/train_ld_psdo.txt', 'r')
    # for line in file.readlines():
        # data_info = line.split()
        # trains.append(line)
        
    # print(len(trains))
    train_data = Dataset(data_dir, trains, mode='train')
    val_data = Dataset(data_dir, vals, mode='val')
    return train_data, val_data, trains, vals

if __name__ == '__main__':
    train_data, val_data, trains, vals = make_dataloader(1,7)
    train_loader = DataLoader(dataset=train_data, batch_size=4,sampler=RandomSampler(trains, 4, 2), shuffle=False, num_workers=2, collate_fn=train_collate)
    val_loader = DataLoader(dataset=val_data, batch_size=2, shuffle=False, num_workers=2, collate_fn=train_collate )
    
    for images,lms,labels in train_loader:
        print(images.size(),lms.size(),labels)

#####
    #train_data, val_data, trains, vals = make_dataloader(1,7)
    #train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=False, num_workers=2, collate_fn=train_collate)
    #val_loader   = DataLoader(dataset=val_data,   batch_size=4, shuffle=False, num_workers=2, collate_fn=train_collate )
    
    #for images,lms,labels in train_loader:
    #    print(images.size(),lms.size(),labels)

#####
    #test_data = TestDataset('./faces_224/anns/test_ld.txt')
    #test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2, collate_fn=test_collate )
    #for images, lms in test_loader:
    #    print(images.size(), lms.size())
    #print(len(test_loader))
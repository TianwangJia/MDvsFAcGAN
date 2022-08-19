import glob
import os
# from types import NoneType
import torch
import re

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 构建Image数据集
class InfraredImageDataset(Dataset):
    def __init__(self, train_path, test_path, org_transform=None, gt_transfrom=None, mode='train', test_data= 'MDvsFA'):
        '''
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
            org_transform: 红外图片transform
            gt_transform: 小目标显著性gt transfrom
            mode: 训练或测试模式
            test_data: 测试数据集选择, 仅在mode='test'生效
                MDvsFA: MDvsFA测试集, 100张
                Sirst: Sirst测试集, 427张 (Sirst数据与MDvsFA训练数据不一致, 注意转换)
                All: 两个测试集结合, 527张 
        '''
        self.org_transform = transforms.Compose(org_transform)
        self.gt_transform = transforms.Compose(gt_transfrom)
        self.mode = mode
        self.test_data = test_data

        self.train_org_files = sorted(glob.glob(train_path+'/*_1.png'))
        self.train_gt_files = sorted(glob.glob(train_path+'/*_2.png'))

        self.test_org_files_1 = sorted(glob.glob(test_path+'/MDvsFA_test/test_org/*.png'))
        self.test_gt_files_1 = sorted(glob.glob(test_path+'/MDvsFA_test/test_gt/*.png'))
        self.test_org_files_2 = sorted(glob.glob(test_path+'/Sirst_test/images/*.png'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        self.test_gt_files_2 = sorted(glob.glob(test_path+'/Sirst_test/gts/*.png'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        self.test_org_files = self.test_org_files_1 + self.test_org_files_2
        self.test_gt_files = self.test_gt_files_1+self.test_gt_files_2

    def __getitem__(self, index):
        if self.mode == 'train':
            org_imgs = Image.open(self.train_org_files[index % len(self.train_org_files)])
            gt_imgs = Image.open(self.train_gt_files[index %  len(self.train_gt_files)])
        elif self.mode == 'test':
            if self.test_data == 'MDvsFA':
                org_imgs = Image.open(self.test_org_files_1[index % len(self.test_org_files_1)])
                gt_imgs = Image.open(self.test_gt_files_1[index % len(self.test_gt_files_1)])
            elif self.test_data == 'Sirst':
                org_imgs = Image.open(self.test_org_files_2[index % len(self.test_org_files_2)])
                gt_imgs = Image.open(self.test_gt_files_2[index % len(self.test_gt_files_2)])
            elif self.test_data == 'All':
                org_imgs = Image.open(self.test_org_files[index % len(self.test_org_files)])
                gt_imgs = Image.open(self.test_gt_files[index % len(self.test_gt_files)])
            else: 
                raise NotImplementedError
        else: 
            raise NotImplementedError
        
        img_org = self.org_transform(org_imgs)
        img_gt = self.gt_transform(gt_imgs) 

        return {'org':img_org, 'gt':img_gt}


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_org_files)
        
        elif self.mode == 'test':
            if self.test_data == 'MDvsFA':
                return len(self.test_org_files_1)
            elif self.test_data == 'Sirst':
                return len(self.test_org_files_2)
            elif self.test_data == 'All':
                return len(self.test_org_files)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

if __name__ == '__main__':
    
    train_path = '/root/autodl-tmp/data/MDvsFA_train'
    test_path = '/root/autodl-tmp/data/test'

    train_data = InfraredImageDataset(train_path, test_path, mode='train')
    test_data = InfraredImageDataset(train_path, test_path, mode='test', test_data='MDvsFA')
    test_data_2 = InfraredImageDataset(train_path, test_path, mode='test', test_data='Sirst')
    test_data_3 = InfraredImageDataset(train_path, test_path, mode='test', test_data='All')

    print(len(train_data.train_org_files))
    print(len(train_data.train_gt_files))
    print(len(train_data.test_org_files))
    print(len(train_data.test_gt_files))
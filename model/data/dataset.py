import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption
import os,glob

import torch
import numpy as np

# from ram import get_transform_img

# direactions = {'left', 'right', 'above', 'below', 'upper left', 'upper right', 'bottom right' ,'bottom left'}

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, class_num = 4585, root = ''): 

        self.ann = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann += ann
            
        self.transform = transform
        self.class_num = class_num
        self.root = root

    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]   

        image_path_use = os.path.join(self.root, ann['image_path'])
        image = Image.open(image_path_use).convert('RGB')   
        image = self.transform(image)

        # required for tag2text support
        if ann.get('union_label_id') is not None:
            num = ann['union_label_id'] 
            image_tag = np.zeros([self.class_num]) 
            image_tag[num] = 1 
            image_tag = torch.tensor(image_tag, dtype = torch.long)
        else:
            image_tag = None

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index],100)

        nums = ann['parse_label_id'][caption_index]
        tag_num = []

        parse_tag = np.zeros([self.class_num])
        parse_tag[tag_num] = 1
        parse_tag = torch.tensor(parse_tag, dtype = torch.long)

        return image, caption, image_tag, parse_tag
    

class finetune_dataset(Dataset):
    def __init__(self, ann_file, transform, transform_224, class_num = 3662, root = ''): 

        self.ann = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann += ann
            
        self.transform = transform
        self.transform_224 = transform_224
        self.class_num = class_num
        self.root = root

    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]

        image_path_use = os.path.join(self.root, ann['image_path'])
        image_ori = Image.open(image_path_use).convert('RGB')
        image_shape = image_ori.size
        # print('image origin size:',image.size)   
        image = self.transform(image_ori)


        image_224 = Image.open(image_path_use).convert('RGB')  
        image_224 = self.transform_224(image_224)

        # required for tag2text support
        if ann.get('union_label_id') is not None:
            num = ann['union_label_id'] 
            image_tag = np.zeros([self.class_num]) 
            image_tag[num] = 1 
            image_tag = torch.tensor(image_tag, dtype = torch.long)
        else:
            # image_tag = None
            image_tag = []

        caption_index = np.random.randint(0, len(ann['caption']))

        caption = pre_caption(ann['caption'][caption_index],100)

        num = ann['parse_label_id'][caption_index]
        # num = ann['parse_label_id']
        
        parse_tag = np.zeros([self.class_num])
        parse_tag[num] = 1
        parse_tag = torch.tensor(parse_tag, dtype = torch.long)

        return image, image_224, caption, image_tag, parse_tag, np.array(image_shape)
    

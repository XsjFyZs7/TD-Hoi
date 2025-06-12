import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import json  # 添加缺少的json导入

def build_place(config):
    # 从 JSON 文件加载配置
    with open(config, 'r') as f:
        args = json.load(f)

    assert args['arch'] in ['resnet18', 'resnet50'], args['arch']
    
    if args['arch'] == 'resnet50':
        model_file = '/home/bit118/wangyuhan/places365/resnet50_places365.pth.tar'
    else:
        model_file = 'resnet18_places365.pth.tar'
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    if 'file_name' in args and args['file_name'] is not None:
        classes = list()
        with open(args['file_name']) as class_file:
            for line in class_file:  # 修复：缩进错误
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)
    else:
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
        # 修复：添加classes的定义
        classes = []
        with open('categories_places365.txt') as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

    model = models.__dict__[args['arch']](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)  # 移除weights_only参数
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, classes
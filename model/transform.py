'''
 * Based on tag2text code base
 * https://github.com/xinyu1205/recognize-anything
'''

import torch
from torch.autograd import Variable as V

from torchvision.transforms import Normalize, Compose, Resize, ToTensor, CenterCrop
import torchvision
from PIL import Image
import cv2
import numpy as np

def convert_to_rgb(image):
    return image.convert("RGB")

def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def read_cv2_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hh, img_ww = img.shape[0:2]
    return np.array(img), (img_hh, img_ww)

def resize_ensure_shortest_edge(img, size, max_size):
    def get_size_with_aspect_ratio(img_size, _size, _max_size=None):
        h, w = img_size
        if _max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * _size > _max_size:
                _size = int(round(_max_size * min_original_size / max_original_size))
        if (w <= h and w == _size) or (h <= w and h == _size):
            return h, w
        if w < h:
            ow = _size
            oh = int(_size * h / w)
        else:
            oh = _size
            ow = int(_size * w / h)
        return ow, oh

    rescale_size = get_size_with_aspect_ratio(img_size=img.shape[0:2], _size=size, _max_size=max_size)
    img_rescale = cv2.resize(img, rescale_size)
    return img_rescale

def prepare_cv2_image4nn(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
    img = torchvision.transforms.functional.to_tensor(img)
    img_tensor = torchvision.transforms.functional.normalize(img, mean=mean, std=std)
    return img_tensor

def get_transform_img(img_path):
    
    img, img_size = read_cv2_image(img_path=img_path)
    img_rescale = resize_ensure_shortest_edge(img=img, size=672, max_size=1333)
    img_tensor = prepare_cv2_image4nn(img=img_rescale)

    return img_tensor, img_size

def get_transform_place(img_path):
    
    centre_crop = Compose([
        Resize((256,256)),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path)
    place_img = V(centre_crop(img).unsqueeze(0))

    return place_img

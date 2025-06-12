'''
 * Based on tag2text and hoitr code base
 * https://github.com/xinyu1205/recognize-anything
 * https://github.com/kakaobrain/hotr
 * Writting by Yuhan Wang
 * Final update 2025.6.12
'''

import numpy as np
import json
import torch
import warnings

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .bert import BertConfig, BertModel, BertLMHeadModel
from .position_encoding import build_position_encoding
from .hoitr import build_HOITR
from .backbone import build_backbone
from .vcoco import hoi_interaction_names as hoi_interaction_names_vcoco
from .vcoco import coco_instance_ID_to_name as coco_instance_ID_to_name_vcoco
from ..utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .utils import *
from copy import deepcopy

warnings.filterwarnings("ignore")

class tag_proj(nn.Module):
    def __init__(self):
        super(tag_proj, self).__init__()

        self.proj = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(256, 1024)),
                ("GELU", nn.GELU()),
                ("linear2", nn.Linear(1024, 1024))
            ]))
    
    def forward(self, input):
        with torch.cuda.amp.autocast():
             output = self.proj(input)
        return output

class Tag2Text(nn.Module):

    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=224,
                 text_encoder_type='bert-base-uncased',
                 prompt='a picture of ',
                 threshold=0.6,
                 delete_tag_index=[127,2961, 3351, 3265, 3338, 3355, 3359],
                 tag_list=f'{CONFIG_PATH}/data/HOI2A_tag_list.txt',
                 stage='eval'):
        super().__init__()
        # create tokenzier
        self.tokenizer = init_tokenizer(text_encoder_type)
        # HOI2A employ encoder-decoder architecture for image-tag-text generation: hoitr encoder and image-tag-text decoder
        # create hoitr encoder
        model, criterion = build_HOITR(config = '/home/bit118/wangyuhan/Index_HOIP2A/HOI2A/configs/hoitr_config.json')
        self.tagging_head = model
        self.load_and_freeze_hoitr_model('/home/bit118/wangyuhan/Index_HOIP2A/pretrain_models/HOITR/resnet50_hotr_image224.pth')
        # create image-tag-text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        vision_width = 1024
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.tag_encoder = BertModel(config=encoder_config,
                                     add_pooling_layer=False)

        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back"; 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)
        
        # create projecter from 256 to 1024
        self.imageto1024 = tag_proj()
        
        self.threshold = threshold
        self.num_class = len(self.tag_list)

        tag_thrshold = {2701:0.7, 2828: 0.7, 1167: 0.7}
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        for key,value in tag_thrshold.items():
            self.class_threshold[key] = value
        
    def load_and_freeze_hoitr_model(self, hoitr_model_path):
        """
        loading pretrained HoiTR mod and freezing parameters
        """
        print("Loading Tag2Text HoiTR pretrained model from ", hoitr_model_path)
        
        # load HoiTR parameters
        HoiTR_pretrain_model = hoitr_model_path
        pretrain_dict = torch.load(HoiTR_pretrain_model, map_location='cpu', weights_only=False)['model']
        self.tagging_head.load_state_dict(pretrain_dict)
        
        # freezing parameters
        for param in self.tagging_head.parameters():
            param.requires_grad = False
            
        print("Pretrained HoiTR model loaded and parameters frozen.")
    
    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r') as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    def triplet_nms(self, hoi_list):
        hoi_list.sort(key=lambda x: x['h_cls'] * x['o_cls'] * x['i_cls'], reverse=True)
        mask = [True] * len(hoi_list)
        for idx_x in range(len(hoi_list)):
            if mask[idx_x] is False:
                continue
            for idx_y in range(idx_x+1, len(hoi_list)):
                x = hoi_list[idx_x]
                y = hoi_list[idx_y]
                if x['i_name'] == y['i_name'] and x['o_name'] == y['o_name']:
                    mask[idx_y] = False
        new_hoi_list = []
        for idx in range(len(mask)):
            if mask[idx] is True:
                new_hoi_list.append(hoi_list[idx])
        return new_hoi_list
    
    def dict2tensor(self, samples, dict, image_ori_size, hoi_th=0, human_th=0, object_th=0, max_to_viz=10, HOI_res=False):
        
        hoi_th = self.threshold
        human_th = self.threshold
        object_th = self.threshold
        num_classes = 112
        num_actions = 44
        top_k = 35
        hoi_interaction_names = hoi_interaction_names_vcoco
        coco_instance_id_to_name = coco_instance_ID_to_name_vcoco
        final_hoi_result_list = []
        
        id_list = [idx for idx in range(len(samples))]
        action_pred_logits = dict['action_pred_logits']
        object_pred_logits = dict['object_pred_logits']
        object_pred_boxes = dict['object_pred_boxes']
        human_pred_logits = dict['human_pred_logits']
        human_pred_boxes = dict['human_pred_boxes']

        for idx in range(len(action_pred_logits)):
            hh, ww = image_ori_size[idx][0], image_ori_size[idx][1]

            act_cls = torch.nn.Softmax(dim=1)(action_pred_logits[idx]).detach().cpu().numpy()[:, :-1]
            human_cls = torch.nn.Softmax(dim=1)(human_pred_logits[idx]).detach().cpu().numpy()[:, :-1]
            object_cls = torch.nn.Softmax(dim=1)(object_pred_logits[idx]).detach().cpu().numpy()[:, :-1]
            human_box = human_pred_boxes[idx].detach().cpu().numpy()
            object_box = object_pred_boxes[idx].detach().cpu().numpy()

            keep = (act_cls.argmax(axis=1) != num_actions)
            keep = keep * (human_cls.argmax(axis=1) != 2)
            keep = keep * (object_cls.argmax(axis=1) != num_classes)
            keep = keep * (act_cls > hoi_th).any(axis=1)

            keep = keep * (human_cls > human_th).any(axis=1)
            keep = keep * (object_cls > object_th).any(axis=1)

            human_idx_max_list = human_cls[keep].argmax(axis=1)
            human_val_max_list = human_cls[keep].max(axis=1)
            human_box_max_list = human_box[keep]
            object_idx_max_list = object_cls[keep].argmax(axis=1)
            object_val_max_list = object_cls[keep].max(axis=1)
            object_box_max_list = object_box[keep]
            keep_act_scores = act_cls[keep]

            keep_act_scores_1d = keep_act_scores.reshape(-1)
            top_k_idx_1d = np.argsort(-keep_act_scores_1d)[:top_k]
            box_action_pairs = [(idx_1d // num_actions, idx_1d % num_actions) for idx_1d in top_k_idx_1d]

            hoi_list = []
            for idx_box, idx_action in box_action_pairs:
                # action
                i_cls = keep_act_scores[idx_box, idx_action]
                i_name = hoi_interaction_names[int(idx_action)]
                if i_name in ['__background__']:
                    continue
                # human
                cid = human_idx_max_list[idx_box]
                cx, cy, w, h = human_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                h_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                h_cls = human_val_max_list[idx_box]
                h_name = coco_instance_id_to_name[int(cid)]
                # object
                cid = object_idx_max_list[idx_box]
                cx, cy, w, h = object_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                o_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                o_cls = object_val_max_list[idx_box]
                o_name = coco_instance_id_to_name[int(cid)]
                
                if i_cls < hoi_th or h_cls < human_th or o_cls < object_th:
                    continue
                
                pp = {
                    'h_box': h_box, 
                    'o_box': o_box, 
                    'h_cls': h_cls, 
                    'o_cls': o_cls, 
                    'i_cls': i_cls,
                    'h_name': h_name, 
                    'o_name': o_name, 
                    'i_name': i_name,
                }
                hoi_list.append(pp)
                
            new_hoi_list = self.triplet_nms(hoi_list)
            item = {
                'image_id': id_list, 
                'hoi_list': new_hoi_list
            }
            final_hoi_result_list.append(item)
        # hoitr中tag与tag2text中tag进行对齐
        res_list = self.tagalignment(final_hoi_result_list, tag2text_tag_list_path = '/home/bit118/wangyuhan/Tag2Text/ram/data/tag2text_ori_tag_list.txt')
        # 将res_list转换为tensor
        res_tensor, res_bbox = self.gen_tensor(res_list, self.num_class)

        if HOI_res == True:
            return res_tensor, res_bbox, final_hoi_result_list

        return res_tensor, res_bbox
    
    def tagalignment(self, items, tag2text_tag_list_path):
        # 初始化一个字典来保存映射
        tag_to_index = {}
        tag_list = []
    
        # 从文件中读取标签及其序号
        with open(tag2text_tag_list_path, 'r') as file:
            for line in file:
                tag = line.strip()
                tag_list.append(tag)
            
        # 初始化新的列表以保存序号
        joi_num_list = []
        
        # 提取 image_id 和 hoi_list
        for item in items:  # 假设 items 是一个包含多个字典的列表
            image_id = item['image_id']  # 获取当前字典中的 image_id
            hoi_list = item['hoi_list']  # 获取当前字典中的 hoi_list

            # 遍历 hoi_list，并根据 tag 查找对应的序号
            for hoi in hoi_list:
                h_name = hoi['h_name']
                o_name = hoi['o_name']
                i_name = hoi['i_name']
                h_cls = hoi['h_cls']
                o_cls = hoi['o_cls']
                i_cls = hoi['i_cls']
                h_box = hoi['h_box']
                o_box = hoi['o_box']

                if '_instr' in i_name:
                    i_name = i_name.replace('_instr', '')
                if '_obj' in i_name:
                    i_name = i_name.replace('_obj', '')
                if 'work_on_computer' in i_name:
                    i_name = 'work'
                if 'talk_on_phone' in i_name:
                    i_name = 'talk'
                if 'hair drier' in o_name or 'skis' in o_name or 'hot dog' in o_name or 'scissors' in o_name:
                    o_name = ''
                if 'potted plant' in o_name:
                    o_name = 'plant pot'
                if 'sports ball' in o_name:
                    o_name = 'ball'

                h_index = tag_list.index(h_name) if h_name in tag_list else -1
                o_index = tag_list.index(o_name) if o_name in tag_list else -1
                i_index = tag_list.index(i_name) if i_name in tag_list else -1

                index = {
                    'h_cls': h_cls, 
                    'o_cls': o_cls, 
                    'i_cls': i_cls,
                    'h_index': h_index, 
                    'o_index': o_index, 
                    'i_index': i_index,
                    'h_box': h_box, 
                    'o_box': o_box,
                }
        
                joi_num_list.append(index)
    
        # 构建新的字典
        item_num = {
            'image_id': image_id, 
            'hoi_list': joi_num_list
        }
        return item_num
    
    def gen_tensor(self, res_list, num_labels):
        res_tensor = np.random.uniform(-0.001, 0.001, (len(res_list['image_id']), num_labels))
        res_tensor = res_tensor.astype(np.float32)

        bbox_list = [[0] * num_labels for _ in range(len(res_list['image_id']))]

        idx = res_list['image_id']
        hoi_list = res_list['hoi_list']
    
        tags_classes = {}
        tags_bboxes = {}
        for hoi in hoi_list:
            h_tag_idx = hoi['h_index']
            o_tag_idx = hoi['o_index']
            i_tag_idx = hoi['i_index']

            h_cls = hoi['h_cls']
            o_cls = hoi['o_cls']
            i_cls = hoi['i_cls']

            h_box = hoi['h_box']
            o_box = hoi['o_box']
        
            # 将收集的标签索引及其对应的类添加到字典中
            tags_classes[h_tag_idx] = h_cls
            tags_classes[o_tag_idx] = o_cls
            tags_classes[i_tag_idx] = i_cls

            tags_bboxes[h_tag_idx] = h_box
            tags_bboxes[o_tag_idx] = o_box
    
        # 将 tags 中的对应位置标记为 cls_value
        for id in range(len(idx)):
            for tag_idx, cls_value in tags_classes.items():
                if res_tensor[id, tag_idx] < cls_value:
                    res_tensor[id, tag_idx] = cls_value
        for id in range(len(idx)):
            for tag_idx, bbox in tags_bboxes.items():
                bbox_list[id][tag_idx] = bbox
            
        return torch.tensor(res_tensor), bbox_list

    def forward(self, samples, caption, tag, image_ori_size):
        """
        call function as forward

        Args:
            samples: type: tensor_list: NestedTensor
            caption: type: list[string]  len: batch_size
            tag: type: torch.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0

        Returns:
            loss: type: torch.Tensor
        """
        ##================= Image Tagging ================##
        trans_image_embeds = self.tagging_head.get_hoitr_embeds(samples).permute(1, 3, 0, 2)
        trans_image_embeds = trans_image_embeds.flatten(2).permute(0, 2, 1)
        image_embeds  = self.imageto1024(trans_image_embeds)
        image_embeds = image_embeds.to(torch.float32)

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(samples.device)
        ##================= Image-Tag-Text Generation ================##
        bs = image_embeds.shape[0]
        tag = tag.cpu().numpy()
        tag_input = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_input.append(' | '.join(token))

        tag_input_tokenzier = self.tokenizer(tag_input,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=40,
                                             return_tensors="pt").to(
                                                 samples.device)
        encoder_input_ids = tag_input_tokenzier.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text = self.tokenizer(caption,
                              padding='longest',
                              truncation=True,
                              max_length=60,
                                return_tensors="pt").to(
                                    samples.device)
        
        decoder_input_ids = text.input_ids
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id

        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100
        
        decoder_output = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = output_tagembedding.last_hidden_state,
                                           encoder_attention_mask = None,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        
        loss_t2t = decoder_output.loss

        return loss_t2t


    def generate(self,
                 hoi_samples,
                 sample=False,
                 num_beams=3,
                 max_length=60,
                 min_length=10,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 tag_input=None,
                 return_tag_predict=False,
                 image_ori_size=[]):
                 
        trans_image_embeds = self.tagging_head.get_hoitr_embeds(hoi_samples).permute(1, 3, 0, 2)
        trans_image_embeds = trans_image_embeds.flatten(2).permute(0, 2, 1)
        image_embeds  = self.imageto1024(trans_image_embeds)
        image_embeds = image_embeds.to(torch.float32)

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(hoi_samples.device)
        if tag_input == None:

            bs = image_embeds.shape[0]
            
            hoi_res = self.tagging_head(hoi_samples)

            hoi_tensor, res_bbox, hoi_list= self.dict2tensor(hoi_samples, hoi_res, image_ori_size, HOI_res = True)
            targets = torch.where(
                torch.sigmoid(hoi_tensor.to(hoi_samples.device)) > self.class_threshold.to(hoi_samples.device),
                torch.tensor(1.0).to(hoi_samples.device),
                torch.zeros(self.num_class).to(hoi_samples.device))
            
            tag = targets.cpu().numpy()

            # delete some tags that may disturb captioning
            tag[:, self.delete_tag_index] = 0

            tag_input = []
            for b in range(bs):
                index = np.argwhere(tag[b] == 1)
                token = self.tag_list[index].squeeze(axis=1)
                tag_input.append(' | '.join(token))
            
        tag_output = tag_input

        # beam search for text generation(default)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            tag_input_temp = []
            for tag in tag_input:
                for i in range(num_beams):
                    tag_input_temp.append(tag)
            tag_input = tag_input_temp

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(hoi_samples.device)

        # tokenizer input tags
        tag_input_tokenzier = self.tokenizer(tag_input,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=40,
                                             return_tensors="pt").to(
                                                 hoi_samples.device)
        encoder_input_ids = tag_input_tokenzier.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # put input tag into image-tag interaction encoder to interact with image embeddings
        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # prompt trick for better captioning, followed BLIP
        prompt = [self.prompt] * hoi_samples.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            hoi_samples.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search (default)
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        if return_tag_predict == True:
            return  captions, tag_output
        return captions

# load HOI2A pretrained model parameters
def HOI_tag2text(pretrained='', **kwargs):
    model = Tag2Text(**kwargs)
    state_dict = deepcopy(model.tagging_head.state_dict())
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        state_dict = deepcopy(model.tagging_head.state_dict())
    
    return model
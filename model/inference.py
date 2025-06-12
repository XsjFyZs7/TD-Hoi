'''
 * Based on tag2text code base
 * https://github.com/xinyu1205/recognize-anything
'''
import torch

def inference_tag2text(image, model, image_shape):
    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True,
                                              image_ori_size=image_shape)

    return tag_predict[0], None, caption[0]

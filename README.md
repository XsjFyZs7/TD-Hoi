# Hoi2Threat

An Interpretable Threat Detection Method for Human Violence Scenarios Guided by Human-Object Interaction

## Model Architecture

![example](./imgs/图3.jpg)

> Overview of Hoi2Threat：
> - Hoi2Threat takes visual information as input and generates interpretations of detected anomalous events as output.

## Inference Hoi2Threat

```bash
python scripts/inference_hoi2t.py \
    --image /path/to/your/image.jpg \
    --pretrained /path/to/pretrained/model.pth \
    --image-size 224 \
    --thre 0.6


# RE-CTFNet

**RE-CTFNet (Restoration-Enhanced Cross-Task Fusion Network)** is an adverse-weather object detection framework. It combines a restoration branch and a YOLOX-based detection branch through the **Restoration-Enhanced Cross-Task Fusion (RE-CTF) module**, which transfers useful restoration features to improve object detection.

## 1. Setup

```bash
git clone https://github.com/KomanMorungman/RE-CTFNet.git
cd RE-CTFNet
pip install -r requirements.txt
```

## 2. Datasets

Download the datasets and follow the preparation instructions in the [RDMNet repository](https://github.com/xfwang23/RDMNet). Prepare the annotation lists and class file, then update the paths for your environment.

## 3. Train

Training uses paired degraded/clean images and RDMNet-format annotations. By default, the detection branch is initialized using pretrained YOLOX-S weights.

```bash
python train.py \
    --train-list "path/to/train.txt" \
    --val-list "path/to/val.txt" \
    --classes "path/to/classes.txt" \
    --image-root "path/to/Dataset" \
    --pretrained-detection "path/to/yolox_s.pth" \
    --epochs 100 \
    --batch-size 8 \
    --workers 2 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --amp fp16 \
    --output-dir "path/to/checkpoints"
```

Download [YOLOX-S checkpoint](https://drive.google.com/file/d/1mUSuZTmKjaYt6U-tRU3xOiUIz-E3wFJC/view?usp=sharing). To train without YOLOX-S initialization, use `--no-pretrained-detection` instead. Use `--resume "path/to/checkpoint.pth"` to resume training.

## 4. Evaluate

Evaluate a trained checkpoint using an RDMNet-format test annotation file:

```bash
python evaluation.py \
    --checkpoint "path/to/best.pth" \
    --annotations "path/to/test.txt" \
    --image-root "path/to/Dataset" \
    --classes "path/to/classes.txt" \
    --output-dir "path/to/results"
```

The evaluation script reports per-class AP and mAP@0.5, and can save detection visualisations, restored images, and metric files. 

## 5. Run inference

A trained checkpoint is available on [Google Drive](https://drive.google.com/file/d/1Wz9UTC6DPF0c_dvTC_p-a-pzyHEsnE8z/view?usp=sharing).


```bash
python predict.py \
    --checkpoint "path/to/checkpoint.pth" \
    --image "path/to/input_image.jpg"
```

The script loads the full RE-CTFNet architecture, including RE-CTF fusion, and displays the detected image with bounding boxes, class labels, and confidence scores.

## Acknowledgement

I would like to thank the authors and contributors of the datasets used in this work, including the synthetic haze and real-world hazy image datasets. I also acknowledge the researchers behind RDMNet, YOLOX, and HOGformer, whose work served as a basis for the proposed RE-CTFNet framework.

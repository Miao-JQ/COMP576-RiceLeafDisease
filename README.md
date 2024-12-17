# COMP576 RiceDisease

## Data Preparation

Download dataset and see dataset description at https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image/data

## Train

for efficientnet-b3

```
python main.py --model_name=efficientnet --efficient_name=efficientnet-b3 --lr=0.0003 --weight_decay=0.002
```

for resnet-50:

```
python main.py --model_name=resnet50 --lr=0.0001 --weight_decay=0.002
```

for vgg16:

```
python main.py --model_name=vgg16 --lr=0.0001 --weight_decay=0.002
```

## Prediction

To classify an image, we can use classify_image.py. For example

```
python classify_image.py --input_path=[IMAGE YOU WANT TO CLASSIFY] --ckpt_path=best_model.pth
```


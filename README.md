# COMP576 RiceDisease

## Data Preparation

Download dataset and see dataset description at https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image/data

## Train

for efficientnet-b3

```
python main.py --efficient_name=efficientnet-b3 --lr=0.0003 --weight_decay=0.002
```

for resnet-50:

```
python main.py --efficient_name=efficientnet-b3 --lr=0.0001 --weight_decay=0.002
```

for vgg16:

```
python main.py --efficient_name=efficientnet-b3 --lr=0.0001 --weight_decay=0.002
```

## Results

### initial acc

efficient-b3: 0.1997

resnet-50: 0.3345

vgg16: 0.3016

### acc

efficient-b3: 0.9975

resnet-50: 1.0000

vgg16: 0.9857

### MFLOPS

efficient-b3: 56.921924

resnet-50: 7704.627972

vgg16: 27326.40282

### data augmentation

with aug: 0.9975

without aug: 0.9857
# Dropout Continual Semantic Segmentation

## Master's thesis

**DCSS: Dropout Continual Semantic Segmentation** <br />
Maciej Kowalski<sup>1</sup><br>

<sup>1</sup> <sub>School of Informatics, University of Edinbburgh</sub><br />

# Abtract

# Experimental Results (mIoU all)

|  Method     | VOC 10-1 (11 tasks) | VOC 15-1 (6 tasks) | VOC 5-3 (6 tasks) | VOC 19-1 (2 tasks) | VOC 15-5 (2 tasks) | VOC 5-1 (16 tasks) | 
| :--------:  | :-----------------: | :--------------:   | :---------------: | :----------------: | :----------------: |:----------------: |
| MiB   |         12.65       |  29.29      |   46.71  |  69.15     |   70.08 | 10.03 |
| PLOP  |         30.45       |  54.64      |   18.68  |  73.54     |   70.09 | 6.46 |
| SSUL  |         59.25       |  67.61     |   52.75  |  **75.44**    |   71.22 | 48.65 |
| **DCSS**  |       **62.32**      |  **69.33**      |  **54.34**  |  75.30     |  **71.30** | **49.05** |

# Getting Started

### Requirements
- torch>=1.7.1
- torchvision>=0.8.2
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib
- tensorboardX
- ptflops
- tqdm


### Datasets
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
    --- ADEChallengeData2016
        --- annotations
            --- training
            --- validation
        --- images
            --- training
            --- validation
```

Download [SegmentationClassAug](https://drive.google.com/file/d/17ylg3RHZCQRyGVk6rcmkAjcMi6jeuXLr/view?usp=sharing) and [saliency_map](https://drive.google.com/file/d/1NDPBKbg5aoCismuU9R_IJ9cJp5ncww-M/view?usp=sharing)

### Class-Incremental Segmentation Segmentation on VOC 2012

```
DATASET=voc
SEED=1
TASK=15-1
EPOCH=50
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=0
GPU_ID=0,1,2,3
GPUS=$((${#GPU_ID}/2 + 1))
CPUS=4
PORT=$((9000 + RANDOM % 1000))
BATCH=$((24/${GPUS}))
CONTINUAL_BATCH=$((24/${GPUS}))
NAME=experiment

DATA_ROOT=~/datasets/VOCdevkit/VOC2012
CHECKPOINTS_ROOT=./checkpoints

OMP_NUM_THREADS=${CPUS} python3 -m torch.distributed.run --master_port ${PORT} --nproc_per_node=${GPUS} main.py  \
    --name ${NAME} --num_worker ${CPUS} --gpu_id ${GPU_ID} \
    --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --crop_val --port 9000 \
    --checkpoints_root ${CHECKPOINTS_ROOT} --lr ${LR} --continual_batch ${CONTINUAL_BATCH} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze --random_seed ${SEED} \
    --unknown --mem_size ${MEMORY} --crop_size 513 \
    --curr_step 0 --amp --shared_head -num_classif_features 2048 \
    --dropout 0.3 - --dropout_type 1d --no-w_transfer \
    --separable_head --scheduled_drop --remove_dropout --lr_schedule 
```

### Qualitative Results

<img src = "https://github.com/mazo20/Dropout-Continual-Semantic-Segmentation/blob/master/figures/examples.png" width="100%" height="100%">

# Acknowledgement

Our implementation is based on these repositories: [SSUL](https://github.com/clovaai/SSUL), [PLOP](https://github.com/arthurdouillard/CVPR2021_PLOP).


# License
```
Copyright (c) Maciej Kowalski 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

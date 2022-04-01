"""
Copyright (c) 2022 Maciej Kowalski.
MIT License
"""

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

SCRATCH_DISK=/disk/scratch_fast
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

CONDA_ENV_NAME=minf
echo "Activating conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME}

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

src_path=~/datasets
dest_path=${SCRATCH_HOME}/datasets/data/input
mkdir -p ${dest_path}  # make it if required

rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

DATA_ROOT=${dest_path}/VOCdevkit/VOC2012
CHECKPOINTS_ROOT=${SCRATCH_HOME}/checkpoints

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
    
"""
Copyright (c) 2022 Maciej Kowalski.
MIT License
"""

#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
import sys
import itertools
import random

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'
DATA_HOME = f'{SCRATCH_HOME}/datasets/data'

GPU_IDS = '0,1,2,3'
CPUS = '4'
GPUS = len(GPU_IDS.replace(',', ''))

config = {
    'checkpoints_root': [f'{SCRATCH_HOME}/checkpoints'],
    'data_root': [f'{DATA_HOME}/input/VOCdevkit/VOC2012'],
    'model': ['deeplabv3_resnet101'],
    'name': ['cluster/tasks'],
    'crop_val': [True],
    'dataset': ['voc'],
    'task': ['15-1', '10-1', '19-1', '15-5', '5-3'],
    'train_epoch': [50],
    'continual_epochs': [50],
    'batch_size': [int(24 / GPUS)],
    'continual_batch': [int(24 / GPUS)],
    'loss_type': ['bce_loss'],
    'lr_policy': ['poly'],
    'lr_schedule': [True],
    'lr': [0.01],
    'pseudo': [True],
    'freeze': [True],
    'bn_freeze': [True],
    'unknown': [True],
    'w_transfer': [False],
    'overlap': [True],
    'curr_step': [0],
    'amp': [True],
    'detached_residual': [False],
    'proj_dim_reduction': ['none'],
    'shared_head': [True],
    'shared_projection': [False],
    'scheduled_drop': [True],
    'dropout': [0.3],
    'dropout_aspp': [0.1],
    'dropout_type': ['1d'],
    'remove_dropout': [True],
    'num_classif_features': [256, 1024, 2048, 4096],
    'separable_head': [True],
    'pseudo_thresh': [0.9],
    'weight_decay': [0.0001],
    'mem_size': [0],
    'crop_size': [513],
    'num_worker': [4],
    'gpu_id': [GPU_IDS],
    
    
}

keys, values = zip(*config.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

nr_expts = len(permutations_dicts)
nr_servers = 10
avg_expt_time = 3  # hours
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {((round(nr_expts / nr_servers) + 1) * avg_expt_time)} hrs')

output_file = open("experiments.txt", "w")

for i, dictionary in enumerate(permutations_dicts):
    PORT = random.randint(9000, 9999)
    call = f"OMP_NUM_THREADS={CPUS} python3 -m torch.distributed.run --nproc_per_node={GPUS} --master_port {PORT} main.py"
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                call += " --" + key
            else:
                call += " --no-" + key
        else:
            call += " --" + key + " " + str(value)
    call += " " + "--random_seed" + " " + str(PORT)
    call += " " + "--port" + " " + str(PORT)
    print(call, file=output_file)
    
output_file.close()
from tqdm import tqdm
import network
import utils
import os
import time
import json
import random
import argparse
import numpy as np
import cv2
import copy

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from datetime import datetime

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced

from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--port", type=int, default=0)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--checkpoints_root", type=str, default='checkpoints')
    parser.add_argument("--logdir", type=str, default='./logs')
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--small_dataset", default=False, action=argparse.BooleanOptionalAction)

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3_resnet50moco', 'deeplabv3plus_resnet50moco'], help='model name')
    parser.add_argument("--separable_conv", default=False, action=argparse.BooleanOptionalAction, help="Operation introduced in DeepLabV3+")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    #DCSS arguments
    parser.add_argument("--shared_head", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_classif_features", type=int, default=256)
    parser.add_argument("--detached_residual", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--low_level_out", type=int, default=24)
    parser.add_argument('--shared_projection', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--proj_dim_reduction", type=str, default='none', choices=['none', 'atrous', 'pooling'])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_type", type=str, default='2d', choices=['1d', '2d'])
    parser.add_argument("--continual_batch", type=int, default=24)
    parser.add_argument("--dropout_aspp", type=float, default=0.1)
    parser.add_argument("--scheduled_drop", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--remove_dropout", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr_schedule", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--continual_epochs", type=int, default=50)
    parser.add_argument("--separable_head", default=True, action=argparse.BooleanOptionalAction, help="Operation introduced in DCSS")
    parser.add_argument("--count_flops", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--ret_samples", default=False, action=argparse.BooleanOptionalAction)

    # Train Options
    parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test_only", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--train_epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", type=int, default=32, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default=None, type=str, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bce_loss', choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--num_workers", type=int, default=4)
    

    # CIL options
    parser.add_argument("--pseudo", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--bn_freeze", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--w_transfer", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--unknown", default=False, action=argparse.BooleanOptionalAction)
    
    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=opts.dataset_mean, std=opts.dataset_std),
    ])
    if opts.crop_val:

        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=opts.dataset_mean, std=opts.dataset_std),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=opts.dataset_mean, std=opts.dataset_std),
        ])
        
    
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    if opts.small_dataset:
        def resize(d):
            l = int(len(d.images) * 0.3)
            d.images = d.images[:l]
            return d
        dataset_dict = {k: resize(v) for k, v in dataset_dict.items()}


    return dataset_dict


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    model.eval()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _, true_labels) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs = model(images)
            
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

            if opts.ret_samples and i in opts.ret_samples:
                
                ret_samples.append((images[0].detach().cpu().numpy(), true_labels[0].cpu().numpy(), targets[0], preds[0]))
                
        metrics.synch(device)
        score = metrics.get_results()

    if opts.ret_samples:
        for k, (img, true_label, target, pred) in enumerate(ret_samples):

            decode = loader.dataset.decode_target
            img = (denorm(img) * 255).astype(np.uint8)
            target = decode(target).transpose(2, 0, 1).astype(np.uint8)
            true_label = decode(true_label).transpose(2, 0, 1).astype(np.uint8)
            pred = decode(pred).transpose(2, 0, 1).astype(np.uint8)

            concat_img = np.concatenate((img, true_label, target, pred), axis=2)
            # logger.add_image(f"Sample_{k}", concat_img, opts.curr_step)
            logger.add_image(f"image_{k}", pred, opts.curr_step)
            logger.add_image(f"img_{k}", img, opts.curr_step)
            logger.add_image(f"target_{k}", target, opts.curr_step)
            
    
    model.train()
    return score, ret_samples

def main(opts):
    if opts.curr_step > 0:
        bn_freeze = opts.bn_freeze
        opts.batch_size = opts.continual_batch
        opts.train_epoch = opts.continual_epochs
        if opts.train_epoch == 1:
            opts.lr_schedule = False
        if opts.remove_dropout:
            opts.dropout = 0.0
            opts.dropout_aspp = 0.0
    else:
        bn_freeze = False

        
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: # re-labeling: [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    
    logger.info("==============================================")
    logger.info(f"  task : {opts.task}")
    logger.info(f"  step : {opts.curr_step}")
    logger.info(f"  opts : {opts}")
    logger.info("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up model

    model = network.init_model(
        name=opts.model,
        num_classes=opts.num_classes, 
        output_stride=opts.output_stride, 
        bn_freeze=bn_freeze, 
        opts=opts)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

        
    if opts.curr_step > 0:
        """ load previous model """
        model_prev = network.init_model(
            name=opts.model,
            num_classes=opts.num_classes if opts.test_only else opts.num_classes[:-1], 
            output_stride=opts.output_stride, 
            bn_freeze=bn_freeze, 
            opts=opts)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None

    
    flops, params = utils.count_flops(model, opts, logger)
    if opts.count_flops:
        return
    
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), len(get_tasks(opts.dataset, opts.task, 0)), opts.dataset)

    # logger.info(model.classifier.head)
    
    # Set up optimizer & parameters
    if opts.freeze and opts.curr_step > 0:
        training_params = []

        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.head[-1].parameters(): # classifier for new class
            param.requires_grad = True

        training_params.append({'params': model.classifier.head[-1].parameters(), 'lr': opts.lr})

        if "plus" in opts.model and not opts.shared_projection:
            for param in model.classifier.project[-1].parameters():
                param.requires_grad = True
            training_params.append({'params': model.classifier.project[-1].parameters(), 'lr': opts.lr})
            
        if opts.unknown:

            for param in model.classifier.head[0].parameters(): # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
            
            for param in model.classifier.head[1].parameters(): # background
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})

            if "plus" in opts.model and not opts.shared_projection:
                for param in model.classifier.project[0].parameters():
                    param.requires_grad = True
                training_params.append({'params': model.classifier.project[0].parameters(), 'lr': opts.lr})

                for param in model.classifier.project[1].parameters():
                    param.requires_grad = True
                training_params.append({'params': model.classifier.project[1].parameters(), 'lr': opts.lr*1e-4})
    else:
        if torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # for param in model.backbone.parameters(): # classifier for new class
        #     param.requires_grad = False

        # training_params = [{'params': model.classifier.parameters(), 'lr': 0.01}]

        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]
        
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=opts.lr, 
                                momentum=0.9, 
                                weight_decay=opts.weight_decay,
                                nesterov=True)

    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        
    
    # Restore
    best_score = -1
    cur_epoch = 0
    
    utils.mkdir(opts.checkpoints_root)
    if opts.overlap:
        ckpt_str = opts.checkpoints_root + "/%s_%s_%s_%d_step_%d_%d_overlap.pth"
    else:
        ckpt_str = opts.checkpoints_root + "/%s_%s_%s_%d_step_%d_%d_disjoint.pth"
    
    if opts.curr_step > 0: # previous step checkpoint
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.crop_size, opts.curr_step if opts.test_only else opts.curr_step-1, opts.port)
        
    if opts.ckpt and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)
        
        if opts.unknown and opts.w_transfer and not opts.test_only:
            curr_head_num = len(model.classifier.head) - 1
            mod_name = "classifier.head"

            if opts.shared_head:
                #If we have new_classes > 1, we have to copy the unknown class' weights to each channel
                for i in range(opts.num_classes[-1]):
                    model.state_dict()[f"{mod_name}.{curr_head_num}.weight"][i] = checkpoint[f"{mod_name}.0.weight"]
                    model.state_dict()[f"{mod_name}.{curr_head_num}.bias"][i] = checkpoint[f"{mod_name}.0.bias"]
            else:
                mod_name = f"classifier.head.0"
                target_mod_name = f"classifier.head.{curr_head_num}"
                for key in copy.deepcopy(list(checkpoint.keys())):
                    if mod_name in key:
                        appendix = key[len(mod_name):]
                        checkpoint[target_mod_name + appendix] = checkpoint[key]

                # for i in range(opts.num_classes[-1]):
                checkpoint[target_mod_name + ".3.weight"] = checkpoint[f"{mod_name}.3.weight"].repeat(opts.num_classes[-1], 1, 1, 1)
                checkpoint[target_mod_name + ".3.bias"] = checkpoint[f"{mod_name}.3.bias"].repeat(opts.num_classes[-1])

                if 'plus' in opts.model:
                    mod_name = f"classifier.project.0"
                    target_mod_name = f"classifier.project.{curr_head_num}"
                    for key in copy.deepcopy(list(checkpoint.keys())):
                        if mod_name in key:
                            appendix = key[len(mod_name):]
                            checkpoint[target_mod_name + appendix] = checkpoint[key]

            
        model.load_state_dict(checkpoint, strict=False)
        
        logger.info("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        logger.info("[!] Retrain")
    

    model = DistributedDataParallel(model.to(device), device_ids=[device_id] if torch.cuda.is_available() else None, find_unused_parameters= not opts.shared_head and opts.curr_step > 0)
    model.train()

    logger.info("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name} {param.shape}")
    logger.info("-----------------------------------------------")
    
    if opts.curr_step > 0:
        model_prev.to(device)
        model_prev.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)
            
        # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], 
        batch_size=opts.batch_size, 
        sampler=DistributedSampler(dataset_dict['train'], num_replicas=world_size, rank=rank),
        num_workers=opts.num_workers, 
        pin_memory=True, 
        drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], 
        batch_size=opts.val_batch_size, 
        sampler=DistributedSampler(dataset_dict['val'], num_replicas=world_size, rank=rank, shuffle=False),  
        num_workers=opts.num_workers, 
        pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], 
        batch_size=opts.val_batch_size, 
        sampler=DistributedSampler(dataset_dict['test'], num_replicas=world_size, rank=rank, shuffle=False),
        num_workers=opts.num_workers, 
        pin_memory=True)

    #==========   Test Only   ==========#
    if opts.test_only:
        test_score, ret_samples = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        logger.info(metrics.to_str(test_score))
        return
    
    logger.info("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'],
            batch_size=opts.batch_size,
            sampler=DistributedSampler(dataset_dict['memory'], num_replicas=world_size, rank=rank),
            num_workers=opts.num_workers,
            pin_memory=True,
            drop_last=True)
    
    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = len(train_loader)
    logger.info(f"... train epoch : {opts.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")
        
    #==========   Train Loop   ==========#

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                           reduction='mean')
        
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    avg_loss = AverageMeter()
    
    model.train()
    
    # =====  Train  =====
    if rank == 0:
        pbar = tqdm(range(1, total_itrs + 1))
    else:
        pbar = range(1, total_itrs + 1)

    for cur_itrs in pbar:
        optimizer.zero_grad()

        if opts.scheduled_drop:
            model.module.updateDropout(cur_itrs, total_itrs)
        
        """ data load """
        try:
            images, labels, sal_maps, _, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, sal_maps, _, _ = train_iter.next()
            cur_epoch += 1
            avg_loss.reset()
            
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
        """ memory """
        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, m_sal_maps, _, _ = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, m_sal_maps, _, _ = mem_iter.next()

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_sal_maps = m_sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]
            sal_maps[rand_index, ...] = m_sal_maps[rand_index, ...]

        
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=opts.amp):

            outputs = model(images)

            if opts.pseudo and opts.curr_step > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs_prev = model_prev(images)

                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh), 
                                            pred_labels, 
                                            labels)
                    
                loss = criterion(outputs, pseudo_labels)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if opts.lr_schedule:
            scheduler.step()
        avg_loss.update(loss.item())

        if cur_itrs % val_interval == 0 or cur_itrs == total_itrs:
            val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, 
                                 device=device, metrics=metrics)

            curr_score = val_score['Mean IoU']
            curr_class_score = list(val_score['Class IoU'].items())[-1][1]
            logger.add_scalar("Val_MeanIoU", curr_score, cur_epoch)
            logger.add_scalar("Val_Last_Class_IoU", curr_class_score, cur_epoch)
            if rank == 0:
                pbar.set_postfix({
                    "IoU": "%.3f" % curr_score,
                    "Cur IoU": "%.3f" % curr_class_score,
                    "Ep": cur_epoch
                })

            # for k, (img, true_label, target, pred) in enumerate(ret_samples):

            #     decode = val_loader.dataset.decode_target
            #     img = (denorm(img) * 255).astype(np.uint8)
            #     target = decode(target).transpose(2, 0, 1).astype(np.uint8)
            #     true_label = decode(true_label).transpose(2, 0, 1).astype(np.uint8)
            #     pred = decode(pred).transpose(2, 0, 1).astype(np.uint8)

            #     concat_img = np.concatenate((img, true_label, target, pred), axis=2)
            #     logger.add_image(f"Sample_{k}", concat_img, cur_epoch)
            
            #if curr_score > best_score and rank == 0:  # save best model
            if rank == 0 and cur_itrs == total_itrs:  # save best model
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.crop_size, opts.curr_step, opts.port))

            torch.distributed.barrier()

    
    logger.info("... Testing Best Model")
    best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.crop_size, opts.curr_step, opts.port)
    
    checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    model.module.load_state_dict(checkpoint["model_state"], strict=True)
    
    test_score, ret_samples = validate(opts=opts, model=model, loader=test_loader, 
                            device=device, metrics=metrics)
    logger.info(metrics.to_str(test_score))
    logger.add_table("Val_Class_IoU", metrics.to_table(test_score["Class IoU"]), opts.curr_step, new_per_step=False)
    logger.add_table("Val_Class_Acc", metrics.to_table(test_score["Class Acc"]), opts.curr_step, new_per_step=False)
    logger.add_scalar("Mean_IoU/All", test_score["Mean IoU"], opts.curr_step, new_per_step=False)
    logger.add_scalar("Mean_IoU/Old", test_score["Init Class IoU"][2], opts.curr_step, new_per_step=False)
    if test_score["Cont Class IoU"][2] != 0: logger.add_scalar("Mean_IoU/New", test_score["Cont Class IoU"][2], opts.curr_step, new_per_step=False)
    logger.add_scalar("Mean_Acc/All", test_score["Mean Acc"], opts.curr_step, new_per_step=False)
    logger.add_scalar("Mean_Acc/Old", test_score["Init Class Acc"][2], opts.curr_step, new_per_step=False)
    if test_score["Cont Class Acc"][2] != 0: logger.add_scalar("Mean_Acc/New", test_score["Cont Class Acc"][2], opts.curr_step, new_per_step=False)
    logger.add_figure("Val_Conf_Matrix", test_score["Confusion Matrix"], opts.curr_step)
    logger.add_scalar("Efficiency/Parameters", params, opts.curr_step, new_per_step=False)
    logger.add_scalar("Efficiency/MAdds", flops, opts.curr_step, new_per_step=False)

if __name__ == '__main__':
            
    opts = get_argparser().parse_args()

    if 'SLURM_JOB_GPUS' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    else:
        print(os.environ['SLURM_JOB_GPUS'] + os.environ['CUDA_VISIBLE_DEVICES'])

    if opts.dataset == 'voc':
        dataset = VOCSegmentation
        opts.dataset_mean = [0.485, 0.456, 0.406]
        opts.dataset_std = [0.229, 0.224, 0.225]
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError

    denorm = utils.Denormalize(opts.dataset_mean, opts.dataset_std)
        
    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset, opts.task))

    # Initialize torch.distributed. Assign rank to each device. If cpu, still use rank
    if torch.cuda.is_available():
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id = int(os.environ['LOCAL_RANK'])
        device = torch.device(device_id)
        torch.cuda.set_device(device_id)
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
    else:
        distributed.init_process_group(backend='gloo', init_method='env://')
        device_id = int(os.environ['LOCAL_RANK'])
        device = torch.device('cpu')
        rank, world_size = distributed.get_rank(), distributed.get_world_size()


    logdir = f"{opts.logdir}/{opts.task}/{opts.name}_{str(datetime.now())[5:-7]}_{opts.port}"
    logger = utils.Logger(logdir, rank, debug=False, step=0)

    logger.add_table("Params", vars(opts), 0)

    if opts.ret_samples:
        #[7,8,9,10, 24, 26, 31, 36]
        #[10, 14, 16, 20, 30, 33, 34, 36, 42, 44, 52]
        #[55, 57, 59, 66, 70, 72, 77, 80]
        # opts.ret_samples = [1, 10, 14, 16, 20, 30, 33, 34, 36, 42, 44, 52]
        # opts.ret_samples = [55, 57, 59, 66, 70, 72, 77, 80]
        # opts.ret_samples = [1, 14, 80, 30]
        opts.ret_samples = [1447, 1443, 1442, 1440, 1435, 1427]
    
    for step in range(start_step, total_step):
        opts.curr_step = step
        logger.step = step
        main(opts)

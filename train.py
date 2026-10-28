import logging
import os
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from datasets.get_data import get_loader, test_dataset, get_test_loader
from sam2.sam2_model.build_backbone_ours import Model
from sam2.tools.loss import iou_loss, optimized_structure_loss, BoundaryLoss, structure_loss, dice_loss, structure_loss
from metric.metric import Smeasure

from option import get_args_parser
from dataset_config import (
    TRAIN_DATASET,
    get_train_paths,
    get_test_sets,
    check_dataset,
    describe,
)


def resolve_device(opt):
    requested = str(getattr(opt, "device", "cuda")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()

def setup_logging(save_path):
    log_filename = 'log.txt'
    full_log_path = os.path.join(save_path, log_filename)

    # 确保日志目录存在
    os.makedirs(save_path, exist_ok=True)

    try:
        logging.basicConfig(
            filename=full_log_path,
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO,
            filemode='a',
            datefmt='%Y-%m-%d %I:%M:%S %p',
            force=True,
        )
    except Exception as e:
        print(f"配置日志时发生异常: {e}")
        logging.critical(f"配置日志时发生异常: {e}")


def build_model_and_optimizer(opt):
    print('Learning Rate: {}'.format(opt.lr))

    model = Model(opt)
    model.setup()
    model.to(resolve_device(opt))

    # ======== ✅ 参数量统计 START ========

    dumpy = torch.randn(1, 3, opt.input_size, opt.input_size).to(resolve_device(opt))
    model.eval()
    with torch.no_grad():
        _ = model(dumpy, dumpy)  # 前向传播一次以确保所有参数都被正确初始化

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Model Parameters ==========")
    print("Total params: {:.2f} M".format(total_params / 1e6))
    print("Trainable params: {:.2f} M".format(trainable_params / 1e6))
    print("======================================\n")
    # ======== ✅ 参数量统计 END ========


    # ======== ❌ 注释模型结构/参数名打印 START ========
    params_with_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)   # ← 注释掉这一行
            params_with_names.append({'name': name, 'param': param})
    # ======== ❌ 注释模型结构/参数名打印 END ========

    adapter_params=[]
    vmamba_params=[]
    fusion_params=[]
    decoder_params=[]

    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'adapter' in name:
            adapter_params.append(param)
        elif 'vssm' in name or 'vmamba' in name:
            vmamba_params.append(param)
        elif 'fusion' in name or 'hga' in name or 'gate' in name:
            fusion_params.append(param)
        else:
            decoder_params.append(param)



    optimizer=torch.optim.AdamW([
        {
            'params':adapter_params,
            'lr':opt.lr
        },{
            'params':vmamba_params,
            'lr':opt.lr*0.1
        },{
            'params':fusion_params,
            'lr':opt.lr
        },{
            'params':decoder_params,
            'lr':opt.lr
        }
    ])

    for group in optimizer.param_groups:
        group['name'] = [p['name'] for p in params_with_names]

    return model, optimizer


def build_dataloaders(opt):
    # 数据集路径配置统一在 dataset_config.py 中维护
    train_im_root, train_gt_root, train_modality_root = get_train_paths()
    test_sets = get_test_sets()

    # 打印数据集路径并校验目录是否齐全，缺失时给出明确提示
    print(describe([TRAIN_DATASET] + [t['name'] for t in test_sets]))
    check_dataset(TRAIN_DATASET, 'train')
    for test_set in test_sets:
        check_dataset(test_set['name'], 'test')

    train_loader = get_loader(str(train_im_root),
                              str(train_gt_root),
                              modality_root=str(train_modality_root),
                              batchsize=opt.batch_size_train, trainsize=opt.input_size,
                              num_workers=4, pin_memory=True)

    test_loaders = []
    for test_set in test_sets:
        #loader = test_dataset(
        loader = get_test_loader(
            str(test_set['im_root']),
            str(test_set['gt_root']),
            modality_root=str(test_set['modality_root']),
            testsize=opt.input_size,
            batchsize=1,
            num_workers=4
        )
        test_loaders.append((test_set['name'], loader))

    return train_loader, test_loaders


def train(train_loader, model, optimizer, epoch, total_step, total_epoch, device):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    bce_loss = nn.BCEWithLogitsLoss().to(device)

    # 实例化一些新的损失函数
    criterion_struct = optimized_structure_loss
    criterion_boundary = BoundaryLoss().to(device)
    criterion_dice = dice_loss

    progress = tqdm(
        enumerate(train_loader, start=1),
        total=total_step,
        desc=f"Train E{epoch}/{total_epoch}",
        ncols=100,
    )
    for i, pack in progress:
        optimizer.zero_grad()

        _, aug_image, modality, gts = pack

        aug_image = aug_image.to(device)
        modality = modality.to(device)
        gts = gts.to(device)

        with autocast_ctx(device):
            sam_out, vssm_out, vssm_out_prediction, vssm_out_detection = model(aug_image, modality)
        
        with nullcontext():
            sam_mask_fp32 = sam_out.float()
            vssm_mask_prediction_fp32 = vssm_out_prediction.float()
            vssm_mask_detection_fp32 = vssm_out_detection.float()
            gts_fp32 = gts.float()

            #loss = loss_s + 1.0 * loss_b + 0.5 * loss_d

            #sam_loss = bce_loss(sam_mask_fp32, gts_fp32) + dice_loss(sam_mask_fp32, gts_fp32)
            #sam_loss = bce_loss(sam_mask_fp32, gts_fp32) + dice_loss(sam_mask_fp32, gts_fp32)
            sam_loss = structure_loss(sam_mask_fp32, gts_fp32)# + dice_loss(sam_mask_fp32, gts_fp32)
            #edge_loss = dice_loss(edge_mask_fp32, edge)
            vssm_loss = 0.2 * structure_loss(vssm_mask_prediction_fp32, gts_fp32) + 0.5 * structure_loss(vssm_mask_detection_fp32, gts_fp32)

            loss = sam_loss + vssm_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=f"{loss.item():.4f}")

        if i % 100 == 0 or i == total_step or i == 1:
            now_formatted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.
                                    format(now_formatted, epoch, total_epoch, i, total_step, loss.data))
            logging.info(
                '#TRAIN#: Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.
                                format(epoch, total_epoch, i, total_step, loss.data))

def test(test_loaders, model, epoch, save_path, best_sm, best_epoch, device):
    model.eval()
    with torch.no_grad():
        total_sm = 0
        total_size = 0
        for dataset_name, test_loader in test_loaders:
            SM = Smeasure()
            # 使用标准的 enumerate(test_loader)
            progress = tqdm(test_loader, desc=f"Test {dataset_name} E{epoch}", ncols=100)
            
            for pack in progress:
                # 根据 get_test_loader 的返回值解包
                image, _, modality, gt, name = pack 
                
                image = image.to(device)
                modality = modality.to(device)
                gt = gt.to(device).float() # 直接送入 GPU 参与计算

                res = model(image, modality)
                if isinstance(res, (tuple, list)):
                    res = res[0]
                
                # 在 GPU 上完成插值和归一化
                res = F.interpolate(res, size=(gt.shape[1], gt.shape[2]), mode='bilinear', align_corners=False)
                res = res.sigmoid()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                
                res_np = (res.squeeze().detach().cpu().numpy()*255).astype(np.uint8)
                gt_np = (gt.squeeze().detach().cpu().numpy()*255).astype(np.uint8)
                SM.step(res_np, gt_np)
                
            dataset_sm = SM.get_results()['sm']
            total_sm += dataset_sm
            total_size += 1
            print('Epoch: {} {} SM: {:.10f}'.format(epoch, dataset_name, dataset_sm))
            logging.info('#TEST#:{} Epoch:{} SM:{:.10f}'.format(dataset_name, epoch, dataset_sm))
        sm = total_sm / max(total_size, 1)
        print('Epoch: {} SM: {:.10f} ####  bestSM: {:.10f} bestEpoch: {}'.format(epoch, sm, best_sm, best_epoch))

        if epoch == 1:
            best_sm = sm
            best_epoch = epoch
            torch.save(model.state_dict(),
                       save_path + '-sam_rgb-' + str(sm) + '_%d-' % epoch + '.pth')
        else:
            if sm > best_sm:
                best_sm = sm
                best_epoch = epoch
                torch.save(model.state_dict(),
                           save_path + '-rgbt-' + str(best_sm) + '-best' + '_%d-' % epoch + '.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} SM:{:.10f} bestEpoch:{} bestSM:{:.10f}'.
                     format(epoch, sm, best_epoch, best_sm))

    return best_sm, best_epoch

def main():
    opt = get_args_parser()
    device = resolve_device(opt)
    warmup_epochs = opt.warmup_epochs
    print(f"Using device: {device}")

    setup_logging(opt.output_dir)

    model, optimizer = build_model_and_optimizer(opt)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (opt.max_epoch_num - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )

    train_loader, test_loaders = build_dataloaders(opt)
    total_step = len(train_loader)

    best_sm = 0
    best_epoch = 0

    print("Start Training!")
    logging.info("训练rgb")

    for epoch in range(1, opt.max_epoch_num + 1):
        train(train_loader, model, optimizer, epoch, total_step, opt.max_epoch_num, device)
        scheduler.step()
        best_sm, best_epoch = test(test_loaders, model, epoch, opt.output_dir, best_sm, best_epoch, device)


if __name__ == '__main__':
    main()

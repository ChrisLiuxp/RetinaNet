#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import retinanet_dataset_collate, RetinanetDataset
from nets.retinanet import Retinanet
from nets.retinanet_training import FocalLoss
from nets.fcos_training import FCOSLoss
from tqdm import tqdm
from utils.utils import warmup_lr_scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def fit_one_epoch(net,focal_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()

            _, regression, classification, anchors = net(images)
            loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets, cuda=cuda)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-2)
            optimizer.step()

            total_loss += loss
            total_r_loss += r_loss
            total_c_loss += c_loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss,_,_ = focal_loss(classification, regression, anchors, targets_val, cuda=cuda)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)


def fit_one_epoch_new(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_ctn_loss = 0
    total_loss = 0
    val_loss = 0

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            # images shape:[batch_size, 3, input_image_size, input_image_size]
            # targets是包含batch_size个元素的list 元素shape:[一张图GT个数, 5]
            cls_heads, reg_heads, center_heads, batch_positions = net(images)
            cls_loss, reg_loss, center_ness_loss = fcos_loss(cls_heads, reg_heads, center_heads, batch_positions, targets, cuda=cuda)
            loss = cls_loss + reg_loss + center_ness_loss
            if cls_loss.item() == 0.0 or reg_loss.item() == 0.0:
                optimizer.zero_grad()
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_r_loss += cls_loss.item()
            total_c_loss += reg_loss.item()
            total_ctn_loss += center_ness_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Classification Loss': total_c_loss / (iteration+1),
                                'Regression Loss'    : total_r_loss / (iteration+1),
                                'Center-ness Loss'   : total_ctn_loss / (iteration+1),
                                'lr'                 : get_lr(optimizer),
                                'step/s'             : waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                cls_heads, reg_heads, center_heads, batch_positions = net(images_val)
                cls_loss, reg_loss, center_ness_loss = fcos_loss(cls_heads, reg_heads, center_heads, batch_positions, targets_val, cuda=cuda)
                loss = cls_loss + reg_loss + center_ness_loss
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)


def fit_one_epoch_warmup(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_ctn_loss = 0
    total_loss = 0
    val_loss = 0

    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(gen) - 1)
    #
    #     lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            # images shape:[batch_size, 3, input_image_size, input_image_size]
            # targets是包含batch_size个元素的list 元素shape:[一张图GT个数, 5]
            cls_heads, reg_heads, center_heads, batch_positions = net(images)
            cls_loss, reg_loss, center_ness_loss = fcos_loss(cls_heads, reg_heads, center_heads, batch_positions, targets, cuda=cuda)
            loss = cls_loss + reg_loss + center_ness_loss
            if cls_loss.item() == 0.0 or reg_loss.item() == 0.0:
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-2)
            optimizer.step()

            total_loss += loss.item()
            total_r_loss += cls_loss.item()
            total_c_loss += reg_loss.item()
            total_ctn_loss += center_ness_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Classification Loss': total_c_loss / (iteration+1),
                                'Regression Loss'    : total_r_loss / (iteration+1),
                                'Center-ness Loss'   : total_ctn_loss / (iteration+1),
                                'lr'                 : get_lr(optimizer),
                                'step/s'             : waste_time})
            pbar.update(1)

            start_time = time.time()

            # if lr_scheduler is not None:
            #     lr_scheduler.step()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                cls_heads, reg_heads, center_heads, batch_positions = net(images_val)
                cls_loss, reg_loss, center_ness_loss = fcos_loss(cls_heads, reg_heads, center_heads, batch_positions, targets_val, cuda=cuda)
                loss = cls_loss + reg_loss + center_ness_loss
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f || Learning rate: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), optimizer.state_dict()['param_groups'][0]['lr']))

    # print('Saving state, iter:', str(epoch+1))
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    if epoch % 10 == 0:
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_scheduler': lr_scheduler.state_dict()
        }
        if not os.path.isdir("./model_data/checkpoint"):
            os.mkdir("./model_data/checkpoint")
        torch.save(checkpoint, './model_data/checkpoint/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    return val_loss/(epoch_size_val+1)


if __name__ == "__main__":
    #--------------------------------------------#
    #   phi == 0 : resnet18
    #   phi == 1 : resnet34
    #   phi == 2 : resnet50
    #   phi == 3 : resnet101
    #   phi == 4 : resnet152
    #--------------------------------------------#
    phi = 2
    # 是否使用GPU
    Cuda = True
    # 是否需要BiFPN
    BiFPN_on = True
    # 是否断点续训
    RESUME = False
    # 是否加载模型进行FineTurn
    FineTurn = False

    # 起始epoch(不需更改)
    start_epoch = -1
    # 冻结backbone训练模型，[start_epoch+1, Freeze_Epoch]
    Freeze_Epoch = 100
    # 解冻backbone训练模型，[Freeze_Epoch+1, Unfreeze_Epoch]
    Unfreeze_Epoch = 200

    # 冻结、解冻backbone训练模型的batch_size
    Freeze_Batch_size = 32
    Unfreeze_Batch_size = 16
    # 冻结backbone训练模型的初始学习率
    lr = 1e-4

    #--------------------------------------------#
    #   输入图像大小
    #--------------------------------------------#
    if BiFPN_on:
        input_shape = (512, 512)
    else:
        input_shape = (600, 600)
    annotation_path = '2007_train.txt'

    #--------------------------------------------#
    #   训练自己的模型需要修改txt
    #--------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    # 创建模型
    model = Retinanet(num_classes, phi, True, BiFPN_on)

    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    if FineTurn:
        model_path = "model_data/retinanet_resnet50.pth"
        # 加快模型训练的效率
        print('Loading weights into state dict...')
        model_dict = model.state_dict()
        if Cuda:
            pretrained_dict = torch.load(model_path)
        else:
            pretrained_dict = torch.load(model_path, map_location='cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Finished!')

    optimizer = optim.Adam(model.parameters(),lr)
    # 学习率阶层性下降
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    # 学习率余弦退火下降
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # 是否执行冻结backbone的训练（不需手动更改）
    freezeflag = True
    if RESUME:
        path_checkpoint = "./model_parameter/test/ckpt_best_50.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        if start_epoch+1 > Freeze_Epoch:
            freezeflag = False
            Freeze_Epoch = start_epoch+1
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # focal_loss = FocalLoss()
    fcos_loss = FCOSLoss()

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if freezeflag:

        train_dataset = RetinanetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
        val_dataset = RetinanetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
        gen = DataLoader(train_dataset, batch_size=Freeze_Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=retinanet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Freeze_Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=retinanet_dataset_collate)

        epoch_size = num_train//Freeze_Batch_size
        epoch_size_val = num_val//Freeze_Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(start_epoch+1,Freeze_Epoch):
            val_loss = fit_one_epoch_warmup(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)


    if True:
        # 解冻backbone训练模型的初始学习率
        lr = 1e-5

        if freezeflag:
            optimizer = optim.Adam(net.parameters(),lr)
            # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

        train_dataset = RetinanetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
        val_dataset = RetinanetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
        gen = DataLoader(train_dataset, batch_size=Unfreeze_Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=retinanet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Unfreeze_Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=retinanet_dataset_collate)

        epoch_size = num_train//Unfreeze_Batch_size
        epoch_size_val = num_val//Unfreeze_Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch_warmup(net,fcos_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
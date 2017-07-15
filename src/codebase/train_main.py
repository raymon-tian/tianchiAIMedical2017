#coding=utf-8
"""
本脚本函数
main
train
test
singletest
"""
import os
import sys
import time
import glob
import torch
import shutil
import argparse

import numpy as np

from torch import optim
from torch.backends import cudnn
from torch.nn import DataParallel
from importlib import import_module
from torch.autograd import Variable
from torch.utils.data import DataLoader


# from data import *
from utils import *
from test_main import test_detect
from split_combine import SplitComb
from config_train import config as config_train
# from layers import acc


parser = argparse.ArgumentParser(description='PyTorch Lung nodule Detector')

# 设置网络计算图，不是训练好的模型参数
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
# 数据加载线程
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')

parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')

parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')

parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')

parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

def main():
    global args
    args = parser.parse_args()
    
    torch.manual_seed(0)          # 随机种子

    model = import_module(args.model)               # 动态的载入 net_res18.py模块
    config, net, loss, get_pbb = model.get_model()  # 网络配置参数，网络，损失，bounding box 
    start_epoch = args.start_epoch  
    weights_save_dir = config_train['weights_path']                       # 存储网络权重的文件夹 res18
    val_result_dir = config_train['val_result_path']
   
    if args.resume:                                 # 载入训练好的模型权重，继续训练，如果我们需要fine tune,要是用这个模式
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1   # 在之前的epoch上继续累加计数
        net.load_state_dict(checkpoint['state_dict'])    # 载入模型权重
    else:                                           # 全新的训练
        if start_epoch == 0:
            start_epoch = 1                         # epoch从0开始计数

    if not os.path.exists(weights_save_dir):
        os.makedirs(weights_save_dir)
    if not os.path.exists(val_result_dir):
        os.makedirs(val_result_dir)


    net = net.cuda()                # 网络设置为 GPU格式
    loss = loss.cuda()              # 损失设置为 GPU格式
    cudnn.benchmark = True          # 使用cudnn加速
    net = DataParallel(net)         # 若有多GPU，则将batch分开，并行处理
    train_datadir = config_train['train_preprocess_result_path']  # 训练数据预处理结果路径
    val_datadir = config_train['val_preprocess_result_path']      # 验证数据预处理结果路径

    if args.test == 1:   # 因为使用了train的数据做训练，就用val的数据做验证
        margin = 32
        sidelen = 144

        split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
        val_dataset = DataBowl3Detector(
            val_datadir,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            val_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = args.workers,
            collate_fn = collate,
            pin_memory=False)
        
        test_detect(test_loader, net, get_pbb, val_result_dir,config,arg.ntest)
        return

    # 准备训练数据和val数据，在fine tune的时候，修改成自己的数据
    train_dataset = DataBowl3Detector(         
        train_datadir,
        config,
        phase = 'train')

    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    val_dataset = DataBowl3Detector(           # validation数据
        val_datadir,
        config,
        phase = 'val')
    val_loader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(            # 梯度下降法，学习率0.01，
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)
    
    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:      # opechs分为三个阶段，分别为 0.01, 0.001, 0.0001
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    
    # 每个epoch都可以设置不同的参数来进行训练和验证
    for epoch in range(start_epoch, args.epochs + 1):  
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, weights_save_dir)
        validate(val_loader, net, loss)

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):

    start_time = time.time()
    
    net.train()                                       # 设置为训练模式，batchNM和Dropout
    lr = get_lr(epoch)                                # 每一个epoch都可以单独设置训练参数
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                        # 重置学习率

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True))      # 图像数据
        target = Variable(target.cuda(async = True))  #
        coord = Variable(coord.cuda(async = True))    # 都是什么数据，要弄清楚

        output = net(data, coord)                     # 网络输出
        loss_output = loss(output, target)            # 计算损失
        optimizer.zero_grad()                         # 梯度清零
        loss_output[0].backward()                     # 计算梯度
        optimizer.step()                              # 更新权重

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:                   # args.save_freq : 10    
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu() # 存盘需要，将数据保存成CPU格式
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print

def validate(data_loader, net, loss):
    start_time = time.time()
    
    net.eval()                                # 转换成 测试模式，BatchNM和drop会受到影响

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async = True), volatile = True)       # 转换成torch格式
        target = Variable(target.cuda(async = True), volatile = True)   # 转换成torch格式
        coord = Variable(coord.cuda(async = True), volatile = True)     # 转换成torch格式

        output = net(data, coord)                                       # 网络输出
        loss_output = loss(output, target, train = False)               # 计算损失

        loss_output[0] = loss_output[0].data[0]                         # 损失数据
        metrics.append(loss_output)    
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))

def singletest(data,net,config,splitfun,combinefun,n_per_run,margin = 64,isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data,config['max_stride'],margin)
    data = Variable(data.cuda(async = True), volatile = True,requires_grad=False)
    splitlist = range(0,args.split+1,n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist)-1):
        output = net(data[splitlist[i]:splitlist[i+1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)
        
    output = np.concatenate(outputlist,0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    return output

if __name__ == '__main__':
    main()


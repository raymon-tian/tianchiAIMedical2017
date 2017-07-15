#coding=utf-8

import os
import sys
import time
import glob
import torch
import pandas
import shutil
import argparse

import numpy as np

from torch import optim          # 实现了多种优化算法的库
from pprint import pprint
from torch.backends import cudnn
from torch.nn import DataParallel   # 多GPU并行运算，将输入平均分成几个不同部分
from torch.autograd import Variable
from importlib import import_module   #在程序中动态加载模块和类
from torch.utils.data import DataLoader


from utils import *
from net_layers import acc
from split_combine import SplitComb
from data_load import DataBowl3Detector,collate

from config_test import config as config_test

parser = argparse.ArgumentParser(description='PyTorch Lung nodule Detector')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

def test_detect(data_loader, net, get_pbb, save_dir, config, n_gpu):
    """
    :param data_loader: PyTorch 提供
    :param net:         网络
    :param get_pbb:     ？
    :param save_dir:    结果保存路径 这里应该修改一下，直接保存成csv文件，保存到final_results中，文件名中应该由当前时间
    :param config:      网络参数   
    :param n_gpu:       使用几个GPU运算
    """
    start_time = time.time()
    net.eval()  # 将模型设置为evaluation模式，会影响到Dropout 和 BatchNorm
    split_comber = data_loader.dataset.split_comber
    # data:3D CT; target:该CT的所有肺结节标注信息; coord：? ; nzhw：grid的维度
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        # 这样返回过来的数据，并不是对应 Dataset.__getitem__，而是在这个基础上又增加了一维，batch_size
        # data 是6个维度，batch_size,patch_num,channel,Z,H,D
        print('=============================')  
        print('data shape',len(data))             # batch_size=1
        print('target shape',len(target))         # batch_size=1
        print('coord shape',len(coord))           # batch_size=1
        print('nzhw shape',len(nzhw))             # batch_size=1
        print('data[0] shape', data[0].size())    # (1,27,1,208,208,208)
        print('target[0] shape', target[0].shape) # (0,)
        print('coord[0] shape', coord[0].size())  # (1,27,3,52,52,52) 
        print('nzhw[0] shape', nzhw[0].shape)     # ((3,)

        s = time.time()

        name = data_loader.dataset.filenames[i_name].split('/')[-1]  # 当前处理的图像的名字
        shortname = name.split('_clean')[0]    # 不带后缀的名字 

        print([i_name,shortname])

        nzhw = nzhw[0]  # 第一个batch的内容, nzhw.shape()
        lbb = target[0] # 因为batchsize=1,所以这里只选了第一个 targe.shape(N,4)
        data = data[0][0]                               # 现在的data 应该是 (patch_num,1,D,H,W)
        coord = coord[0][0] 
        target = [np.asarray(t, np.float32) for t in target] #按batch取                            # (patch_num,3,D,H,W)
        splitlist = range(0,len(data)+1,n_gpu)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []

        print('**** input & output shape of detector network\n')
        # 循环处理每一个 CT的patch
        for i in range(len(splitlist)-1):  # 每次循环，n_gpu个GPU同时各自跑一个patch
            # CT图像的一个patch shape为 (1,1,208,208,208) 208=144+32+32
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            output = net(input,inputcoord)
            #print('input size',input.size())
            #rint('output size',output.size())
            outputlist.append(output.data.cpu().numpy())

        output = np.concatenate(outputlist,0)
        print('patch output size',np.shape(output)) # shape (52,52,52,3,5)
        output = split_comber.combine(output,nzhw=nzhw)#将输出整合起来
        print('split_comber.combine 将网络针对一个CT的所有output结合起来的shape ',output.shape)
        print('final output size',np.shape(output))

        thresh = config['thresh']   # 这里是概率值，负例为负数，正例为整数。 比如我们可以通过该参数控制最终保存的结果数量
        pbb = get_pbb(output,thresh) # 根据输出得到预测的bounding box

        e = time.time()

        # pbb的shape为(N,5) lbb的shape为(1,4)
        np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
        np.savetxt(os.path.join(save_dir, shortname+'_pbb.txt'), pbb)
        np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


def main():
    global args
    args = parser.parse_args()
    test_datadir = config_test['test_preprocess_result_path']
    nodmodel = import_module(config_test['detector_model']) #动态加载net_detector模块，其中定义了Config参数,检测网络结构，前传函数
    config, nod_net, loss, get_pbb = nodmodel.get_model()  # 生成模型
    checkpoint = torch.load(args.resume)  # 加载整个模型权重
    nod_net.load_state_dict(checkpoint['state_dict'])  # 仅加载模型参数

    nod_net = nod_net.cuda() # 将数据移动到GPU上计算
    cudnn.benchmark = True # 使用cudnn加速
    nod_net = DataParallel(nod_net) # 将输入的minibatch平均分开，放入多个GPU前传，各块GPU上的梯度加起来以后前传

    test_result_dir = config_test['test_result_path'] # bounding_box结果
    if not os.path.exists(test_result_dir):
        os.mkdir(test_result_dir)

    margin = 32
    sidelen = 144 # 将完整的CT图像切割成小立方体
    n_gpu = args.n_test

    # config['datadir'] = prep_result_path #输入图像路径，为经过预处理之后的文件夹路径
    # 先将3DCT图像进行分裂，因为如果不分裂，GPU会爆显存
    # 构造函数，不执行具体计算
    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,pad_value= config['pad_value'])
    # 构造函数，不执行具体计算
    dataset = DataBowl3Detector(test_datadir,config,phase='test',split_comber=split_comber)
    # 构造函数，不执行具体计算
    test_loader = DataLoader(dataset,batch_size = 1,shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)
    # 开始执行计算
    test_detect(test_loader, nod_net, get_pbb, test_result_dir, config, n_gpu=n_gpu)
 
if __name__ == '__main__':
    main()
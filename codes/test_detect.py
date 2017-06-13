#coding=utf-8

import argparse
import os
import time
import numpy as np
from importlib import import_module
import shutil
from utils import *
import sys

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from split_combine import SplitComb
from layers import acc

def test_detect(data_loader, net, get_pbb, save_dir, config,n_gpu):
    """
    usage : 
    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

    :param data_loader: from torch.utils.data import DataLoader 由torch提供
    :param net: 
    :param get_pbb: 
    :param save_dir: 
    :param config: 
    :param n_gpu: 
    :return: 
    """
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber
    # data:3D CT; target:该CT的所有肺结节标注信息; coord：CT patch的list ; nzhw：grid的维度
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        print('=============================')
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target] # [第一条肺结节信息，第二条肺结节信息]
        lbb = target[0]
        nzhw = nzhw[0]
        # 要修改，有问题
        # name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        name = data_loader.dataset.filenames[i_name].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0] # 现在的data 应该是 (D,H,W)
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:#config中没有这么一项
            if config['output_feature']:
                isfeat = True
        n_per_run = n_gpu
        # print(data.size())
        splitlist = range(0,len(data)+1,n_gpu)
        if splitlist[-1]!=len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist)-1):
            input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            # isfeat 为 False
            if isfeat:
                output,feature = net(input,inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input,inputcoord)# 这才是网络的真正输入
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output,nzhw=nzhw)#将输出整合起来
        # isfeat 为 False
        if isfeat:
            feature = np.concatenate(featurelist,0).transpose([0,2,3,4,1])[:,:,:,:,:,np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = -3
        pbb,mask = get_pbb(output,thresh,ismask=True)# 根据输出得到预测的bounding box
        # isfeat 为 False
        if isfeat:
            feature_selected = feature[mask[0],mask[1],mask[2]]
            np.save(os.path.join(save_dir, shortname+'_feature.npy'), feature_selected)
        #tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        #print([len(tp),len(fp),len(fn)])
        print([i_name,shortname])
        e = time.time()
        
        np.save(os.path.join(save_dir, shortname+'_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, shortname+'_lbb.npy'), lbb)
    end_time = time.time()


    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print

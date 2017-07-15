#coding=utf-8
"""
准备训练模型所需要的数据
"""
import os
import sys
import h5py
import glob
import scipy
import shutil
import pandas
import argparse
import warnings
import multiprocessing

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pprint import pprint
from skimage import measure
from scipy.io import loadmat
from functools import partial
from multiprocessing import Pool
from scipy.ndimage.interpolation import zoom
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure

from data_prep_func import *
from config_train import config as config_train
from config_test import config as config_test



parser = argparse.ArgumentParser(description='PyTorch Lung nodule Preprocessing')

parser.add_argument('--mode', default='train', type=str, help='test or train or val')


def start_process():
    print 'Starting',multiprocessing.current_process().name  # 输出进程名


def full_prep(n_worker=None):
    global args
    args = parser.parse_args()
    warnings.filterwarnings("ignore")               # 忽略python的warning信息
    mode = args.mode
    assert(mode == 'train' or mode == 'val' or mode == 'test')


    if mode =="train":
        prep_folder = config_train['train_preprocess_result_path']
        data_path = config_train['train_data_path']
        finished_flag = '.flag_prepTianChi'
        if not os.path.exists(prep_folder):
            os.makedirs(prep_folder)
        if not os.path.exists(finished_flag):
            csv_path = config_train['train_annos_path']
            assert csv_path[0].split('/')[-1] == 'annotations.csv','请检查天池数据集annotations.csv的设置路径'
            annos = pandas.read_csv(csv_path[0])        # 读取csv内容
            annos = annos.as_matrix()                   # 转化成矩阵表示
            print('starting preprocessing')
            train_list = [f for f in glob.glob(data_path+'/*.mhd')]  # 训练数据的全部文件
            
            partial_savenpy = partial(savenpy, annos=annos, img_list=train_list, prep_folder=prep_folder,use_existing=True )
            pool = Pool(processes=n_worker,initializer=start_process) # multiprocessing 进程管理包的Pool类，n_worker=none 表示开始能使用的全部进程
            N = len(train_list)
            _ = pool.map(partial_savenpy, range(N)) # 同时开启多个线程进行预处理
            pool.close()  # 关闭进程池，使其不再接受新的任务
            pool.join()   # 主进程阻塞，等待子进程的退出


            print('end preprocessing')
        f = open(finished_flag, "w+")


    if mode =="test":
        prep_folder = config_test['test_preprocess_result_path']
        data_path = config_test['test_data_path']
        finished_flag = '.flag_prepTianChi'
        if not os.path.exists(prep_folder):
            os.makedirs(prep_folder)
        if not os.path.exists(finished_flag):
            print('starting preprocessing')

            test_list = [f for f in glob.glob(data_path+'/*.mhd')]  # 测试数据的全部文件

            partial_savenpy = partial(savenpy, annos=None, img_list=test_list, prep_folder=prep_folder,use_existing=True )
            pool = Pool(processes=n_worker,initializer=start_process) # multiprocessing 进程管理包的Pool类，n_worker=none 表示开始能使用的全部进程
            N = len(test_list)  # 测试文件个数
            _ = pool.map(partial_savenpy, range(N)) # 同时开启多个线程进行预处理
            pool.close()  # 关闭进程池，使其不再接受新的任务
            pool.join()   # 主进程阻塞，等待子进程的退出
            print('end preprocessing')
        f = open(finished_flag, "w+")

if __name__=='__main__':
    full_prep(n_worker=None)


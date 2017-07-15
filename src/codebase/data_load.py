#coding=utf-8

"""
定义了3个类
DataBowl3Detector
Crop
LabelMapping
两个函数
augment

collate
"""
import os
import time
import glob
import torch
import random
import warnings
import collections

import numpy as np

from scipy.ndimage import zoom
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

from net_layers import iou

class DataBowl3Detector(Dataset):
    """
    继承自Dataset类，必须实现其中的 __getitem__ 以及 __len__ 方法
    """
    def __init__(self, data_dir, config, phase = 'train',split_comber=None):
        """
        
        :param data_dir:      预处理结果的存放路径
        :param split_path:    存放所有需处理的CT的id
        :param config:        网络的配置字典 net_res18中
        :param phase:         运行模式：train、test、val
        :param split_comber:  切分和组合CT图像函数
        """

        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']              # 16
        self.stride = config['stride']                      # 4
        sizelim = config['sizelim']/config['reso']          # 6 mm
        sizelim2 = config['sizelim2']/config['reso']        # 30 mm
        sizelim3 = config['sizelim3']/config['reso']        # 40 mm
        self.blacklist = config['blacklist']                # blacklist
        self.isScale = config['aug_scale']                  # True
        self.r_rand = config['r_rand_crop']                 # 0.3
        self.augtype = config['augtype']                    # flip scale
        self.pad_value = config['pad_value']                # 170
        self.split_comber = split_comber                    # split combine method

        
        split_0 = [f for f in glob.glob(data_dir+'/*clean.npy')]  # 训练数据的全部文件 
        split_1 = [f.split('/')[-1] for f in split_0]
        split_list = [f.split('_clean')[0] for f in split_1]

        idcs = split_list                                          # 所有CT图像的ID列表，没有后缀

        if phase!='test':                                  
            idcs = [f for f in idcs if (f not in self.blacklist)]  # 非测试模式下，将黑名单中的文件去除
                                                           
        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs] 
                                                                   # 所有的CT图像数据列表,完整路径，带后缀
        
        labels = []
        
        for idx in idcs:
            l = np.load(os.path.join(data_dir, '%s_label.npy' % idx))
            if np.all(l==0): # 若l中全为0
                l=np.array([])
            labels.append(l)  # 按idx顺序索引。 测试时都为0，下面有没有相应的处理策略呢？

        # 标签 [ ndarray(?,4),ndarray(?,4),...,ndarray(?,4),ndarray(?,4)]
        # sample_bboxes是按照CT索引的，也就是一个CT的标注信息是集中到一起的
        self.sample_bboxes = labels 
        if self.phase != 'test':
            self.bboxes = []   # 一个肺结节标注是一条记录
            for i, l in enumerate(labels):  # 对labels进行遍历，以图像为单位索引
                if len(l) > 0 :             # l是某一张CT图像的所有肺结节标注信息  # 应对  不同直径肺结节样本  不均衡的问题
                    for t in l:
                        self.bboxes.append([np.concatenate([[i],t])])  # id,d,h,w,d
            # 这个时候，来自同一个CT的肺结节已经分开放置
            # 每一行表示一个标注信息，shape:(N,5), 分别表示为 id，z,h,w,dia
            self.bboxes = np.concatenate(self.bboxes,axis = 0)

        self.crop = Crop(config)                              # 构造函数，不进行具体计算
        self.label_mapping = LabelMapping(config, self.phase) # 构造函数，不进行具体计算

    def __getitem__(self, idx, split=None):
        """
        继承父类方法，必须实现
        :param idx: 是对 一条肺结节标注信息 的索引，不是对CT图像的索引
        :param split: 
        :return: 
        """
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))    #seed according to time

        isRandomImg  = False
        if self.phase !='test':
            if idx>=len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                # 随机生成 整数 0 以及 1
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False
        
        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]                   # 取一条肺结节标注记录
                filename = self.filenames[int(bbox[0])]   # 取出索引，和self.sample_bbox一致
                imgs = np.load(filename)                  # 读取图像
                # 该CT图像对应的所有肺结节
                bboxes = self.sample_bboxes[int(bbox[0])] # 读取该图像对应的所有标注
                
                isScale = self.augtype['scale']
            
                #  这里有疑问？？？？？？？？
                if self.phase=='train':  # 只在 train 阶段 进行crop
                    sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes,isScale,isRandom)

                if self.phase=='train' and not isRandom:
                    # 进行数据增强
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip = self.augtype['flip'], 
                                                            ifrotate=self.augtype['rotate'], 
                                                            ifswap = self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.filenames))   # 随机生成一个图像ID
                filename = self.filenames[randimid]                 # 图像路径
                imgs = np.load(filename)                            # 读取图像
                bboxes = self.sample_bboxes[randimid]               # 读取该图像的所有标注信息
                isScale = self.augtype['scale'] and (self.phase=='train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes,isScale=False,isRand=True)
            label = self.label_mapping(sample.shape[1:], target, bboxes)

            # 处理到[-0.5,0.5]
            sample = (sample.astype(np.float32)-128)/128
           
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            # test 阶段
            # 将图像的 D H W 处理为self.stride 的整数倍(ceil方式)，多出来的像素值填充为self.pad_value
            imgs = np.load(self.filenames[idx]) # xxxx_clean.npy 预处理过后的CT图像
            bboxes = self.sample_bboxes[idx]    # 读取该图像对应的全部标注信息
            nz, nh, nw = imgs.shape[1:]         # imgs 的shape为（1，D ,H W），取后三个维度的大小
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride   # 转化为stride的整数倍，ceil向上取整
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride   # 转化为stride的整数倍
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride   # 转化为stride的整数倍
            imgs = np.pad(imgs, [[0,0],[0, pz - nz], [0, ph - nh], [0, pw - nw]],
                         'constant',constant_values = self.pad_value)  # 用170填充，此时imgs为填充后的图像
            # 将CT分裂，分写成patch,需要图像的patch 以及 patch的坐标信息
            xx,yy,zz = np.meshgrid(np.linspace(-0.5,0.5,imgs.shape[1]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[2]/self.stride),
                                   np.linspace(-0.5,0.5,imgs.shape[3]/self.stride),indexing ='ij')
            coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

            # 上面到此为止， imgs和coord都是四维的（channel,Z,H,W）
            imgs, nzhw = self.split_comber.split(imgs)
            # 这里返回来的imgs是五个维度(patch_num,channel,Z,H,W)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                   side_len = self.split_comber.side_len/self.stride,
                                                   max_stride = self.split_comber.max_stride/self.stride,
                                                   margin = self.split_comber.margin/self.stride)
            assert np.all(nzhw==nzhw2)
            imgs = (imgs.astype(np.float32)-128)/128               # 归一化到（-1,1）
            imgs = torch.from_numpy(imgs)                          # 转化到tensor格式
            coord2 = torch.from_numpy(coord2.astype(np.float32))   # 转化到tensor模式
            nzhw = np.array(nzhw)
            print('Dataset __ getitem__ return shape\n')
            print('img shapes:',imgs.size())                       # 图像大小
            print('bboxes shape',bboxes.shape)                     # 该图像的所有标注信息
            print('coord2 shape',coord2.size())                    # coord2也经过了切割，是imgs的四分之一
            print('nzhw shape',nzhw.shape)                         # D,H,W 各被切成了多少块
            return imgs, bboxes, coord2, nzhw
            # 切分好的imgs图像，肺结节标注信息，。。。，切分后的图像shape

    def __len__(self):
        """
        继承父类方法，必须实现
        :return: 
        """
        if self.phase == 'train':
            # train
            return len(self.bboxes)/(1-self.r_rand)     # 训练标注的个数，总的标记数除以0.7
        elif self.phase =='val':
            # val
            return len(self.bboxes)                     # val标注的个数
        else:
            # test
            return len(self.sample_bboxes)              # 一共有多少个训练数据
        
        
def augment(sample, target, bboxes, coord, ifflip = True, ifrotate=True, ifswap = True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand()*180
            # angle1 = (np.random.rand()-0.5)*20
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1/180*np.pi),-np.sin(angle1/180*np.pi)],
                               [np.sin(angle1/180*np.pi),np.cos(angle1/180*np.pi)]])
            newtarget[1:3] = np.dot(rotmat,target[1:3]-size/2)+size/2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample,angle1,axes=(2,3),reshape=False)
                coord = rotate(coord,angle1,axes=(2,3),reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat,box[1:3] - size / 2) + size / 2 
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample,np.concatenate([[0],axisorder + 1]))
            coord = np.transpose(coord,np.concatenate([[0],axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:,:3] = bboxes[:,:3][:,axisorder]
            
    if ifflip:

        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1   
        # flipid 为 1 or -1， 1表示不变，-1表示翻转
        sample = np.ascontiguousarray(sample[:,::flipid[0],::flipid[1],::flipid[2]])
        coord = np.ascontiguousarray(coord[:,::flipid[0],::flipid[1],::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:   # 如果某一维图像翻转了，那么标注也做相应的翻转。用总长度减去原坐标
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:,ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord 
    # 图像，一条标注信息，所有的标注信息，coord
class Crop(object):

    def __init__(self, config):

        """
        :param config: dict 网络的配置字典
        """
        self.crop_size = config['crop_size']   # [128,128,128]
        self.bound_size = config['bound_size'] # 12
        self.stride = config['stride']         # 4
        self.pad_value = config['pad_value']   # 170

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        """
        :param imgs: 一张CT图像的 narray数据
        :param target: 一条肺结节坐标标注  D H W dia
        :param bboxes: 一张图像的所有肺结节标注
        :param isScale: bool
        :param isRand: bool
        :return: 
        crop : 来自 imgs
        """
        if isScale:
            radiusLim = [8.,120.]
            # radiusLim = [8.,100.]
            scaleLim = [0.75,1.25]

            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                         ,np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            # scale 在 0~2 之间
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size   # [128,128,128]
        bound_size = self.bound_size     #  12
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        
        start = []
        for i in range(3):
            if not isRand:   # 不随机，crop的是肺结节周边区域
                r = target[3] / 2    # 肺结节半径长度
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil (target[i] + r) + 1 + bound_size - crop_size[i] 
            else:  # img 的维度（1,D,H,W）  从图像上随意crop一块
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i]/2, imgs.shape[i + 1] / 2-bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i] / 2 + np.random.randint( -bound_size/2,bound_size/2))
                
                
        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                               np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                               np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                               indexing ='ij')
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')

        pad = []
        pad.append([0,0])
        # 如果crop之后的图像超过了边缘，就用170像素值补全，最终得到的都是完整的crop size的立方体
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
            max(start[0], 0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1], 0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2], 0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop, pad,'constant',constant_values =self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]  # 标注减去crop的原点
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]  # 标注减去crop的原点
                
        if isScale: # 如果做尺度变换的话，就把 crop,target,bboxes,coord统一缩放一下
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad > 0:
                pad2 = [[0,0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2,'constant', constant_values = self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord
        # 原始图像信息，一条标注信息，全部的标注信息，coord
class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])       # 4
        self.num_neg = int(config['num_neg'])          # 800
        self.th_neg = config['th_neg']                 # 0.02
        self.anchors = np.asarray(config['anchors'])   # [10,30,60]
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']       # 0.5
        elif phase == 'val':
            self.th_pos = config['th_pos_val']         # 1
          
    def __call__(self, input_size, target, bboxes):
        """
        usage: label = self.label_mapping(sample.shape[1:], target, bboxes)
        :param input_size: 图像的D,H，W
        :param target:  一条标注信息 坐标
        :param bboxes:  全部标注信息
        :return: 
        """
        stride = self.stride              # 4
        num_neg = self.num_neg            # 800
        th_neg = self.th_neg              # 0.01
        anchors = self.anchors            # [10,30,60]
        th_pos = self.th_pos              # 0.5 or 1
        ## struct = generate_binary_structure(3,1)
        output_size = []
        for i in range(3):
            assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)  # 输入尺寸除以4，为了创造输出的ground truth
        
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)  # label初始化为全-1
        ## label = np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0
        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z))) # 随机选取不重复的乱序的第二个参数个数的序列
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0  # 所有的label标记为0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1 # 选中的负例点label赋值为-1

        if np.isnan(target[0]):  # 以为这这是随机选取的patch，
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True 
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        # 果然，在这里将 肺结节的标注信息由 (1,4) 变成了 （1,5）； 新增的一维，表示的是肺结节的 概率
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label        

def select_samples(bbox, anchor, th, oz, oh, ow):
    # bbox 一条肺结节标注
    # [10,30,60]
    # th: th_neg 0.01
    # oz [1.5,5.5,9.5....]
    z, h, w, d = bbox  # 将bbox中的值取出来，依次赋予四个变量
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z  - (max_overlap - min_overlap) - 0.5 * np.abs(d - anchor)
        e = z  + (max_overlap - min_overlap) + 0.5 * np.abs(d - anchor) 
        mz = np.logical_and(oz >= s, oz <= e)  # 跟oz长度一样，但是都是 True or False
        iz = np.where(mz)[0] # iz 存储的是true的位置索引，加上[0]是要取出第一个记录
        
        s = h  - (max_overlap - min_overlap) - 0.5 * np.abs(d - anchor)
        e = h  + (max_overlap - min_overlap) + 0.5 * np.abs(d - anchor) 
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]
            
        s = w  - (max_overlap - min_overlap) - 0.5 * np.abs(d - anchor)
        e = w  + (max_overlap - min_overlap) + 0.5 * np.abs(d - anchor)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        
        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis = 1)
        # centers得到的是肺结节加上周围一段区域的的一个立方体块的所有体素坐标
        
        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0
        
        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))
        
        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union
        if iou >= th:
            return iz, ih, iw
        else:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]   # 添加一个假维度
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


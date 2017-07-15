#coding=utf-8
"""
类：
PostRes2d
PostRes
Rec3
Loss
GetPBB
方法：
hard_mining
nms
iou
acc
topkpbb
"""
import math
import torch

import numpy as np

from torch import nn


class PostRes2d(nn.Module):   # 暂时用不到，可以不看

    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):

        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:   # 输入和输出大小不一致的情况
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)
# 找出得分最大的K个负例，因为得分太低的负例是以高置信度和正例区分开的，不会对网络的权重改变有太多贡献，应
# 该选取和正例差不多的负例
# torch.topk 选择的是最大的K个
# neg_out 是一个列向量
# num_hard = self.num_hard * batch_size 可以理解为，每张图片选取 self.num_hard个负例，其它的都舍弃
def hard_mining(neg_output, neg_labels, num_hard): 
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class Loss(nn.Module):

    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()  # 二进制交叉熵
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train = True):
        """
        计算 分类loss z h w 上的回归loss 
        :param output: torch.Tensor (N,P,D,H,W,直径) N:batch_size P:是肺结节的概率 
        :param labels: torch.Tensor (N,P,D,H,W,直径) 
        :param train: 
        :return: 
        """
        batch_size = labels.size(0)
        # -1表示自动推断 N * 5
        output = output.view(-1, 5)  # 变成了N*5的二维矩阵
        labels = labels.view(-1, 5)

        # 概率大于0.5的作为正例
        pos_idcs = labels[:, 0] > 0.5        # 每行的第一个元素是概率
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5) # 将一列复制为5列
        # 所有的正例
        pos_output = output[pos_idcs].view(-1, 5)    # 将正例输出 每一行都完整取出来
        pos_labels = labels[pos_idcs].view(-1, 5)    # 将正例GT   每一行都完整取出来

        # 概率小于-0.5的作为负例
        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]          # 只取负例输出的概率列
        neg_labels = labels[:, 0][neg_idcs]          # 只取负例GT的概率列
        
        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)          # 负例的 Sigmoid输出

        if len(pos_output)>0:  # 如果一次前传得到的output中，有得分超过0.5的正例
            pos_prob = self.sigmoid(pos_output[:, 0])   # 正例得分的Sigmoid输出
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.data[0] for l in regress_losses]  # 坐标的 L1损失

            # 正负例最后还是转换成了sigmoid输出，然后和labels比较。
            classify_loss = 0.5 * self.classify_loss(pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum() # 整理的Sigmoid值大于0.5的个数
            pos_total = len(pos_prob)  # 所有的正例个数

        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(neg_prob, neg_labels + 1)  # 在label中，正例为1，负例为-1
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        classify_loss_data = classify_loss.data[0]           # 正负例分类损失

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)
        # 总的loss  分类的loss Z方向上的回归loss H方向上的回归loss W方向上的回归loss
        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]

class GetPBB(object):
    """
    不懂
    """
    def __init__(self, config):
        self.stride = config['stride']                #      4
        self.anchors = np.asarray(config['anchors'])  #  [10,30,60]
        self.nms_th = config['nms_th']
    def __call__(self, output,thresh = -3):
        stride = self.stride                          #      4
        anchors = self.anchors                        #  [10,30,60]
        output = np.copy(output)
        offset = (float(stride) - 1) / 2              # 1.5
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh   # 将P得分大于-3的过滤出来
        xx,yy,zz,aa = np.where(mask)     # 得到mask的下标
        
        output = output[xx,yy,zz,aa]     # 选取符合mask条件的output
        output = nms(output, self.nms_th)

        return output

def nms(output, nms_th):
    """
    应该是非极大值抑制
    :param output: 
    :param nms_th: 
    :return: 
    """
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])] # 对output按P列排序，得分越高越靠前
    bboxes = [output[0]]                       # 取第一条记录，先放在bboxes中
    
    for i in np.arange(1, len(output)):        # 从第二条记录开始进行判断
        bbox = output[i]                       # 取出一条记录
        flag = 1                               # 设置判断是否存储的标记
        for j in range(len(bboxes)):           # 将当前记录，与bboxes中的所有记录进行对比
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:  # 只要有一次两者的IOU值大于某阈值
                flag = -1                      # 将flag标记为-1
        if flag == 1:                          # 如果跟bboxes中的所有记录的bboxes值都小于某阈值
            bboxes.append(bbox)                # 将该记录添加到bboxes中
    
    bboxes = np.asarray(bboxes, np.float32)    # 将bboxes转换成 nparray格式
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th] 
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score>bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, len(lbb)    

def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp)+len(fp)<topk:
        conf_th = conf_th-0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th<-3:
            break
    tp = np.array(tp).reshape([len(tp),6])
    fp = np.array(fp).reshape([len(fp),6])
    fn = np.array(fn).reshape([len(fn),5])
    allp    = np.concatenate([tp,fp],0)
    sorting = np.argsort(allp[:,0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk,len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
    fn_i       = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp    = allp[:topk]
    if len(fn_i)>0:
        fn = np.concatenate([fn,tp[fn_i,:5]])
    else:
        fn = fn
    if len(tp_in_topk)>0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk)>0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp , fn

#coding=utf-8
"""
进行肺结节检测的网络配置文件
"""
import torch
from torch import nn
from net_layers import *

config = {}
config['thresh'] = 0                 # 对最终输出结果的概率的选取阈值  >thresh
config['nms_th'] = 0.4                 # 最终输出结果进行非极大值抑制，此为阈值
config['anchors'] = [ 10.0, 20.0, 30.0] # 三个不同的尺度，10mm,30mm,60mm
config['chanel'] = 1                   # 灰度图
config['crop_size'] = [128, 128, 128]  # 输入的 cubic patch的大小  训练的时候？
config['stride'] = 4                   # ?
config['max_stride'] = 16              # ?
config['num_neg'] = 800                # 负样本个数
config['th_neg'] = 0.02                # IOU小于0.02的为负样本
config['th_pos_train'] = 0.5           # IOU大于0.5的为正样本
config['th_pos_val'] = 1               # ？
config['num_hard'] = 2                 # 负样本hardmining参数
config['bound_size'] = 12              # ？
config['reso'] = 1                     # ？
config['sizelim'] = 6.                 # mm单位
config['sizelim2'] = 30                # mm单位
config['sizelim3'] = 40                # mm单位
config['aug_scale'] = True             # ？
config['r_rand_crop'] = 0.3            # ？ 
config['pad_value'] = 170              # 在区域外的都填充成170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}  # 样本增强类型
config['blacklist'] = [ ]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),   # 3*3*3卷积核，zero-padding，pad一个点
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        # 传入的参数，依次为
        # self.forw1 [(24, 32), (32, 32)]
        # self.forw2 [(32, 64), (64, 64)]
        # self.forw3 [(64, 64), (64, 64), (64, 64)]
        # self.forw4 [(64, 64), (64, 64), (64, 64)]
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        # 输入参数
        # self.back1 [(131, 128), (128, 128), (128, 128)]
        # self.back2 [(128, 64), (64, 64), (64, 64)]
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)   # drop 的比例 0.5？0.2？ test时需不需要？
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        out = self.preBlock(x)   # 1->24->24
        out_pool,indices0 = self.maxpool1(out)  # 128*128->64*64
        out1 = self.forw1(out_pool) # 24->32->32
        out1_pool,indices1 = self.maxpool2(out1) # 64*64->32*32
        out2 = self.forw2(out1_pool) # 32->64->64
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2) # 32*32->16*16
        out3 = self.forw3(out2_pool) # 64->64->64
        out3_pool,indices3 = self.maxpool4(out3) # 16*16->8*8
        out4 = self.forw4(out3_pool) # 64->64->64   8*8*64
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)  # 64->64  8*8->16*16
        comb3 = self.back3(torch.cat((rev3, out3), 1)) # 128->64->64->64 16*16
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3) # 64->64  16*16->32*32
        
        comb2 = self.back2(torch.cat((rev2, out2, coord), 1)) # 131->128->128->128  32*32 coord? 32*32*3？
        #comb2 = self.drop(comb2)
        out = self.output(comb2) # 128->64->15
        size = out.size()   # (1,15,52,52,52)
        out = out.view(out.size(0), out.size(1), -1) # (1,15,140608)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        # 先transpose(1,2)  -> (1,140608)
        # 再分开 -> (1,52,52,52,3,5)
        #
        return out

    
def get_model():
    net = Net()
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb

#coding=utf-8

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from importlib import import_module
import pandas

# ==========  自定义 ==========
from utils import *
from split_combine import SplitComb
from test_detect import test_detect


from preprocessing import full_prep
from config_submit import config as config_submit

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    testsplit = full_prep(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
    # testsplit = os.listdir(datapath)
    # ['xxxx.mhd','xxxx.mhd']
    filelist = [f for f in os.listdir(datapath) if f.endswith('.mhd')]

# ===========   开始 detection ========
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

# 是否跳过 detection的过程，天池肯定不能跳过
if not skip_detect:
    # 作用？
    margin = 32
    # 作用？
    sidelen = 144
    config1['datadir'] = prep_result_path
    # 先将3DCT图像进行分裂，因为如果不分裂，GPU会爆显存
    # 构造函数，不执行具体计算
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])
    # 构造函数，不执行具体计算
    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    # 构造函数，不执行具体计算
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)
    # 开始执行计算
    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])

    
exit()
# ==== 开始分类 ===========
casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
config2 = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])
casenet.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)

filename = config_submit['outputfile']



def test_casenet(model,testset):
    data_loader = DataLoader(
        testset,
        batch_size = 1,
        shuffle = False,
        num_workers = 32,
        pin_memory=True)
    #model = model.cuda()
    model.eval()
    predlist = []
    
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i,(x,coord) in enumerate(data_loader):

        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred,casePred,_ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist

config2['bboxpath'] = bbox_result_path
config2['datadir'] = prep_result_path



dataset = DataBowl3Classifier(testsplit, config2, phase = 'test')
predlist = test_casenet(casenet,dataset).T
# testsplit:测试样本的id
anstable = np.concatenate([[testsplit],predlist],axis=0).T
df = pandas.DataFrame(anstable)
df.columns={'id','cancer'}
df.to_csv(filename,index=False)

#coding=utf-8
"""
准备训练模型所需要的数据
"""
import os
import shutil
import numpy as np
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
import multiprocessing
from functools import partial
import warnings
import glob
from pprint import pprint
import matplotlib.pyplot as plt

import sys
sys.path.append('../preprocessing')

from step1 import step1_python
from step1 import visualize_2D,visualize_3D
from config_training import config

def visualize_raw_nodules():
    img_path = config['tc_data_path']
    anno_path = config['tc_annos_path'][0]
    imgs = glob.glob(img_path+'/*.mhd')
    annos = pandas.read_csv(anno_path)
    annos = annos.as_matrix()
    for img in imgs:
        img_name = img.split('/')[-1].split('.mhd')[0]
        print(img_name)
        np_img,origin,spacing,_=load_itk_image(img)
        this_annos = np.copy(annos[annos[:, 0] == img_name])
        label = []
        if(len(this_annos)>0):
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))
        label = np.array(label)
        nodules = visualize_nodule(np_img,label)
        for i in range(len(nodules)):
            visualize_3D(nodules[i])
            plt.imshow(np.ones((256, 256),dtype=np.uint8))
            plt.show()

def visualize_nodule(np_image,label):
    """
    可视化肺结节
    :param np_image: ndarray (D,H,W) 
    :param label: ndarray (D,4) (z,y,x,直径)
    :return: 
    """
    # visualize_3D(np_image)
    label = label.astype(int)
    print(label)
    nodules = []
    bias = 2
    for l in label:
        nodule = np_image[
            max(0,l[0]-l[3]/2-bias):min(np_image.shape[0],l[0]+l[3]/2+bias),
            max(0,l[1]-l[3]/2-bias):min(np_image.shape[1],l[1]+l[3]/2+bias),
            max(0,l[2]-l[3]/2-bias):min(np_image.shape[2],l[2]+l[3]/2+bias)
        ]
        print(nodule.shape)
        nodules.append(nodule)
    return nodules
def visualize_nodules():
    path = config['preprocess_result_path']
    imgs = glob.glob(path+'/*_clean.npy')
    labels = glob.glob(path+'/*_label.npy')
    print(imgs)
    print(labels)
    assert len(imgs)==len(labels),'样本数与标签数不匹配'
    for i in range(len(imgs)):
        name1 = imgs[i].split('/')[-1].split('_')[0]
        name2 = labels[i].split('/')[-1].split('_')[0]
        print(name1,name2)
        assert name1==name2,'样本与标签不匹配'
        img = np.load(imgs[i])[0]
        label = np.load(labels[i])
        nodules = visualize_nodule(img,label)
        print(len(nodules))
        visualize_3D(nodules[0].astype(np.uint8))
def start_process():
    print 'Starting',multiprocessing.current_process().name

def resample(imgs, spacing, new_spacing,order=2):
    """
    重采样，不过这部分代码好像冗余，在preprocessing模块也有
    对一个3D图像或者一组3D图像进行重采样
    :param imgs: ndarray (D,H,W) 或者  (D,H,W,NumOfImg)
    :param spacing: 
    :param new_spacing: 
    :param order: 
    :return: imgs ： ndarray (D,H,W) 或者  (D,H,W,NumOfImg)
    """
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    将世界坐标转化为对应的体素坐标，世界中心的体素坐标是固定的，就是(0,0,0)
    :param worldCoord: 
    :param origin: 
    :param spacing: 
    :return: 
    """
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    """
    SimpleITK加载mhd格式图像
    :param filename: 
    :return: 
    numpyImage   : ndarray (D,H,W) 一般是 np.int16 这个好像是图像的mask
    numpyOrigin  : ndarray (3,) 顺序为  D H W,表示世界坐标原点
    numpySpacing ：ndarray (3,) 顺序为  D H W,表示三个轴方向的间距
    isflip       ： 不太清楚  mhd文件中TransformMatrix字段为[1,0,0, 0, 1, 0, 0, 0, 1]，与之存在不一致，则isflip为真
                    应该是为了保持 图像 H W 在两个方向上的一致性
    """
    # 这个是将.mhd文件打开
    with open(filename) as f:
        contents = f.readlines()
        # 得到 TransformMatrix = 1 0 0 0 1 0 0 0 1  这一行
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        # 只要存在一个元素 不一致，isflip便为True，完全一致为False
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    # D,H,W
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    """
    和预处理模块的process_mask基本一致，略有出入
    :param mask: 
    :return: 
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    """
    代码冗余，将HU处理到0-255
    :param img: 
    :return: 
    """
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def savenpy(id,annos,filelist,data_path,prep_folder):
    """
    
    :param id: 
    :param annos: 
    :param filelist: 
    :param data_path: 
    :param prep_folder: 
    :return: 
    """
    resolution = np.array([1,1,1])
    name = filelist[id]
    label = annos[annos[:,0]==name]
    label = label[:,[3,1,2,4]].astype('float')
    
    im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
    Mask = m1+m2
    
    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)

    
    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)

    print(name)

def full_prep(step1=True,step2 = True):
    """
    完全预处理，可以直接在这上面 嫁接  天池，注意这个完全处理的是 DSB的数据
    或者，对于天池数据的处理，应该重写一个函数
    :param step1: 
    :param step2: 
    :return: 
    """
    warnings.filterwarnings("ignore")

    #preprocess_result_path = './prep_result'
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    finished_flag = '.flag_prepkaggle'
    
    if not os.path.exists(finished_flag):
        alllabelfiles = config['stage1_annos_path']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:,0]!=np.nan]
            tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)
        filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        print('starting preprocessing')
        pool = Pool()
        filelist = [f for f in os.listdir(data_path)]
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    f= open(finished_flag,"w+")        

def savenpy_luna(id,annos,filelist,luna_segment,luna_data,savepath):
    """
    真正处理Luna数据的执行体，其中包含了对肺结节标注信息的处理，应该和tianchi保持一致；
    准确地说，天池结合了DSB和Luna两者，DSB需要进行分割得到mask，而Luna本身就提供了mask，所以需要对标注进行修改
    :param id: 
    :param annos: 从csv中得到的numpy
    :param filelist: 
    :param luna_segment: 
    :param luna_data: 
    :param savepath: 
    :return: 
    """
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    # 不太清楚 根据isflip来调整mask的方向
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3 #这个应该是第一片肺叶
    m2 = Mask==4 #这个应该是第二片肺叶
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)# xx表示第一个轴的坐标；yy表示第二个轴的坐标；zz表示第三个轴的坐标
    # 3x2 [min,max]
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    # 向下取整
    box = np.floor(box).astype('int')
    margin = 5
    # 3x2 [较小的坐标，较大的坐标]
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    this_annos = np.copy(annos[annos[:,0]==int(name)])        

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        # 加载原始的CT图像
        sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath,name+'_clean.npy'),sliceim)


    if islabel:

        this_annos = np.copy(annos[annos[:,0]==int(name)])
        label = []
        # 因为一个CT图像会标注多个肺结节
        if len(this_annos)>0:
            
            for c in this_annos:
                # c应该是1D的吗，为什么是2D的；这里应该处理的是三维坐标
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                # 所以，label是 体素三维坐标，直径的体素长度  ？
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            # (1,4)
            label2 = np.array([[0,0,0,0]])
        else:
            # N x 4 =====>   4 x N
            label2 = np.copy(label).T
            # 得到体素坐标，上面不是已经得到了吗
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            # 得打直径的体素长度
            label2[3] = label2[3]*spacing[1]/resolution[1]
            # 因为进行了crop，所以要平移
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'),label2)
        
    print(name)

def preprocess_luna():
    """
    处理Luna数据，好像Luna给出了mask
    :return: 
    """
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        
        pool = Pool()
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                       luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist)
        #savenpy(1)
        _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")
    
def prepare_luna():
    """
    处理luna数据之前的一些准备操作
    :return: 
    """
    print('start changing luna name')
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):
        # luna数据集下面有很多个子文件夹
        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

#         allnames = []
#         for d in subsetdirs:
#             files = os.listdir(d)
#             names = [f[:-4] for f in files if f.endswith('mhd')]
#             allnames = allnames + names
#         allnames = np.array(allnames)
#         allnames = np.sort(allnames)

#         ids = np.arange(len(allnames)).astype('str')
#         ids = np.array(['0'*(3-len(n))+n for n in ids])
#         pds = pandas.DataFrame(np.array([ids,allnames]).T)
#         namelist = list(allnames)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]

        # 将luna的子数据集合并到一个文件夹中
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))
                print(os.path.join(luna_data,str(id)+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                print(os.path.join(luna_segment,filename+lastfix))


        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")


def full_prep_tc(n_worker=None):
    """
    自定义，对天池数据的完全预处理
    """
    warnings.filterwarnings("ignore")

    prep_folder = config['preprocess_result_path']
    data_path = config['tc_data_path']
    finished_flag = '.flag_prepTianChi'

    if not os.path.exists(finished_flag):
        # 现在只考虑  annotations.csv 文件
        all_csv_files = config['tc_annos_path']
        assert all_csv_files[0].split('/')[-1] == 'annotations.csv','请检查天池数据集annotations.csv的设置路径'
        annos = pandas.read_csv(all_csv_files[0])
        annos = annos.as_matrix()

        if not os.path.exists(prep_folder):
            os.makedirs(prep_folder)

        print('starting preprocessing')

        # 这里得到的是 全路径
        filelist = [f for f in glob.glob(data_path+'/*.mhd')]
        N = len(filelist)
        partial_savenpy = partial(savenpy_tc, annos=annos, filelist=filelist, prep_folder=prep_folder,total=N)
        pool = Pool(processes=n_worker,initializer=start_process)
        _ = pool.map(partial_savenpy, range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    f = open(finished_flag, "w+")


def savenpy_tc(id, annos, filelist, prep_folder,total):
    """
    对天池数据中一张CT的处理
    DSB和Luna数据的处理两者的整合
    :param id: filelist中的索引
    :param annos: 根据csv创建的ndarray，存放对肺结节的标注信息
    :param filelist: .mhd文件的list，存放的是全路径
    :param prep_folder: 
    :return: 
    """
    print('********** %d / %d\n'%(id+1,total))
    islabel = True
    isClean = True
    # 非固定，可以修改，但是一般为[1,1,1]
    resolution = np.array([1, 1, 1])
    # 该mhd文件的完整路径
    full_name = filelist[id]
    # CT的id
    name = full_name.split('/')[-1].split('.mhd')[0]
    print('======= begin handle %s  ======== \n'%name)

    im, m1, m2, spacing = step1_python(full_name,is_tc=True)

    Mask = m1 + m2
    # Mask是根据CT原始图像计算得到的，所以两者方向相同，应该不用考虑flip的情况
    # if isflip:
    #     Mask = Mask[:, ::-1, ::-1]
    # Mask的尺寸和原始CT图像应该完全一致吧？
    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(Mask)
    # (3,2)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    # box 变成了体素坐标
    box = np.floor(box).astype('int')
    margin = 5
    # (3,2) [较小的坐标 较大的坐标]
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T
    extendbox = extendbox.astype('int')

    if isClean:

        # 加载原始的CT图像
        _, origin, spacing1, isflip = load_itk_image(full_name)
        # assert spacing == spacing1, 'spacing不一致！'
        assert isflip == False, 'isflip 为 False！'
        if isflip:
            # sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        # 这个地方注意！！！！！！！！！
        # DSB的处理方式是  extramask = dilatedMask - Mask
        # Luna的处理方式是  extramask = dilatedMask ^ Mask
        extramask = dilatedMask - Mask
        bone_thresh = 210
        pad_value = 170
        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = sliceim * extramask > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        visualize_3D(sliceim.astype(np.uint8))

        np.save(os.path.join(prep_folder, name + '_clean.npy'), sliceim)

    if islabel:

        this_annos = np.copy(annos[annos[:, 0] == name])
        label = []
        # 因为一个CT图像会标注多个肺结节 Nx5 : id(str) X(float) Y(float) Z(float) diameter_mm（float）
        if len(this_annos) > 0:

            for c in this_annos:
                # c是1D的,只有1条记录；这里处理的应该是三维坐标
                # c[1:4][::-1] 取出 x y z 世界坐标，并且倒序，变成了 (z,y,x)
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                # note 不懂
                assert isflip == False, 'isflip 为 False！'
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                # 所以，label是 体素三维坐标，直径的体素长度  ？ note 不懂 直径的长度是沿着哪个轴度量出来的？ 好像直径的长度是沿着 y 轴测量的
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        # 该CT图像没有标注的肺结节
        if len(label) == 0:
            # (1,4)
            label2 = np.array([[0, 0, 0, 0]])
        else:
            # N x 4 =====>   4 x N
            label2 = np.copy(label).T
            # 得到体素坐标，上面不是已经得到了吗?不是已经得到 体素坐标了吗？  上面得到的只是spacing分辨率下的体素坐标，所以这一步是必须的
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            # 得到直径的体素长度
            label2[3] = label2[3] * spacing[1] / resolution[1]
            # 因为进行了crop，所以要平移
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + '_label.npy'), label2)

    print(name)

if __name__=='__main__':
    visualize_raw_nodules()
    exit()
    visualize_nodules()
    exit()
    full_prep_tc(n_worker=1)
    exit()
    # 处理DSB数据
    full_prep(step1=True,step2=True)
    # 准备Luna数据
    prepare_luna()
    # 处理 Luna数据
    preprocess_luna()
    

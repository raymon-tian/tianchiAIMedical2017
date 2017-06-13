#coding=utf-8
"""
注释以 ### 开头的，表示原作者的代码
"""
import os
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
import glob
from tqdm import tqdm

from step1 import step1_python

def process_mask(mask):
    """
    对一片肺叶的3D binary mask 进行处理
    :param mask: 其中一片肺叶的3Dmask，和CT的尺寸一致 ndarray (D,H,W) np.bool
    :return: dilatedMask 尺寸与输入一致
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        # 并非完全是背景
        if np.sum(mask1)>0:
            # 计算能完全包住mask1前景区域的最小多边形
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        # 完全是背景
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):
    """
    很重要的一步，将HU值处理到了0-255之间，所以HU在[-1200,600]会得到有效的保留，这样的处理是很合理的
    1.img的所有元素+1200，再除以1800
    2.<0以及<1的元素分别设置为0和1 
    3.逐个元素乘以255
    :param img: ndarray (D,H,W) np.int16 
    :return: newimg: ndarray (D,H,W) np.uint8,0-255之间
    """
    lungwin = np.array([-1200.,600.])
    # 将img中所有元素先+1200，然后再除以1800
    # 所以HU超过600的，会大于1
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    # 将HU值处理到了0-255的区间
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    """
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

def savenpy(id,filelist,prep_folder,data_path,use_existing=True):
    """
    线程函数，执行事实上的数据处理
    :param id: int 当前处理的CT图像的索引
    :param filelist: [str] 包含所有CT图像名的list
    :param prep_folder: str 预处理结果存储路径
    :param data_path: str 数据存放基路径
    :param use_existing: bool 是否使用已经存在的预处理结果
    :return: 
    """
    print('%d / %d\n'%(id,len(filelist)))
    resolution = np.array([1,1,1])
    ### name = filelist[id]
    name = filelist[id].replace('.mhd','')
    print('start handle  '+name)

    if use_existing:
        # xxx_label.npy 以及 xxx_clean.npy都在prep_folder中存在的话，说明该CT图像的预处理已经完毕
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        # 得到CT图像的3D图像，以及两个肺叶的mask，各个轴之间的距离
        im, m1, m2, spacing = step1_python(os.path.join(data_path,filelist[id]))
        Mask = m1+m2

        # 将尺寸进行统一的设置
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        # Mask每一个元素都为False的情况下，xx yy zz的值都为空，出现bug
        xx,yy,zz= np.where(Mask)
        if(len(xx)==0 or len(yy)==0 or len(zz)==0):
            print('bug in ' + name +' 请之后务必处理它！！！')
            return
        assert len(xx)!=0 and len(yy)!=0 and len(zz)!=0
        # 3x2 box 还是进行尺寸的统一设置
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')#向下取整
        margin = 5
        # extendbox ： 3x2 左面为 max:x y z 右面为 min:x y z
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],axis=0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')



        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2 # 将两片肺叶的mask合并起来
        Mask = m1+m2
        extramask = dilatedMask ^ Mask # 按位亦或？
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000# 值为nan的地方重新设置为-2000
        sliceim = lumTrans(im)
        #
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        # (1,D,H,W)
        sliceim = sliceim2[np.newaxis,...]
        # 将 sliceim 与 np.array([[0,0,0,0]]) 进行存盘
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        # xxx_label np.array([[0,0,0,0]]) (1,4)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
    except:
        print('bug in '+name)
        raise
    print(name+' done')

    
def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
    """
    完全预处理，将预处理结果存盘
    :param data_path: CT图像存放路径
    :param prep_folder: 预处理结果存放路径
    :param n_worker: 预处理线程数
    :param use_existing: bool 
    :return: mhd文件文件名的list 
    filelist
    """
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

            
    print('starting preprocessing')
    # n_worker为None的时候，表示开启全部线程数
    pool = Pool(n_worker)#Python多线程编程
    ### filelist = [f for f in os.listdir(data_path)]
    filelist = [f for f in os.listdir(data_path) if f.endswith('.mhd')]

    """
    因为一些CT图像目前预处理方面有bug，所以，暂时跳过这些CT图像
    """
    blocklist = ['LKDS-00383','LKDS-00439','LKDS-00300']
    filelist = [f for f in filelist if f not in blocklist]

    
    # filelist = glob.glob(data_path+'*.mhd')
    # partial：内建对象，对可调用对象进行操作
    # partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
    #                          data_path=data_path,use_existing=use_existing)

    # CT图像的总数目
    N = len(filelist)

    for i in tqdm(range(N)):
        savenpy(id=i,filelist=filelist,
                prep_folder=prep_folder,
                data_path=data_path,
                use_existing=use_existing)
        # partial_savenpy(i)

    # _=pool.map(partial_savenpy,range(N))
    # pool.close()
    # pool.join()
    print('======   end preprocessing   ========= \n')
    return filelist

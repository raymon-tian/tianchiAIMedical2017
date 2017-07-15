#coding=utf-8

import os
import glob
import h5py
import warnings
import scipy.ndimage

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas

from tqdm import tqdm
from skimage import measure
from scipy.io import loadmat
from functools import partial
from multiprocessing import Pool
from scipy.ndimage.interpolation import zoom
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure

from config_train import config as train_config


def visualize_2D(*args):  # 显示2D图像，可并排输入多个2D矩阵并可视化
    num_plots = len(args)                     # 显示图片数目
    f,axarr = plt.subplots(1,num_plots)       # 画出相应数据 subplot
    for i in range(num_plots): 
        if(num_plots == 1):                   # 若只有一个输入
            axarr.imshow(args[i])
        else:                                 # 若有多个输入
            axarr[i].imshow(args[i])
    plt.show()                                # 显示图像

def visualize_3D(np_image): # 显示3D CT图像
    sitk_img = sitk.GetImageFromArray(np_image,isVector=False)  # 将数组转换成sitk格式数据
    sitk.Show(sitk_img)                                         # 用sitk数据包显示3D CT图像

def visualize_raw_nodules(train_data_path, train_annos_path):
    img_path = train_data_path              # 训练数据路径标注 annotations.csv
    anno_path = train_annos_path            # 训练
    imgs = glob.glob(img_path+'*.mhd')      # 返回路径下所有以.mhd结尾的文件
    annos = pandas.read_csv(anno_path)      # 读取CSV标注文件
    annos = annos.as_matrix()               # 将CSV文件转换成矩阵表示
    for img in imgs:                        # 对每一个.mhd文件
        img_name = img.split('/')[-1].split('.mhd')[0]           # 获取文件名，不带后缀
        print(img_name)
        np_img,origin,spacing,_=load_itk_image(img)              # 读取图像数据，坐标原点，各轴宽度
        this_annos = np.copy(annos[annos[:, 0] == img_name])     # 在CSV文件中，将本图像相关的标注取出来
        label = []                                               
        if(len(this_annos)>0):                                   # 若本图像中有肺结节标注                         
            for c in this_annos:                                 # 针对每一个标注
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)      # 将世界坐标转换成体素坐标
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))                   # 将肺结节直径转化成以像素为坐标
        label = np.array(label)                                                            # label中保存一些列肺结节的信息（D，H，W, 直径）
        nodules = visualize_nodule(np_img,label)                                           # 输入图像和label信息，将所有肺结节的立方体块切割出来

        # for i in range(len(nodules)):                                                    # 可视化 3d 肺结节
            # visualize_3D(nodules[i])
            # plt.imshow(np.ones((256, 256),dtype=np.uint8))
            # plt.show()


def load_itk_image(filename):
    """
    SimpleITK加载mhd格式图像
    :param filename: .mhd文件路径
    :return: 
    numpyImage   : ndarray (D,H,W) 一般是 np.int16 这个好像是图像的mask
    numpyOrigin  : ndarray (3,) 顺序为  D H W,表示世界坐标原点
    numpySpacing ：ndarray (3,) 顺序为  D H W,表示三个轴方向的间距
    isflip       ： 不太清楚  mhd文件中TransformMatrix字段为[1,0,0, 0, 1, 0, 0, 0, 1]，与之存在不一致，则isflip为真
                    应该是为了保持 图像 H W 在两个方向上的一致性
    """
    # 这个是将.mhd文件打开
    with open(filename) as f:          # 打开.mhd文件
        contents = f.readlines()       # 读取数据
        line = [k for k in contents if k.startswith('TransformMatrix')][0]          # 取出 TransformMatrix 一行
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')      # 将坐标内容转换成浮点数组
        transformM = np.round(transformM)                                           # 将浮点数四舍五入
        
        if np.any( transformM!=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):              # 只要存在一个元素 不一致，isflip便为True，完全一致为False
            isflip = True
        else:
            isflip = False

    itk_image = sitk.ReadImage(filename)
    
    numpyImage = sitk.GetArrayFromImage(itk_image)  # D,H,W   第一维为啥纵轴
     
    numpyOrigin = np.array(list(reversed(itk_image.GetOrigin())))  # D,H,W         # mhd文件中最后一维是纵轴，需要做一个转换，使其与图像一致
    numpySpacing = np.array(list(reversed(itk_image.GetSpacing()))) # D,H,W
     
    return numpyImage, numpyOrigin, numpySpacing, isflip

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

def visualize_nodule(np_image,label):  
    """
    可视化肺结节
    :param np_image: ndarray (D,H,W) 
    :param label: ndarray (D,4) (z,y,x,直径)
    :return: 
    """
    # visualize_3D(np_image)
    label = label.astype(int)      # 将label数据取整，因为是大致的圈出来，不要求亚像素级别精度
    print(label)
    nodules = []
    bias = 2                       # 在给出的直径基础上，再向四周扩2体素距离
    for l in label:                # 对每一个肺结节
        nodule = np_image[
            max(0,l[0]-l[3]/2-bias):min(np_image.shape[0],l[0]+l[3]/2+bias),      # 确定 D 轴边界
            max(0,l[1]-l[3]/2-bias):min(np_image.shape[1],l[1]+l[3]/2+bias),      # 确定 H 轴边界
            max(0,l[2]-l[3]/2-bias):min(np_image.shape[2],l[2]+l[3]/2+bias)       # 确定 W 轴边界
        ]
        print(nodule.shape)
        nodules.append(nodule)                                                    # 将切割出来的肺结节图像 添加到 noludles list中
    return nodules




def print_scan_info(img):

    print('size',img.GetSize())
    print('origin',img.GetOrigin())
    print('spacing',img.GetSpacing())
    print('direction',img.GetDirection())
    print('height',img.GetHeight())
    print('width',img.GetWidth())
    print('depth',img.GetDepth())
    print('NumberOfComponentsPerPixel',img.GetNumberOfComponentsPerPixel())
    print('Dimension',img.GetDimension())
    print('PixelIDValue',img.GetPixelIDValue())
    print('PixelIDTypeAsString',img.GetPixelIDTypeAsString())

def load_scan(path,verbose=False):

    scan = sitk.ReadImage(path)
    if(verbose == True):
        print_scan_info(scan)
    return scan


def get_pixels_hu(slices):
    """
    将SimpleITK给出的图像对象形式转变为numpy形式，返回
    image : ndarray np.int16 (D,H,W)
    spacing : ndarray np.float32 (3,) [D,H,w]
    :param slices: 
    :return: 
    """
    image = sitk.GetArrayFromImage(slices) # (z,y,x) (D,H,W)
    image = np.array(image,dtype=np.int16)
    numpySpacing = np.array(list(reversed(slices.GetSpacing())))
    return image,numpySpacing

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    """
    目的：对每一个slice进行 true  False 的二值化；
    核心算法：求解一系列的连通域，然后将不合适的连通域去除掉
    label = measure.label(current_bw)# 为毗邻的pixel进行label
    properties = measure.regionprops(label)# 度量各个label区域的属性

    涉及到一些参数    
    :param image: image.shape[0]索引图像的slice数量，轴的顺序为 (D,H,W)
    :param spacing: 就是CT图像各个轴之间的间隙，同样的，spacing[0]表示CT深度方向，轴的顺序为 (D,H,W)
    :param intensity_th: 使用该值进行二值操作；按照wikipedia的说法，肺的HU在-700～-600
    :param sigma: 高斯滤波方差
    :param area_th: 连通域面积阈值
    :param eccen_th: 离心率
    :param bg_patch_size: 对一个slice 左上拐角边角区域的选择
    :return: image一样维度的3D mask，只包含 0 1
    """

    assert image.shape[1]==image.shape[2],'CT的H与W不一致，可能出现error'
    #================     得到单张图片的内切圆的  mask ====================

    bw = np.zeros(image.shape, dtype=bool)  # 初始化一个3D mask, 全为False
    

    # 最终结果的肺结节直径是以mm为单位，不论进不进行尺度变换，最后都要统一为mm单位
 
    image_size = image.shape[1]  # 单张slices，宽高一致，是正方形
    
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size) # 将中心点设置到图像的最中心处
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5

    nan_mask = (d<image_size/2).astype(float) # 内切圆内部元素置为1，外部元素置为0

    nan_mask[nan_mask == 0] = np.nan

    assert nan_mask.shape[0] == image.shape[1]

    #================     循环处理每一个slice ====================
    for i in range(image.shape[0]):
        #=========  进行 高斯滤波  ============
        # 判断一下图像左上角10*10区域的元素是否相等
        # current_bw 的 H W 与 原始CT的H W 一致；current_bw只有两类label True False
        # 这里用 大于号 还是 小于号，值取-600 还是其他？
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            # !
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            # !
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th

        current_bw0 = current_bw

        # ==========   选择合适的连通区域 =========
        label = measure.label(current_bw)# 联通区域标记
        properties = measure.regionprops(label)# 返回联通区域属性列表
        valid_label = set()
        for prop in properties:  # 对每一个连通区域
            # ! area表示该连通域中像素的个数  eccentricity:离心率  area_th 单位应该是平方毫米，这个取值也要调试
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label) # label是int，是连通域的ID
        # current_bw依旧为 2值 的
        # current_bw = np.isin(label, list(valid_label)).reshape(label.shape)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

        # if(i%50 == 0):
        #     visualize_2D(image[i],current_bw0,current_bw)

    assert bw.shape == image.shape
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    """
    :param bw: 3D mask True 和 False 两种
    :param spacing: CT图像各个轴之间的间距
    :param cut_num: 这里是切掉的层数，用剩下的层数来计算
    :param vol_limit: [0.68,7.5]
    :param area_th: 科学计算法 6000
    :param dist_th: 
    :return: bw : 3d mask 只有 True False两种取值, len(valid_label)
    """
    bw_input = bw
    print('start all_slice_analysis')
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        # 倒数cut_num层直接置为False？   从后往前切
        bw[-cut_num:] = False
    # 只考虑4位邻居
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2) # 不清楚是哪个维度  高还是宽？ 应该是W
    # 选了12种背景点: 立方体的8个角，四个楞的中点
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    # 将跟上述的12个点属于同一个连通域的点，都设为0，背景点
    for l in bg_label:
        label[label == l] = 0

    properties = measure.regionprops(label)
    for prop in properties:
        # !进一步将过小或者过大的区域设置为0
        # prod 就是 reduce_multiply； 将  体积 小于 680000 mm^3 以及  大于 8200000 mm^3 的区域设置为 背景
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols: # 对每一个连通域来说
        single_vol = label == vol.label             # 仅仅是一个连通域
        slice_area = np.zeros(label.shape[0])       # 存放每一个slice中连通域的面积
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]): # 对每一层来说
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3]) #某一层某一个连通域的面积
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d)) # 该连通域中离正方形中心最近的距离
        # ! 选取合适 label 的条件
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
        # 只将肺提取出来，外围背景区域和小区域都设为False,肺设为True    
    # bw = np.isin(label, list(valid_label)).reshape(label.shape)
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]  # 将切掉的部分补充回bw1
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num) # 将bw形态学扩张
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})                   # 除去背景的连通域ID
        valid_l3 = set()
        for l in l_list:  #这个不明白，应该是进一步筛选
            indices = np.nonzero(label==l)  # 满足label==1的坐标
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]] # l3 就是 该连通域中 坐标最小的位置处
            if l3 > 0:
                valid_l3.add(l3)                                     # 选取共有的前景
        # bw = np.isin(label3, list(valid_l3)).reshape(label3.shape)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    assert bw.shape == bw_input.shape
    return bw, len(valid_label)

def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    """
    切割出来两片肺实质，
    返回 dw1 dw2 dw，表示的是三个3D 二值mask，3个mask尺寸应该完全一样
    :param bw: 3D 二值掩码 只含  True  False
    :param spacing: CT图像各个轴之间的距离
    :param max_iter: 
    :param max_ratio: 
    :return: 
    """
    # !
    def extract_main(bw, cover=0.95):
        """
        提取肺实质
        :param bw: ndarray 3D mask 二值
        :param cover: float
        :return: 
        """
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)# 会把背景忽略的
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        """

        :param bw:
        :return:
        """
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        # area表示该连通区域内的像素/体素的个数
        properties.sort(key=lambda x: x.area, reverse=True)
        # !!! 直观上理解，肯定上2片肺部面积最大 找到两片肺叶的条件; 大肺叶/小肺叶 应该 < 4.8
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            # 如果还没有找到两片肺叶，就不断地进行erosion操作
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        # !
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        assert bw1.shape == bw0.shape and bw2.shape == bw0.shape

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    assert  bw0.shape == bw1.shape and bw0.shape == bw2.shape
    return bw1, bw2, bw

def fill_hole(bw):
    """
    :param bw: ndarray : 3d mask
    :return: 
    """
    # fill 3d holes
    label = measure.label(~bw)# 逐个元素取反，4邻居3D求连通域
    # idendify corner components 3D的，必然有8个角
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    # numpy.in1d(a,b)：只要a中的元素在b中存在，那么就在a的此元素位置处设置为True
    # bw = ~np.isin(label, list(bg_label)).reshape(label.shape)
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def cal_mask(case_path):
    """
    处理一个CT图像的
    :param case_path: 一个CT图像的完整路径
    :return: 
    case_pixels ： ndarray (D,H,W) np.int16 最原始的 numpy CT图像
    bw1         ： ndarray (D,H,W) np.bool 第一片肺叶掩码
    bw2         ： ndarray (D,H,W) np.bool 第二片肺叶掩码
    spacing     ： 各个轴之间的间距
    """

    case = load_scan(case_path)      # 从*.mhd文件中读取SimpleITK格式图像
    case_pixels, spacing = get_pixels_hu(case) # 得到3D ndarray图像和各轴上的Spacing [z,y,x]
    bw = binarize_per_slice(case_pixels, spacing)
    print('binarize_per_slice done')
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    # bw的第一个维度是 深度
    # 当全部切片不好算时，就切掉几层，用剩下的来算，直到全部切完或者能找到合适的肺部mask
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        # 传入binarize_per_slice生成的 3D mask
        # !
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
        print('flag '+str(flag))
        print('%d ====> %d'%(cut_num,bw.shape[0]))

    print('all_slice_analysis done')
    # print('bw shape ',bw.shape)
    # visualize_3D(bw.astype(np.uint8))
    bw = fill_hole(bw)
    print('fill_hole done')
    # visualize_3D(bw.astype(np.uint8))
    # print('fill_hole over')
    bw1, bw2, bw = two_lung_only(bw, spacing)
    print('two_lung_only done')
    # visualize_3D(bw1.astype(np.uint8))
    # visualize_3D(bw2.astype(np.uint8))

    assert case_pixels.shape == bw1.shape and case_pixels.shape == bw2.shape
    return case_pixels, bw1, bw2, spacing

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
        true_spacing = spacing * imgs.shape / new_shape         # 避免精度上的损失
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

def process_mask(mask):
    """
    对一片肺叶的3D binary mask 进行处理
    :param mask: 其中一片肺叶的3Dmask，和CT的尺寸一致 ndarray (D,H,W) np.bool
    :return: dilatedMask 尺寸与输入一致
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        # 将mask1中的所有元素累计求和，并非完全是背景
        if np.sum(mask1)>0:
            # 计算能完全包住mask1前景区域的最小多边形
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):  # 这里取2还是1.5 扩张得过于严重，那么就让就取消 convex_hull_image 操作
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
    :param img: ndarray (D,H,W) np.int16 一般传入的是最原始的 CT numpy图像
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


def savenpy(id,annos=None,img_list=None,prep_folder=None,use_existing=True):
    """
    线程函数，每次对一张CT图像进行处理
    :param id: img_list中图像的索引
    :param img_list: 包含所有CT图像名的list(*.mhd)
    :param prep_folder: 预处理结果存储路径
    :param use_existing: 是否使用已经存在的预处理结果
    :return: 直接将.npy数据保存在硬盘中，不返回数据
    """
    print('********** %d / %d **********'%(id+1,len(img_list)))
    resolution = np.array([1,1,1])     # 非固定，可以修改，但是一般为[1,1,1]
    full_name = img_list[id]  
    name = full_name.split('/')[-1].split('.mhd')[0]  # CT 图像名称，不带后缀

    if use_existing:
        # xxx_label.npy 以及 xxx_clean.npy都在prep_folder中存在的话，说明该CT图像的预处理已经完毕
        existing_flag = os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy'))
        if existing_flag:
            print(name+' had been done')
            return
    try:
        print('======= begin handle %s  ======== '%name)
        # 得到最原始CT图像的3D numpy 图像，以及两个肺叶的mask，各个轴之间的距离
        im, m1, m2, spacing = cal_mask(full_name)
        Mask = m1+m2

        i = 0
        # while(i < im.shape[0]):
        #     visualize_2D(im[i],Mask[i],m1[i],m2[i])
        #     i += 50

        # 将尺寸进行统一的设置
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        # Mask每一个元素都为False的情况下，xx yy zz的值都为空，出现bug
        # numpy.where 当只传入condition的时候，就相当与 numpy.nonzero
        xx,yy,zz= np.where(Mask)
        if(len(xx)==0 or len(yy)==0 or len(zz)==0):
            # 这个情况下，说明得到的mask，所有元素都为 0
            print('bug in ' + name +' 请之后务必处理它！！！')
            return
        assert len(xx)!=0 and len(yy)!=0 and len(zz)!=0
        # 3x2 box 还是进行尺寸的统一设置
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        # np.expand_dims(spacing,1) 为 (3,1)
        box = box * np.expand_dims(spacing,1)/np.expand_dims(resolution,1)  # 得到resolution分辨率下的，最顶、最底 前景元素的坐标
        box = np.floor(box).astype('int')#向下取整
        margin = 5
        # extendbox ： 3x2 左面为 max:x y z 右面为 min:x y z
        # 基于 最顶 最底 两个前景体素的坐标，再考虑一个 margin
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],axis=0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T  # 第一个是axis=0,还是直接写0
        extendbox = extendbox.astype('int')

        # 加载原始的CT图像
        # 其实这一步不太需要，我们应该仅仅想要的是  origin
        # 不太清楚 flip 是干什么的
        numpyImage_org, origin, spacing1, isflip = load_itk_image(full_name)
        assert isflip == False, 'isflip 为 False！'
        if isflip:
            print('flip!')

        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2 # 将两片肺叶的mask合并起来

        # 这个地方注意！！！！！！！！！
        # DSB的处理方式是  extramask = dilatedMask - Mask
        # Luna的处理方式是  extramask = dilatedMask ^ Mask
        # 其实 - 就是 ^ ，都是表示 抑或
        extramask = dilatedMask ^ Mask # 我们的数据应该是和Luna更相近
        bone_thresh = 210  # 换算到HU为：282，差不多是骨头
        pad_value = 170    # 换算到HU为：0，是 水

        # 这个地方应该不是必须的，因为 im 一直没有被更改过
        im[np.isnan(im)]=-2000# 值为nan的地方重新设置为-2000
        sliceim = lumTrans(im)  # 将值转换到 0-255之间
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')   # 进行填充
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value # 将骨头的地方也填充为 170
        assert  sliceim.shape == im.shape == numpyImage_org.shape

        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)


        # 这个地方进行了crop，丢失了一部分信息； 先进行了插值，然后才进行的Crop

        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        # 存储crop的信息
        np.save(os.path.join(prep_folder, name + '_crop.npy'), extendbox)
        # 存储未 插值的 CT
        # np.save(os.path.join(prep_folder, name + '_clean_no_resample.npy'), sliceim)
        # (1,D,H,W)
        # assert numpyImage_org.shape == sliceim2.shape
        sliceim = sliceim2[np.newaxis,...]
        # 将 sliceim 与 np.array([[0,0,0,0]]) 进行存盘
        np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)
        # xxx_label np.array([[0,0,0,0]]) (1,4)

        if np.all(annos == None):
            np.save(os.path.join(prep_folder, name + '_label'), np.array([[0, 0, 0, 0]]))
            return
        this_annos = np.copy(annos[annos[:, 0] == name])
        label = []
        # 因为一个CT图像会标注多个肺结节 Nx5 : id(str) X(float) Y(float) Z(float) diameter_mm（float）
        if len(this_annos) > 0:
            for c in this_annos:
                # c是1D的,只有1条记录；这里处理的应该是三维坐标
                # c[1:4][::-1] 取出 x y z 世界坐标，并且倒序，变成了 (z,y,x); 转换为 spacing分辨率下的 体素坐标
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                assert isflip == False, 'isflip 为 True！'
                if isflip:  # 这是什么意思？
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
    except:
        print('bug in '+name)
        raise
    print(name+' done\n')

if __name__ == '__main__':

    data_path = train_config['data_path']
    pre_result_path = train_config['preprocess_result_path']

    data_list = glob.glob(data_path+'/*.mhd')
    pre_re_list = glob.glob(pre_result_path+'/*_clean_no_resample.npy')

    data_name = [f.split('/')[-1].split('.mhd')[0] for f in data_list]
    pre_name = [f.split('/')[-1].split('_clean_no_resample.npy')[0] for f in pre_re_list]

    np_data = np.array([data_list,data_name]).T
    np_pre = np.array([pre_re_list,pre_name]).T

    df_data = pandas.DataFrame(np_data,columns=['data_path','name'])
    df_pre = pandas.DataFrame(np_pre,columns=['pre_path','name'])

    print(df_data.head())
    print(df_pre.head())
    df_merge = df_data.merge(df_pre,on='name')
    np_merge = df_merge.as_matrix()

    print(df_merge.head())

    for i in range(8):
        idx = np.random.randint(np_merge.shape[0])
        print(np_merge[idx,0])
        print(np_merge[idx,2])
        idx = i
        sitk_img = load_scan(np_merge[idx,0])
        np_img,_ = get_pixels_hu(sitk_img)
        np_pre = np.load(np_merge[idx,2])
        j = 0
        while(j<np_img.shape[0]):
            visualize_2D(np_img[j],np_pre[j])
            # visualize_2D(np_img[j])
            # visualize_2D(np_pre[j])
            j+=50















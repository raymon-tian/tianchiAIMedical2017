#coding=utf-8
"""
# ==========   笔记   ============
1. dicom操作
import dicom
plan = dicom.read_file("rtplan.dcm") # (rtplan.dcm is in the testfiles directory)
In [10]: plan.ImagePositionPatient
Out[10]: ['-151.493508', '-36.6564417', '1295'] 这个前两个坐标就一个CT图像而言，是不变的
2. SimpleITK 中的 numpy 索引顺序为 (z,y,x) (D,H,W) 可以视为 通道在前
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
import SimpleITK as sitk


def visualize_2D(*args):
    num_plots = len(args)
    f,axarr = plt.subplots(1,num_plots)
    for i in range(num_plots):
        if(num_plots == 1):
            axarr.imshow(args[i])
        else:
            axarr[i].imshow(args[i])
    plt.show()

def visualize_3D(np_image):
    sikt_img = sitk.GetImageFromArray(np_image,isVector=False)
    sitk.Show(sikt_img)

def load_scan(path):
    """
    加载一个CT图像；这里有个问题，好像DSB中，一个CT图像为一个文件夹，文件夹下放置很多slice，
    那么这个slice是2D的，还是3D的？应该是2D的
    这个不适用于天池
    :param path: 一个CT图像的完整路径
    :return: [slice,slice,...,slice] 
    """
    # 这里，一个scan是一个文件夹，文件夹下有很多slice，这个是与mhd不同的
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # ImagePositionPatient[2] 应该在 z轴世界坐标
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:# z轴上的比较
        sec_num = 2;
        # 这里超出索引怎么办
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def print_scan_tc_info(img):

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

def load_scan_tc(path,verbose=False):

    scan = sitk.ReadImage(path)
    if(verbose == True):
        print_scan_tc_info(scan)
    # scan_np = sitk.GetArrayFromImage(scan)
    return scan
    # return scan_np

def get_pixels_hu_tc(slices):
    """
    将SimpleITK给出的图像对象形式转变为numpy形式，返回
    image : ndarray np.int16 (D,H,W)
    spacing : ndarray np.float32 (3,) [D,H,w]
    :param slices: 
    :return: 
    """
    image = sitk.GetArrayFromImage(slices) # (z,y,x) (D,H,W)
    image = np.array(image,dtype=np.int16)
    # spacing = slices.GetSpacing()[::-1]
    # spacing = np.array(spacing,dtype=np.float32)
    numpySpacing = np.array(list(reversed(slices.GetSpacing())))
    return image,numpySpacing


def get_pixels_hu(slices):
    """
    这个好像是把原始data转变成HU值，并且返回 image:ndarray 各个轴之间的距离：spacing  ndarray (3,)
    但是不太清楚方向，现在
    返回的是 image: (D,H,W) np.int16   spacing : (3,) [D,H,W] np.float32
    本质上
    :param slices: list of slice  其实可以在load_scan中将其处理为ndarray
    :return: np.array(image, dtype=np.int16) np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)
    """
    image = np.stack([s.pixel_array for s in slices])# 深度为 axis=0
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    # SimpleITK中CT就是int16
    image = image.astype(np.int16)
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)

def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    """
    image.shape[0]索引CT的slice数量;将每一个slice进行 true  False 的二值化；
    核心的是：求解一些列的连通域，然后将不合适的连通域去除掉
    label = measure.label(current_bw)# 为毗邻的pixel进行label
    properties = measure.regionprops(label)# 度量各个label区域的属性
    涉及到一些参数    
    :param image: image.shape[0]索引图像的深度，轴的顺序为 (D,H,W)
    :param spacing: 就是CT图像各个轴之间的间隙，同样的，spacing[0]表示CT深度方向，轴的顺序为 (D,H,W)
    :param intensity_th: 使用该值进行二值操作；按照wikipedia的说法，肺的HU在-700～-600
    :param sigma: 设置高斯滤波
    :param area_th: 连通域面积阈值
    :param eccen_th: 离心率
    :param bg_patch_size: 对一个slice 左上拐角边角区域的选择
    :return: 
    与image一样维度的3D mask，只包含 0 1
    注意：
    1. 一定检测 CT的 H和W 是否相等
    流程：
    1.
    """
    # visualize_3D(image)
    # print(image.shape)
    assert image.shape[1]==image.shape[2],'CT的H与W不一致，可能出现error'
    #================     这一部分，先得到一个内切球的  mask ====================
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    # 不清楚是哪一个维度，但是按道理应该是较短的那个，不过好像天池CT的高宽都是相同的
    # 应该按照度量肺结节直径的那个轴，因为要将整个肺结节包含进去。。。按照后面的代码，直径的度量是在H轴
    image_size = image.shape[1]
    # 将中心点设置到图像的最中心处
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    # 很显然是 点  到  中心原点 的距离
    d = (x**2+y**2)**0.5
    # 以图像最中心处为球心，以image_size/2为半径的球形区域内设置为1，之外设置为0
    # 其实，本质上就是去做一个内切球，球内的部分保留，球外的部分去除
    # nan_mask的尺寸为  image.shape[1] x image.shape[1]
    nan_mask = (d<image_size/2).astype(float)
    # visualize_2D(nan_mask.astype(int))
    nan_mask[nan_mask == 0] = np.nan
    # 循环处理每一个slice
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        # bg_patch_size x bg_patch_size 的patch的所有元素全部相同
        #=========  进行 高斯滤波  ============
        # bug nan_mask 与 image[i]的尺寸可能不相等
        # 一个slice的左上角
        # current_bw 的 H W 与 原始CT的H W 一致；current_bw只有两类label True False
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th

        # select proper components
        # ==========   选择合适的连通区域 =========
        label = measure.label(current_bw)# 为毗邻的pixel进行label
        # visualize_2D(image[i], current_bw,label)
        properties = measure.regionprops(label)# 度量各个label区域的属性
        valid_label = set()
        for prop in properties:
            # area表示该连通域中像素的个数  eccentricity:离心率
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label) # label是int
        # current_bw依旧为 2值 的
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        # if(i%10==0):
        #     visualize_2D(image[i], label,current_bw)
    # visualize_3D(bw.astype(np.uint8))
    # exit()
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    """
    
    :param bw: 3D mask
    :param spacing: CT图像各个轴之间的间距
    :param cut_num: 
    :param vol_limit: 
    :param area_th: 
    :param dist_th: 
    :return: bw : 3d mask 只有 True False两种取值, len(valid_label)
    """
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        # 倒数cut_num层直接置为False？
        bw[-cut_num:] = False
    # 只考虑4位邻居
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2) # 不清楚是哪个维度  高还是宽？ 应该是W
    # 选了12种背景点
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    # 设置背景，并且将背景设置为 0
    for l in bg_label:
        label[label == l] = 0
    # visualize_3D(label.astype(np.uint8))
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        # 进一步设置背景区域 该联通区域的面积
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
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)

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
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw




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
    def extract_main(bw, cover=0.95):
        """
        提取肺实质
        :param bw: 
        :param cover: 
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
        # 直观上理解，肯定上2片肺部面积最大 找到两片肺叶的条件
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            # 如果还没有找到两片肺叶，就不断地进行erosion操作
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python(case_path,is_tc=True):
    """
    处理一个CT图像的pipeline
    :param case_path: 一个CT图像的完整路径
    :param is_tc: 处理数据是否为 天池 的数据
    :return: 
    case_pixels ： ndarray (D,H,W) np.int16 原始的CT图像
    bw1         ： ndarray (D,H,W) np.bool 第一片肺叶掩码
    bw2         ： ndarray (D,H,W) np.bool 第二片肺叶掩码
    spacing     ： 各个轴之间的间距
    """
    if(is_tc==True):
        case = load_scan_tc(case_path)
        case_pixels, spacing = get_pixels_hu_tc(case)
    else:
        case = load_scan(case_path)
        case_pixels, spacing = get_pixels_hu(case)

    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    # bw的第一个维度是 深度
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        # 传入binarize_per_slice生成的 3D mask
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
    # print('bw shape ',bw.shape)
    # visualize_3D(bw.astype(np.uint8))
    bw = fill_hole(bw)
    # visualize_3D(bw.astype(np.uint8))
    # print('fill_hole over')
    bw1, bw2, bw = two_lung_only(bw, spacing)
    # visualize_3D(bw1.astype(np.uint8))
    # visualize_3D(bw2.astype(np.uint8))
    return case_pixels, bw1, bw2, spacing
    
if __name__ == '__main__':
    # INPUT_FOLDER = '/work/DataBowl3/stage1/stage1/'
    # patients = os.listdir(INPUT_FOLDER)
    # patients.sort()
    # case_pixels, m1, m2, spacing = step1_python(os.path.join(INPUT_FOLDER,patients[25]))
    # INPUT_FOLDER = '/home/raymon/Downloads/series-000001'
    INPUT_FOLDER = '/mnt/winC/Users/raymon/Desktop/dongwang/train_subset00/LKDS-00001.mhd'
    # paths = os.listdir(INPUT_FOLDER)
    # slices = [dicom.read_file(INPUT_FOLDER+'/'+p) for p in paths]
    # slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # for s in slices:
    #     print s.ImagePositionPatient
    case_pixels, m1, m2, spacing = step1_python(INPUT_FOLDER,is_tc=True)
    plt.imshow(m1[60])
    plt.show()
    plt.figure()
    plt.imshow(m2[60])
    plt.show()
#     first_patient = load_scan(INPUT_FOLDER + patients[25])
#     first_patient_pixels, spacing = get_pixels_hu(first_patient)
#     plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()
    
#     # Show some slice in the middle
#     h = 80
#     plt.imshow(first_patient_pixels[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw = binarize_per_slice(first_patient_pixels, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     flag = 0
#     cut_num = 0
#     while flag == 0:
#         bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num)
#         cut_num = cut_num + 1
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw = fill_hole(bw)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
    
#     bw1, bw2, bw = two_lung_only(bw, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

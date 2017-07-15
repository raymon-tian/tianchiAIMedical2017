# coding=utf-8
import numpy as np
import os
import glob
import SimpleITK as sitk
import pandas
import numpy as np

from config_test import config as config_test

csv_path = config_test['outputfile']  # 存放最终提交的csv文件的路径
pbb_path = config_test['test_result_path']  # pbb文件路径，循环处理文件
mhd_path = config_test['test_data_path']  # 原始mhd文件的存放路径，用来读取世界坐标原点
pre_path = config_test['test_preprocess_result_path']  # 预处理结果存放路径


def main():
    pbb_list = [f for f in glob.glob(pbb_path + '/*pbb.npy')]  # 训练数据的全部文件
    crop_loc_list = [f for f in glob.glob(pre_path + '/*_crop.npy')]  # crop 坐标信息
    mhd_list = [f for f in glob.glob(mhd_path + '/*.mhd')]  # mhd文件

    pbb_name = [f.split('/')[-1].split('_pbb')[0] for f in pbb_list]
    crop_loc_name = [f.split('/')[-1].split('_crop')[0] for f in crop_loc_list]
    mhd_name = [f.split('/')[-1].split('.mhd')[0] for f in mhd_list]

    np_pbb = np.array([pbb_list, pbb_name]).T
    np_crop_loc = np.array([crop_loc_list, crop_loc_name]).T
    np_mhd = np.array([mhd_list, mhd_name]).T

    df_pbb = pandas.DataFrame(np_pbb, columns=['pbb_path', 'name'])
    df_crop_loc = pandas.DataFrame(np_crop_loc, columns=['crop_loc_path', 'name'])
    df_mhd = pandas.DataFrame(np_mhd, columns=['mhd_path', 'name'])

    df_merge0 = df_pbb.merge(df_mhd, on='name')
    df_merge = df_merge0.merge(df_crop_loc, on='name')

    np_merge = df_merge.as_matrix()

    whole_result = []

    for i in range(len(np_merge)):
        case_name = np_merge[i, 1]
        case_pbb_path = np_merge[i, 0]
        case_crop_path = np_merge[i, 3]
        case_mhd_path = np_merge[i, 2]

        case_pbb = np.load(case_pbb_path)  # (N,5) (P D H W Dia)
        case_crop = np.load(case_crop_path)  # (3,2) (min,max)
        itkimage = sitk.ReadImage(case_mhd_path)
        case_mhd = np.array(list(reversed(itkimage.GetOrigin())))  # 世界坐标原点

        
        case_pbb[:, 1:4] += case_crop[:, 0].reshape((1,3))
        case_pbb[:, 1:4] += case_mhd.reshape((1,3))

        #case_names = [case_name for n in range(case_pbb.shape[0])]
        case_result = np.copy(case_pbb)
        for i in range(len(case_pbb)):
            case_result[i].append(case_name)

        #case_result = np.concatenate((case_names,case_pbb), axis=1)
        whole_result = whole_result.append(case_result)
        case_result_df = pandas.DataFrame(case_result, columns=['seriesuid','probability','coordZ', 'coordY', 'coordX'])

        #case_result_df.to_csv(csv_path + case_name + '.csv', index=False)

    whole_result = np.concatenate(whole_result, axis=0)

    pandas.DataFrame(whole_result).to_csv(csv_path + 'final_result.csv', index=False, columns=['seriesuid','probability','coordZ', 'coordY', 'coordX'])

"""
    final_result = []
    third_result = []
    for pbb in pbb_list:
        second_result = []
    short_name = pbb.split('/')[-1].split('_lbb')[0]  # 最原始的结果，P,D,H,W,Dia, 体素坐标
    first_result = np.load(pbb)  # 最原始的结果，P,D,H,W,Dia, 体素坐标
    itkimage = sitk.ReadImage(mhd_path + short_name + '.mhd')
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # 世界坐标原点
    result_shape = np.shape(first_result)  # 存放结果的numpy的大小
    for i in range(result_shape[0]):
        first_result[i][1] += numpyOrigin[0]
        first_result[i][2] += numpyOrigin[1]
        first_result[i][3] += numpyOrigin[2]
    second_result = first_result.tolist()  # 世界坐标下的标注结果
    for i in range(result_shape[0]):
        second_result.append(short_name)
    third_result.append(second_result)


third_result.to_csv(csv_path)
"""
if __name__ == '__main__':
    main()








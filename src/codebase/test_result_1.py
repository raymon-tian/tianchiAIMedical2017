#coding=utf-8
import numpy as np 
import os
import glob
import SimpleITK as sitk 
import csv
import pandas as pd


csv_path = '../data/final_result/result.csv'  # 存放最终提交的csv文件的路径
pbb_path = '../data/bbox_result/test/'    # pbb文件路径，循环处理文件
mhd_path = '../data/test/'          # 原始mhd文件的存放路径，用来读取世界坐标原点
crop_path = '../data/pre_result/test/'

def main():
    pbb_list = [f for f in glob.glob(pbb_path+'/*pbb.npy')]  # 训练数据的全部文件 
    print(pbb_list)
    final_result = []
    third_result = []
    for pbb in pbb_list:
        second_result = []
        full_name = pbb.split('/')[-1]
        short_name = full_name.split('_pbb')[0]
        print(short_name)
        first_result = np.load(pbb)    # 最原始的结果，P,D,H,W,Dia, 体素坐标
        sitk_path = mhd_path + short_name + '.mhd'
        itkimage = sitk.ReadImage(sitk_path)
        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin()))) # 世界坐标原点
        case_crop_path = crop_path + short_name + '_crop.npy'
        case_crop = np.load(case_crop_path)
        result_shape = np.shape(first_result)  # 存放结果的numpy的大小
        for i in range(result_shape[0]):
        	first_result[i][1] = first_result[i][1]+case_crop[0][0]+numpyOrigin[0]
        	first_result[i][2] = first_result[i][2]+case_crop[1][0]+numpyOrigin[1]
        	first_result[i][3] = first_result[i][3]+case_crop[2][0]+numpyOrigin[2]
        second_result = first_result.tolist()  # 世界坐标下的标注结果
        for i in range(result_shape[0]):
        	second_result[i].append(short_name)
        
        columns = ['P','D','H','W','Dia','Name']
        third_result = pd.DataFrame(columns = columns, data = second_result)
        cvs_path = '../data/final_result/' + short_name + '.csv'
        third_result.to_csv(cvs_path)
if __name__ == '__main__':
	main()
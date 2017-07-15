#coding=utf-8

# test阶段使用配置文件
config = {
    
    'test_data_path':'../data/test/',  # 原始测试数据集存放路径
    
    'test_preprocess_result_path':'../data/pre_result/test/',  # 预处理结果存放路径
   
    'outputfile':'../data/final_result/',   # 最终预测结果CSV文件

    'detector_model':'net_res18',   # 肺结节检测网络,会动态导入

    'n_gpu':1,

    'test_result_path':'../data/bbox_result/test/',
    
    'n_worker_preprocessing':None, # None表示开启机器所支持的最大线程数

    'use_exsiting_preprocessing':True, # 是否使用预处理结果，判断存在预处理结果文件时，直接使用

}

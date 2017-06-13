#coding=utf-8

# test阶段使用配置文件
config = {
    # 原始测试数据集存放路径
    'datapath':'/home/g1002/datasets/test/',
    # 预处理结果存放路径
    'preprocess_result_path':'/home/g1002/dongwang/tianchiAIMedical2017/codes/prep_result/',
    # 输出文件名
    'outputfile':'prediction.csv',
    # 肺结节检测模型（这个应该是个Python脚本）
    'detector_model':'net_detector.py',
    # ckpt好像是tensorflow的模型文件后缀;训练好的模型的参数
    'detector_param':'./model/detector.ckpt',

    'classifier_model':'net_classifier.py',

    'classifier_param':'./model/classifier.ckpt',

    'n_gpu':1,
    # None表示开启机器所支持的最大线程数
    'n_worker_preprocessing':None,

    'use_exsiting_preprocessing':True,
    # 是否跳过预处理
    'skip_preprocessing':True,

    'skip_detect':False
}

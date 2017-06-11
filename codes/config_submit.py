#coding=utf-8

# 测试用配置文件
config = {
    # stage2 data是用来预测的？
    'datapath':'/work/DataBowl3/stage2/stage2/',
    # 路径
    'preprocess_result_path':'./prep_result/',
    # 输出文件名
    'outputfile':'prediction.csv',
    # 这个应该是个Python脚本
    'detector_model':'net_detector.py',
    # ckpt好像是tensorflow的模型文件后缀;训练好的模型的参数
    'detector_param':'./model/detector.ckpt',

    'classifier_model':'net_classifier.py',

    'classifier_param':'./model/classifier.ckpt',

    'n_gpu':1,

    'n_worker_preprocessing':None,

    'use_exsiting_preprocessing':False,
    # 是否跳过预处理
    'skip_preprocessing':False,

    'skip_detect':False
}

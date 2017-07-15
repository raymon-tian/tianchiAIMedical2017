#coding=utf-8
config = {

    # ====== tianchi 的数据    ======
    'train_data_path':'../data/train/',
    'train_annos_path':
        [
            '../data/train/annotations.csv',
            '../data/train/annotations_excluded.csv',
            '../data/train/seriesuids.csv'
        ],
    'train_preprocess_result_path': '../data/pre_result/train/',
    'val_data_path':'../data/val/',
    'val_annos_path':
        [
            '../data/val/annotations.csv',
            '../data/val/annotations_excluded.csv',
            '../data/val/seriesuids.csv'
        ],
    'val_preprocess_result_path':'../data/pre_result/val/',
    'weights_path':'../data/weights/',
    'val_result_path':'../bbox_result/val/', 
    'preprocessing_backend':'python',
}

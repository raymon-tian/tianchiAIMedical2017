#coding=utf-8
config = {
    # ===== DSB 的数据 =====
    'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
    'stage1_annos_path':
        [
            './detector/labels/label_job5.csv',
            './detector/labels/label_job4_2.csv',
            './detector/labels/label_job4_1.csv',
            './detector/labels/label_job0.csv',
            './detector/labels/label_qualified.csv'
        ],
    # ====== tianchi 的数据    ======
    'tc_data_path':'/home/raymon/Desktop/winC/Users/raymon/Desktop/dongwang/train_subset00',
    'tc_annos_path':
        [
            '/home/raymon/PycharmProjects/tianchiAIMedical2017/datasets/csv/train/annotations.csv',
            '/home/raymon/PycharmProjects/tianchiAIMedical2017/datasets/csv/train/annotations_excluded.csv',
            '/home/raymon/PycharmProjects/tianchiAIMedical2017/datasets/csv/train/seriesuids.csv'
        ],
    # ======  Luna2016 的数据  =====
    # 原始的Luna数据，包含几个子文件夹
    'luna_raw': '/work/DataBowl3/luna/raw/',
    # 所以，好像lung给出了mask
    'luna_segment': '/work/DataBowl3/luna/seg-lungs-LUNA16/',
    # 这个好像是将luna的所有数据合并到一起
    'luna_data': '/work/DataBowl3/luna/allset',
    'luna_abbr': './detector/labels/shorter.csv',
    'luna_label': './detector/labels/lunaqualified.csv',
    # 看样子，DSB和Luna的数据处理结果是要放在一起
    'preprocess_result_path': '/home/raymon/tianchiTemp/DataBowl3/stage1/preprocess/',
    'bbox_path':'../detector/results/res18/bbox/',
    'preprocessing_backend':'python'
}

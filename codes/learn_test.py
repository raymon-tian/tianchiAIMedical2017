import multiprocessing
import os
import pandas as pd
import  numpy as np
from tqdm import tqdm

def do_calculation(data):
    return data*2
def start_process():
    print 'Starting',multiprocessing.current_process().name

def ttest(*args):
    print(args)

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile

def get_config():
    config = dict()
    config['path_n'] = '/home/raymon/Desktop/s3/NIR_128x128'
    config['path_v'] = '/home/raymon/Desktop/s3/VIS_128x128'
    return config

if __name__ == '__main__':

    config = get_config()
    f_list_n = dirlist(config['path_n'],[])
    f_list_v = dirlist(config['path_v'],[])

    f_n = np.array(f_list_n).reshape((-1,1))
    f_v = np.array(f_list_v).reshape((-1,1))

    f_n = [f_n,f_n,f_n]
    f_n = np.concatenate(f_n,axis=1)
    f_v = [f_v,f_v,f_v]
    f_v = np.concatenate(f_v,axis=1)


    for i in tqdm(range(len(f_n))):
        split = f_n[i,0].split('/')
        f_n[i,1] = split[-2]
        f_n[i,2] = split[-2]+'/'+split[-1]

    for i in tqdm(range(len(f_v))):
        split = f_v[i,0].split('/')
        f_v[i,1] = split[-2]
        f_v[i,2] = split[-2]+'/'+split[-1]


    # print(f_n[:,2][:5])
    # print(f_v[:,2][:5])

    column_names = ['path','person','person_pic']
    f_n = pd.DataFrame(f_n,columns=column_names)
    f_v = pd.DataFrame(f_v,columns=column_names)
    path_n = f_n['path'].as_matrix().tolist()
    path_v = f_v['path'].as_matrix().tolist()
    # print(f_n.head(20))
    # print(f_v.head(20))

    merged = f_n.merge(right=f_v,how='inner',on=column_names[2],suffixes=('_n','_v'),copy=False)
    # print(merged.head(20))
    leg_path_n = merged['path_n'].as_matrix().tolist()
    leg_path_v = merged['path_v'].as_matrix().tolist()
    # print(leg_path_n[:5])

    for i in tqdm(range(len(path_n))):
        case = path_n[i]
        if case not in leg_path_n:
            os.system('rm %s'%case)
    for i in tqdm(range(len(path_v))):
        case = path_v[i]
        if case not in leg_path_v:
            os.system('rm %s'%case)
"""
if __name__=='__main__':
    ttest(1,2,3)
    inputs=list(range(10))
    print 'Inputs  :',inputs

    builtin_output=map(do_calculation,inputs)
    print 'Build-In :', builtin_output

    pool_size=multiprocessing.cpu_count()*2
    pool_size = None
    pool=multiprocessing.Pool(processes=pool_size,
        initializer=start_process,)

    pool_outputs=pool.map(do_calculation,inputs)
    pool.close()
    pool.join()

    print 'Pool  :',pool_outputs

"""


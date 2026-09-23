import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
import glob
import cv2
import sys
import shutil
sys.path.append('/home/project')

from utils.helper import splitFileName, makedirs, saveImage, cropMid, getNameList
# from utils.log_custom import logI

if __name__ == "__main__":
    categories = ['train', 'val', 'test']
    path_deafult = '/home/project/data'
    path0 =  '{}/1024_split'.format(path_deafult)
    num_ch = 19
    img_size = 128
    path1 =  '{}/origin/data_full_ch{}_size{}_album'.format(path_deafult, num_ch, img_size)
    dirs = sorted(glob.glob('{}/*'.format(path1)))
    # path1_ps =  '{}/origin/20230201_bc_size256_ch19_split/{}'.format(path_deafult, dirs[0])
    # path1_ps =  '{}/origin/20230201_bc_size256_ch19_split/{}'.format(path_deafult, dirs[0])
    path1_ps =  '{}'.format(dirs[0])
    path1_sd =  '{}'.format(dirs[1])
    # path1_sd =  '{}/origin/20230201_bc_size256_ch19_split/{}'.format(path_deafult, dirs[1])
    albs = ['0', '90', '180', '270', 'h', 'v']
    
    # list_ps_files = getNameList(dirs[0], type='*.tif', path_default=False)
    # list_sd_files = getNameList(dirs[1], type='*.tif', path_default=False)

    for category in categories:
        types = glob.glob('{}/{}/*'.format(path0, category))
        for type in types:
            name_dise = os.path.basename(type)
            files = glob.glob('{}/*'.format(type))
            path_new = '{}/ch{}_size{}_album_bc/{}/{}'.format(path_deafult, num_ch, img_size, category, name_dise)
            makedirs(path_new)
            # list_files = getNameList(type, type='*.png', path_default=False)
            for file in files:
                pat, name, ext = splitFileName(file)
                if name_dise == 'ps':
                    # print(path1_ps)
                    # shutil.copy2(src, dst)
                    for spli in range(16):
                        for alb in albs:
                            if num_ch == 3:
                                ext_ = 'png'
                                filename1 = '{}/{}_00_{}_{}.{}'.format(path1_ps, name, spli, alb, ext_)
                                filename2 = '{}/{}_00_{}_{}.{}'.format(path_new, name, spli, alb, ext_)
                            else :
                                ext_ = 'tif'
                                filename1 = '{}/{}_{}_{}.{}'.format(path1_ps, name, spli, alb, ext_)
                                filename2 = '{}/{}_{}_{}.{}'.format(path_new, name, spli, alb, ext_)
                            shutil.copy2(filename1, filename2)    
                    # shutil.copy2('{}/{}_0.tif'.format(path1_ps, name), '{}/{}_0.tif'.format(path_new, name))
                    
                elif name_dise == 'sd':
                    for spli in range(16):
                        for alb in albs:
                            if num_ch == 3:
                                ext_ = 'png'
                                filename1 = '{}/{}_00_{}_{}.{}'.format(path1_sd, name, spli, alb, ext_)
                                filename2 = '{}/{}_00_{}_{}.{}'.format(path_new, name, spli, alb, ext_)
                            else :
                                ext_ = 'tif'
                                filename1 = '{}/{}_{}_{}.{}'.format(path1_sd, name, spli, alb, ext_)
                                filename2 = '{}/{}_{}_{}.{}'.format(path_new, name, spli, alb, ext_)
                            shutil.copy2(filename1, filename2)    
                    

            # print(list_files)
            # for file in files:
                

                # makedirs(path_)
                # make dir
                # match file & copy

    # for dir in dirs:
    #     print()
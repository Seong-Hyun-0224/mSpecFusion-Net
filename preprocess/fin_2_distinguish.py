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
    # corrections = ['ac', 'bc']
    corrections = ['ac']
    for cor in corrections:
        categories = ['train', 'val', 'test']
        path_deafult = '/home/project/data'
        path0 =  '{}/1024_split'.format(path_deafult)
        # num_ch = 19
        img_size = 128
        # path1 =  '{}/origin/20230525_{}_size{}_all_album'.format(path_deafult, cor, img_size)
        path1 =  '{}/origin/20230713_{}_size{}_all__album'.format(path_deafult, cor, img_size)
        dirs = sorted(glob.glob('{}/*'.format(path1)))
        
        path1_ps =  '{}'.format(dirs[0])
        path1_sd =  '{}'.format(dirs[1])

        albs = ['0', '90', '180', '270', 'h', 'v']

        for category in categories:
            types = glob.glob('{}/{}/*'.format(path0, category))
            for type in types:
                name_dise = os.path.basename(type)
                files = glob.glob('{}/*'.format(type))
                # path_new = '{}/origin/20230525_all_size{}_album_{}/{}/{}'.format(path_deafult, img_size, cor, category, name_dise)
                path_new = '{}/origin/20230713_all__size{}_album_{}/{}/{}'.format(path_deafult, img_size, cor, category, name_dise)
                # makedirs(path_new)
                
                for file in files:
                    pat, name, ext = splitFileName(file)
                    if name_dise == 'ps':
                        for spli in range(16):
                            for alb in albs:
                                # print('{}/{}_{:02d}_{}'.format(path_new, name, spli, alb))
                                path_out = '{}/{}_{:02d}_{}'.format(path_new, name, spli, alb)
                                makedirs(path_out)
                                ext_ = 'png'
                                dirname = '{}/{}_{:02d}_{}'.format(path1_ps, name, spli, alb)
                                chs = glob.glob('{}/*.png'.format(dirname))
                                for ch in chs:
                                    patt, name_ch, ext = splitFileName(ch)
                                    filename_dst = '{}/{}.{}'.format(path_out, name_ch, ext)
                                    shutil.copy2(ch, filename_dst)
                    if name_dise == 'sd':
                        for spli in range(16):
                            for alb in albs:
                                # print('{}/{}_{:02d}_{}'.format(path_new, name, spli, alb))
                                path_out = '{}/{}_{:02d}_{}'.format(path_new, name, spli, alb)
                                makedirs(path_out)
                                ext_ = 'png'
                                dirname = '{}/{}_{:02d}_{}'.format(path1_sd, name, spli, alb)
                                chs = glob.glob('{}/*.png'.format(dirname))
                                for ch in chs:
                                    patt, name_ch, ext = splitFileName(ch)
                                    filename_dst = '{}/{}.{}'.format(path_out, name_ch, ext)
                                    shutil.copy2(ch, filename_dst)
import os
import glob
import cv2
import albumentations as A
import numpy as np

import sys
sys.path.append('/home/project')

from utils.helper import splitFileName, makedirs, saveImage, readImage
from utils.log_custom import logI

if __name__ == "__main__":
    path =  '/home/project/data/origin/20230713_ac_size128_all_'
    path_dir = '{}'.format(path)
    path_dir_out = '{}_album'.format(path)
    makedirs(path_dir_out)
    flag_debug = False
    imgtype_ = 'png'

    types = glob.glob('{}/*'.format(path))
    for type in types:
        name_type = os.path.basename(type)
        ids = glob.glob('{}/*'.format(type))
            
        for id in ids:
            chs = glob.glob('{}/*.png'.format(id))
            for ch in chs:
                path, name_file, ext = splitFileName(ch)
                
                img_ = readImage(ch)
                dirname = os.path.basename(path)

                ## album
                aug = A.Flip(p=1)
                ds = [0, 1]
                name_flips = ['v', 'h']
                for d_ in ds:
                    transformed_img = aug.apply(img=img_, d=d_)
                    saveImage('{}/{}/{}_{}'.format(path_dir_out, name_type, dirname, name_flips[d_]),
                                '{}'.format(name_file), transformed_img, flag_debug=flag_debug)
                
                aug = A.RandomRotate90(p=1)
                name_flips = ['0', '90', '180', '270']
                for times in range(4):
                    transformed_img = aug.apply(img=img_, factor=times)
                    saveImage('{}/{}/{}_{}'.format(path_dir_out, name_type, dirname, name_flips[times]),
                                '{}'.format(name_file), transformed_img, flag_debug=flag_debug)
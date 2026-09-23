import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
import glob
import cv2
import sys
sys.path.append('/home/project')

from utils.helper import splitFileName, makedirs, saveImage, cropMid
from utils.log_custom import logI

if __name__ == "__main__":
    path =  '/home/project/data/origin/data_full'
    path_dir = '{}'.format(path)
    path_dir_out = '{}_'.format(path)
    makedirs(path_dir_out)
    num_split_in_1_direction = 2
    flag_debug = True

    categories = glob.glob('{}/*'.format(path_dir))
    # for category in categories[:1]:
    for category in categories:
        name_category = os.path.basename(category)
        types = glob.glob('{}/*'.format(category))
        # for type in types[:1]:
        for type in types:
            name_type = os.path.basename(type)
            files = glob.glob('{}/*'.format(type))
            # sorted(files)
            for file in sorted(files)[:1]:
            # for file in files:
                path, name_file, ext = splitFileName(file)
                img_ = cv2.imread(file)
                
                if len(img_.shape) == 3:
                    h, w, z = img_.shape
                else :
                    h, w = img_.shape

                # unit_img = int(h/num_split_in_1_direction/2)
                # unit_img = int(h/num_split_in_1_direction/2)
                unit_img = 256

                img_crop = cropMid(img_, 1024)
                # print(img_crop.shape)

                # saveImage('{}/{}'.format(path_dir_out, name_category),
                #     '{}'.format(name_type), img_crop, flag_debug=flag_debug)

                for i in range(num_split_in_1_direction):
                    for j in range(num_split_in_1_direction):
                        # img_part = img_[i*unit_img:(i+1)*unit_img, j*unit_img:(j+1)*unit_img, :]
                        img_part = img_crop[512 - unit_img + i*unit_img:512 - unit_img + (i+1)*unit_img, 512 - unit_img +  j*unit_img:512 - unit_img + (j+1)*unit_img, :]
                        # saveImage('{}/{}/{}'.format(path_dir_out, name_category, name_type),
                        #     '{}_{}'.format(name_file, i*num_split_in_1_direction + j), img_part, flag_debug=flag_debug)
                        saveImage('{}/{}'.format(path_dir_out, name_category),
                            '{}_{}_{}'.format(name_type, name_file, i*num_split_in_1_direction + j), img_part, flag_debug=flag_debug)
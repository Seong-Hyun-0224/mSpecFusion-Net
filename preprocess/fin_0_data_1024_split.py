import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage import io
import os
import cv2

import sys
sys.path.append('/home/project')
from utils.helper import splitFileName, saveImage, cropMid

# load img
path =  '/home/project/data/origin/final_20230525_ac'
# path =  '/home/project/data/origin/up2date_crop'
path_dir = '{}'.format(path)

dirs = glob.glob('{}/*'.format(path))
dirs = sorted(dirs)
# num_channels = 19
num_channels = 3
sizes = [512]
num_axis = 4
flag_split = True

for size in sizes:
    if flag_split:
        unit_img = int(size/num_axis)
        str_split = '_split'
    else :
        unit_img = size
        str_split = ''
        
    path_out = '{}/20230713_ac_size{}_all_'.format(os.path.dirname(path), unit_img)
    for dir in dirs:
        ids = glob.glob('{}/*'.format(dir))
        for id in ids:
            cate = os.path.basename(os.path.dirname(id))
            name = os.path.basename(id)
            path_out_dir = '{}/{}/{}'.format(path_out, cate, name)
        
            channels = sorted(glob.glob('{}/*'.format(id)))
            
            for num, ch in enumerate(channels):
                img_ = cv2.imread(ch)
                img_ = cropMid(img_, size)
                
                if flag_split:
                    for i in range(num_axis):
                        for j in range(num_axis):
                            img_part = img_[i*unit_img:(i+1)*unit_img, j*unit_img:(j+1)*unit_img, :]
                            saveImage('{}_{:02d}'.format(path_out_dir, i*num_axis + j),
                                    '{}_ch{:02d}'.format(name, num), img_part.astype(dtype=np.uint8))
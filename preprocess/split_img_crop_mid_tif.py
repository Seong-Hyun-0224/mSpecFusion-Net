
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage import io
import os

import sys
sys.path.append('/home/project')
from utils.helper import splitFileName, saveImage, cropMid

# load img
path =  '/home/project/data/origin/data_full'
path_dir = '{}'.format(path)

dirs = glob.glob('{}/*'.format(path))
dirs = sorted(dirs)
num_channels = 19
# sizes = [1024, 512]
# sizes = [1024, 512]
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
        
    path_out = '{}/20230418_bc_size{}_ch{}{}'.format(os.path.dirname(path), unit_img, num_channels, str_split)
    for dir in dirs:
    # for dir in dirs[:1]:
        ids = glob.glob('{}/*'.format(dir))
        for id in ids:
        # for id in ids[:1]:
            cate = os.path.basename(os.path.dirname(id))
            name = os.path.basename(id)
            path_out_dir = '{}/{}'.format(path_out, cate)
        
            channels = sorted(glob.glob('{}/*'.format(id)))
            img_tif = np.zeros((num_channels, size, size), dtype=np.float64)
            
            for num, ch in enumerate(channels):
            # for num, ch in enumerate(channels[:1]):
                # img_ = cv2.imread(ch)
                img_ = io.imread(ch)

                img_ch2 = np.mean(img_, axis=2)

                img_ch2 = cropMid(img_ch2, size)

                img_tif[num,:,:] = img_ch2
                
            if flag_split:
                for i in range(num_axis):
                    for j in range(num_axis):
                        img_part = img_tif[:, i*unit_img:(i+1)*unit_img, j*unit_img:(j+1)*unit_img]
                        saveImage(path_out_dir,
                                '{}_{}'.format(name, i*num_axis + j), img_part.astype(dtype=np.uint8), imgtype='tif')
                        
            else :
                saveImage(path_out_dir, name, img_tif.astype(dtype=np.uint8), imgtype='tif')
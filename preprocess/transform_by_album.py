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
    path =  '/home/project/data/origin/data_full_ch19_128'
    path_dir = '{}'.format(path)
    path_dir_out = '{}_album'.format(path)
    makedirs(path_dir_out)
    flag_debug = False
    # imgtype_ = 'tif'
    imgtype_ = 'png'

    types = glob.glob('{}/*'.format(path))
    # for type in types[:1]:
    for type in types:
        name_type = os.path.basename(type)
        files = glob.glob('{}/*'.format(type))
        # for file in sorted(files)[:1]:
        for file in files:
            path, name_file, ext = splitFileName(file)
            # img_ = cv2.imread(file)
            img_ = readImage(file, imgtype=imgtype_)
            if imgtype_ == 'tif':
                img_ = np.moveaxis(img_, 0, 2)
            # print(img_.shape)

            ## album
            aug = A.Flip(p=1)
            ds = [0, 1]
            name_flips = ['v', 'h']
            for d_ in ds:
                transformed_img = aug.apply(img=img_, d=d_)
                if imgtype_ == 'tif':
                    transformed_img = np.moveaxis(transformed_img, 2, 0)
                saveImage('{}/{}'.format(path_dir_out, name_type),
                            '{}_{}'.format(name_file, name_flips[d_]), transformed_img, imgtype='tif', flag_debug=flag_debug)
                
            aug = A.RandomRotate90(p=1)
            name_flips = ['0', '90', '180', '270']
            for times in range(4):
                transformed_img = aug.apply(img=img_, factor=times)
                if imgtype_ == 'tif':
                    transformed_img = np.moveaxis(transformed_img, 2, 0)
                saveImage('{}/{}'.format(path_dir_out, name_type),
                            '{}_{}'.format(name_file, name_flips[times]), transformed_img, imgtype='tif', flag_debug=flag_debug)

# # from albumentations import (ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
# #                             RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded 
# #                            )
# image = 1

# aug = A.Flip(p=1)
# transformed_img = aug.apply(img=image, d=1)
# # 0 for vertical, 1 for horizontal, -1 for both

# aug = A.RandomRotate90(p=1)
# transformed_img = aug.apply(img=image, factor=1)
# # 1 for 90, 2 for 180, 3 for 270, factor means times! not angle :(

# transformed_image = transformed_img['image']
import numpy as np
import glob
import os
from skimage import io
import sys
sys.path.append('/home/project')

from utils.helper import saveImage

def converttif(path_dir):
    if 'album_ac' in path_dir:
        calib = 4
        pcalib = 5
    else :
        calib = 1.8
        pcalib = 3
    categories = ['data/{}/train'.format(path_dir), 'data/{}/val'.format(path_dir), 'data/{}/test'.format(path_dir)]
    path_out_dir = 'data/{}_'.format(path_dir)
    for category in categories[:1]:
        classes = glob.glob('{}/*'.format(category))
        name_cate = os.path.basename(category)
        for cl in classes[:1]:
            name_cl = os.path.basename(cl)
            ids = glob.glob('{}/*'.format(cl))
            for id in ids[:1]:
                name_id = os.path.basename(id)
                print(name_id)
                chs = glob.glob('{}/*'.format(id))
                chs = sorted(chs)
                set_temp = np.zeros((128, 128, 23), dtype=np.float64)
                for idx, ch in enumerate(chs):
                    # img = cv2.imread(ch)
                    img_ = io.imread(ch)
                    if idx == 0:
                        set_temp[:,:,0] = img_[:,:,0]
                        set_temp[:,:,1] = img_[:,:,1]
                        set_temp[:,:,2] = img_[:,:,2]
                    elif idx > 0 and idx < 10:
                        set_temp[:,:,idx + 2] = np.mean(img_, axis=2) * calib
                    elif idx == 10:
                        set_temp[:,:,12] = img_[:,:,0]
                        set_temp[:,:,13] = img_[:,:,1]
                        set_temp[:,:,14] = img_[:,:,2]
                    elif idx > 10:
                        set_temp[:,:,idx + 4] = np.mean(img_, axis=2) * pcalib
                # set_temp.shape (128, 128, 23)
                set_temp_ = np.moveaxis(set_temp, 2, 0)
                # set_temp.shape (23, 128, 128)
                
                saveImage('{}____/{}/{}'.format(path_out_dir,name_cate,name_cl),
                        '{}'.format(name_id), set_temp_.astype(dtype=np.uint8), imgtype='tif')
                            
corrections = ['ac', 'bc']
for cor in corrections:
    converttif('origin/all_size128_album_{}'.format(cor))
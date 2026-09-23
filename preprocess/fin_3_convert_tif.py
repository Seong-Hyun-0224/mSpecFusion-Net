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
        uvcalib = 1
    else :
        calib = 1.8
        pcalib = 3
        uvcalib = 1
    categories = ['data/{}/train'.format(path_dir), 'data/{}/val'.format(path_dir), 'data/{}/test'.format(path_dir)]
    path_out_dir = 'data/{}_'.format(path_dir)
    for category in categories:
        classes = glob.glob('{}/*'.format(category))
        name_cate = os.path.basename(category)
        for cl in classes:
            name_cl = os.path.basename(cl)
            ids = glob.glob('{}/*'.format(cl))
            for id in ids:
                name_id = os.path.basename(id)
                print(name_id)
                chs = glob.glob('{}/*'.format(id))
                chs = sorted(chs)
                set_temp = np.zeros((128, 128, 23), dtype=np.float64)
                for idx, ch in enumerate(chs):
                    # img = cv2.imread(ch)
                    img_ = io.imread(ch)
                    # fac_chs = [1, 1.875001172,1.507537688, 3.618158686, 4.048927006, 2.419354839, 9.847245868, 2.112676056, 2.052831156, 1, 1, 2.04620371, 1.614978787, 5.088322542, 4.805588286, 2.639070553, 11.53846154, 2.112676056, 3.142412755]
                    if idx == 0:
                        set_temp[:,:,0] = img_[:,:,0]
                        set_temp[:,:,1] = img_[:,:,1]
                        set_temp[:,:,2] = img_[:,:,2]
                    elif idx > 0 and idx < 9:
                        set_temp[:,:,idx + 2] = np.mean(img_, axis=2) * calib
                    elif idx == 9:
                        set_temp[:,:,idx + 2] = np.mean(img_, axis=2) * uvcalib
                    elif idx == 10:
                        set_temp[:,:,12] = img_[:,:,0]
                        set_temp[:,:,13] = img_[:,:,1]
                        set_temp[:,:,14] = img_[:,:,2]
                    elif idx > 10:
                        set_temp[:,:,idx + 4] = np.mean(img_, axis=2) * pcalib
                # set_temp.shape (128, 128, 23)
                set_temp_ = np.moveaxis(set_temp, 2, 0)
                # set_temp.shape (23, 128, 128)
                
                saveImage('{}/{}/{}'.format(path_out_dir,name_cate,name_cl),
                        '{}'.format(name_id), set_temp_.astype(dtype=np.uint8), imgtype='tif')
                            
# corrections = ['ac', 'bc']
corrections = ['ac']
for cor in corrections:
    converttif('origin/20230525_all_size128_album_{}'.format(cor))
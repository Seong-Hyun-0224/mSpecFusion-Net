import numpy as np
import glob
import cv2
import os
from keras.utils import np_utils
from skimage import io

def load(path_dir, dtype = 'tif'):
    # path_dir = 'data'
    if dtype != 'tif':
        path_default = path_dir
        file_X = '{}/custom_1024_X.npy'.format(path_dir)
        file_y = '{}/custom_1024_y.npy'.format(path_dir)
        if not(os.path.exists(file_X) and os.path.exists(file_y)):
        # if True:
            # categories = glob.glob('{}/*'.format(path_default))
            categories = ['data/train', 'data/val', 'data/test']
            # train, test, val
            Xs = []
            ys = []
            label = []
            for category in categories:
                list_X = []
                list_y = []
                classes = glob.glob('{}/*'.format(category))
                for cl in classes:
                    files = glob.glob('{}/*'.format(cl))
                    if 'ps' in cl:
                        label = [0]
                    elif 'sd' in cl:
                        label = [1]
                    elif 'no' in cl:
                        label = [2]
                    for file in files:
                        img = cv2.imread(file)
                        list_X.append(img)
                        list_y.append(label)
                Xs.append(list_X)
                ys.append(list_y)

            X_train = np.array(Xs[0])
            y_train = np.array(ys[0])

            X_test = np.array(Xs[1])
            y_test = np.array(ys[1])

            X_val = np.array(Xs[2])
            y_val = np.array(ys[2])

            X_train = X_train.astype('float32')
            X_val = X_val.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train/255.
            X_val = X_val/255.
            X_test = X_test/255.

            y_train = np_utils.to_categorical(y_train)
            y_val = np_utils.to_categorical(y_val)
            y_test = np_utils.to_categorical(y_test)

            np.save(file_X, [X_train, X_val, X_test])
            np.save(file_y, [y_train, y_val, y_test])
        else :
            X_train, X_val, X_test = np.load(file_X, allow_pickle=True)
            y_train, y_val, y_test = np.load(file_y, allow_pickle=True)

        return X_train, X_val, X_test, y_train, y_val, y_test
    else :
        # path =  '/Users/dion/workspace/trial/data/dgist/origin'
        path =  '/home/project/data/ch19_size256_bc'
        path_dir = '{}'.format(path)

        categories = ['train', 'val', 'test']

        Xs = []
        ys = []
        label = []
        for cate in categories:
            list_X = []
            list_y = []
            classes = glob.glob('{}/{}/*'.format(path, cate))
            for cl in classes:
                ids = glob.glob('{}/*'.format(cl))
                for id in ids:
                    if 'ps' in id:
                        label = [0]
                    else :
                        label = [1]
                        
                    img_ = io.imread(id)
                    # (19, 1024, 1024)
                    x = np.moveaxis(img_, 0, 2)
                    # (1024, 1024, 19)
                    list_X.append(x)
                    list_y.append(label)
            Xs.append(list_X)
            ys.append(list_y)
            
        X_train = np.array(Xs[0])
        y_train = np.array(ys[0])

        X_test = np.array(Xs[1])
        y_test = np.array(ys[1])

        X_val = np.array(Xs[2])
        y_val = np.array(ys[2])

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train/255.
        X_val = X_val/255.
        X_test = X_test/255.

        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_val)
        y_test = np_utils.to_categorical(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

def read_image(path,img_size=1024,num_ch=19):
    im = np.zeros((np.array(path).size,img_size,img_size,num_ch))
    for i in range(np.array(path).size):
        x = io.imread(path[i])
        # x = x / 255.0
        if x.shape[2] != 3 and x.shape[2] != 4:
            x = np.moveaxis(x, 0, 2)
        im[i,:,:,:] = x
    im = np.float32(im)
    return im

if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = load('', 'tif')
    
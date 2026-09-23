import numpy as np
import glob
import cv2
import os
from keras.utils import np_utils

def load(path_dir, size=1024):
    path_default = path_dir
    file_X = 'data/custom_{}_X.npy'.format(size)
    file_y = 'data/custom_{}_y.npy'.format(size)
    if not(os.path.exists(file_X) and os.path.exists(file_y)):
    # if True:
        # categories = glob.glob('{}/*'.format(path_default))
        categories = ['data/1024_split/train', 'data/1024_split/val', 'data/1024_split/test']
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
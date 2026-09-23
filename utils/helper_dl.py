from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp
from keras.utils import Sequence
import math
import pandas as pd
from skimage import io

def split_data(seed=42):
    # folds = range(1, 11)
    # folds = np.random.RandomState(seed).permutation(folds)
    folds_ = [np.array([2, 3, 9, 4, 8, 7, 10, 5, 6, 1]), np.array([2, 6, 5, 7, 8, 3, 1, 4, 9, 10]), np.array([10, 6, 2, 8, 5, 9, 1, 3, 7, 4]), \
             np.array([9, 4, 6, 10, 2, 7, 3, 1, 8, 5]), np.array([5, 6, 10, 8, 7, 4, 1, 9, 3, 2])]
    folds = folds_[seed]
    return folds[:7], folds[7:8], folds[8:]


class DatastLoader(Sequence):
    @classmethod
    def split(cls, data_dir, path_to_csv, train_folds, valid_folds, test_folds, batch_size=8, name_type_dataset='rgb'):
        train_seq = cls(data_dir, path_to_csv, batch_size, folds=train_folds, type_set=name_type_dataset)
        valid_seq = cls(data_dir, path_to_csv, batch_size, folds=valid_folds, type_set=name_type_dataset)
        test_seq = cls(data_dir, path_to_csv, batch_size, folds=test_folds, type_set=name_type_dataset)
        return train_seq, valid_seq, test_seq

    def __init__(self, data_dir, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None, folds=None, type_set='rgb', size=128):
        if folds is None:
            folds = [1, 3]

        self.__data_dir__ = data_dir
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.__type_set__ = type_set
        self.__img_size__ = size
        num_ch, chs = self.__getChannels__()
        self.__num_ch__ = num_ch
        self.__chs__ = chs

        # CSV 파일을 로드하여 필요한 파일 리스트 및 레이블 정보 저장
        if path_to_csv is None:
            self.list_files = None
            self.y = None
        else:
            data = pd.read_csv(path_to_csv)
            data = data[data['fold'].isin(folds)]
            self.list_files = data['patient_id'].values
            data.drop('fold', inplace=True, axis=1) 
            data.drop('patient_id', inplace=True, axis=1)
            self.y = data.values  # 데이터프레임을 넘파이 배열로 변환

        if end_idx is None:
            self.end_idx = len(self.list_files)
        else:
            self.end_idx = end_idx
            
    

    @property
    def n_classes(self):
        return 0 if self.y is None else self.y.shape[1]

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
    
    def __getChs__(self):
        return self.__num_ch__

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        batch_files = self.list_files[start:end]
        
        batch_data = []
        for file in batch_files:
            set_temp = np.zeros((self.__img_size__, self.__img_size__, self.__num_ch__), dtype=np.uint8)
            img_ = io.imread(f'{self.__data_dir__}/{file}.tif')
            x = np.moveaxis(img_, 0, 2)
            # (128, 128, 23)
            for idx in range(self.__num_ch__):
                set_temp[:,:,idx] = x[:,:,self.__chs__[idx]]
            batch_data.append(set_temp)
        
        batch_data = np.array(batch_data)
        
        if self.y is None:
            return batch_data
        else:
            batch_labels = self.y[start:end]
            return batch_data, batch_labels
        
    def __getChannels__(self):
        '''
        types
            ['rgb', 'prgb', 'sp', 'psp', rgb+uv, prgb+uv, rgb+prgb+uv, sp+psp, sp+psp+uv, sp+uv, psp+uv]
        '''
        if self.__type_set__ == 'rgb':
            num_ch = 3
            chs = [0,1,2]
        elif self.__type_set__ == 'prgb':
            num_ch = 3
            chs = [12,13,14]
        elif self.__type_set__ == 'sp':
            num_ch = 8
            chs = [3,4,5,6,7,8,9,10]
        elif self.__type_set__ == 'psp':
            num_ch = 8
            chs = [15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'rgb+uv':
            num_ch = 4
            chs = [0,1,2,11]
        elif self.__type_set__ == 'prgb+uv':
            num_ch = 4
            chs = [11, 12, 13, 14]
        elif self.__type_set__ == 'sp+uv':
            num_ch = 9
            chs = [3,4,5,6,7,8,9,10, 11]
        elif self.__type_set__ == 'psp+uv':
            num_ch = 9
            chs = [11, 15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'rgb+prgb':
            num_ch = 6
            chs = [0,1,2, 12, 13, 14]
        elif self.__type_set__ == 'rgb+prgb+uv':
            num_ch = 7
            chs = [0,1,2,11, 12, 13, 14]
        elif self.__type_set__ == 'sp+psp':
            num_ch = 16
            chs = [3,4,5,6,7,8,9,10,15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'sp+psp+uv':
            num_ch = 17
            chs = [3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22]
        else :
            num_ch = -1
            return -1, -1, -1, -1, -1, -1, num_ch
        return num_ch, chs


class Datast3DLoader(Sequence):
    @classmethod
    def split(cls, data_dir, path_to_csv, train_folds, valid_folds, test_folds, batch_size=8, name_type_dataset='rgb'):
        train_seq = cls(data_dir, path_to_csv, batch_size, folds=train_folds, type_set=name_type_dataset)
        valid_seq = cls(data_dir, path_to_csv, batch_size, folds=valid_folds, type_set=name_type_dataset)
        test_seq = cls(data_dir, path_to_csv, batch_size, folds=test_folds, type_set=name_type_dataset)
        return train_seq, valid_seq, test_seq

    def __init__(self, data_dir, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None, folds=None, type_set='rgb', size=128):
        if folds is None:
            folds = [1, 3]

        self.__data_dir__ = data_dir
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.__type_set__ = type_set
        self.__img_size__ = size
        num_ch, chs = self.__getChannels__()
        self.__num_ch__ = num_ch
        self.__chs__ = chs

        # CSV 파일을 로드하여 필요한 파일 리스트 및 레이블 정보 저장
        if path_to_csv is None:
            self.list_files = None
            self.y = None
        else:
            data = pd.read_csv(path_to_csv)
            data = data[data['fold'].isin(folds)]
            self.list_files = data['patient_id'].values
            data.drop('fold', inplace=True, axis=1) 
            data.drop('patient_id', inplace=True, axis=1)
            self.y = data.values  # 데이터프레임을 넘파이 배열로 변환

        if end_idx is None:
            self.end_idx = len(self.list_files)
        else:
            self.end_idx = end_idx

    @property
    def n_classes(self):
        return 0 if self.y is None else self.y.shape[1]

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
    
    def __getChs__(self):
        return self.__num_ch__

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        batch_files = self.list_files[start:end]

        
        
        batch_data = []
        for file in batch_files:
            set_temp = np.zeros((self.__img_size__, self.__img_size__, self.__num_ch__), dtype=np.uint8)
            img_ = io.imread(f'{self.__data_dir__}/{file}.tif')
            x = np.moveaxis(img_, 0, 2)
            # (128, 128, 23)
            for idx in range(self.__num_ch__):
                set_temp[:,:,idx] = x[:,:,self.__chs__[idx]]
            batch_data.append(set_temp)
        
        batch_data = np.expand_dims(np.array(batch_data), axis=-1)
        
        if self.y is None:
            return batch_data
        else:
            batch_labels = self.y[start:end]
            return batch_data, batch_labels
        
    def __getChannels__(self):
        '''
        types
            ['rgb', 'prgb', 'sp', 'psp', rgb+uv, prgb+uv, rgb+prgb+uv, sp+psp, sp+psp+uv, sp+uv, psp+uv]
        '''
        if self.__type_set__ == 'rgb':
            num_ch = 3
            chs = [0,1,2]
        elif self.__type_set__ == 'prgb':
            num_ch = 3
            chs = [12,13,14]
        elif self.__type_set__ == 'sp':
            num_ch = 8
            chs = [3,4,5,6,7,8,9,10]
        elif self.__type_set__ == 'psp':
            num_ch = 8
            chs = [15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'rgb+uv':
            num_ch = 4
            chs = [0,1,2,11]
        elif self.__type_set__ == 'prgb+uv':
            num_ch = 4
            chs = [11, 12, 13, 14]
        elif self.__type_set__ == 'sp+uv':
            num_ch = 9
            chs = [3,4,5,6,7,8,9,10, 11]
        elif self.__type_set__ == 'psp+uv':
            num_ch = 9
            chs = [11, 15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'rgb+prgb':
            num_ch = 6
            chs = [0,1,2, 12, 13, 14]
        elif self.__type_set__ == 'rgb+prgb+uv':
            num_ch = 7
            chs = [0,1,2,11, 12, 13, 14]
        elif self.__type_set__ == 'sp+psp':
            num_ch = 16
            chs = [3,4,5,6,7,8,9,10,15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'sp+psp+uv':
            num_ch = 17
            chs = [3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22]
        elif self.__type_set__ == 'sp_best':
            num_ch = 5
            chs = [5,6,7,8,9]
        elif self.__type_set__ == 'psp_best':
            num_ch = 5
            chs = [15, 17, 18, 19, 22]
        elif self.__type_set__ == 'all':
            num_ch = 23
            chs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20,21,22]
        else :
            num_ch = -1
            return -1, -1, -1, -1, -1, -1, num_ch
        return num_ch, chs


def plot_PRC(n_classes, y_test, y_score, dir, filename, flag_save=False):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    if flag_save:
        plt.savefig('{}/graph_PRC_{}.png'.format(dir, filename)) #precision-recall curve
    plt.clf()

    return average_precision["micro"], precision["micro"], recall["micro"]

def plot_ROC(n_classes, y_test, y_score, dir, filename, flag_save=False):
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if flag_save:
        plt.savefig('{}/graph_ROC_{}.png'.format(dir, filename))
    plt.clf()
    return roc_auc["macro"], roc_auc["micro"]

def vis(history,dir, filename, name, flag_save=False) :
    if name == 'loss':
        # plt.ylim([0, 50])
        plt.ylim([0, 0.5])
        legend_loc = 'upper right'
    else :
        plt.ylim([0, 1])
        legend_loc = 'lower right'
        
    plt.title(f"{name.upper()}_{filename}")
    plt.xlabel('epochs')
    plt.ylabel(f"{name.lower()}")
    plt.grid(True, alpha=0.2)
    value = history.history.get(name)
    val_value = history.history.get(f"val_{name}",None)
    epochs = range(1, len(value)+1)
    plt.xticks(np.arange(1, len(value)+1, 1))
    plt.plot(epochs, value, 'b-', label=f'training {name}')
    if val_value is not None :
        plt.plot(epochs, val_value, 'r-', label=f'validation {name}')
    plt.legend(loc=legend_loc, fontsize=9 , ncol=1)
    if flag_save:
        plt.savefig('{}/graph_{}_{}'.format(dir, name,  filename))
        plt.clf()
    
def plot_history(history, dir, filename) :
    # key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
    key_value = sorted(list(set([i.split("val_")[-1] for i in list(history.history.keys())])))

    for idx , key in enumerate(key_value) :
        vis(history, dir, filename, key, flag_save=True)

    plt.figure(figsize=(12, 4))
    for idx , key in enumerate(key_value) :
        plt.subplot(1, 3, idx+1)
        vis(history, dir, filename, key)
        ## TODO - savefigs()
    
    plt.tight_layout()
    plt.savefig('{}/graph_total'.format(dir))
    plt.clf()
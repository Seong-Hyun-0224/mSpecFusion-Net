import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import csv
from utils.fin_load_data import load_data_fin
from tensorflow_addons.metrics import F1Score
from keras.applications import ResNet50, EfficientNetV2L
from keras.applications import Xception, NASNetLarge, InceptionV3, DenseNet201

import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp


# def plot_ROC(n_classes, y_test, y_score, dir, filename, flag_save=False):
#     # Plot linewidth.
#     lw = 2

#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#     # Compute macro-average ROC curve and ROC area

#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#     # Finally average it and compute AUC
#     mean_tpr /= n_classes

#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#     # Plot all ROC curves
#     plt.figure(1)
#     plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)


#     # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     # for i, color in zip(range(n_classes), colors):
#     #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#     #          label='ROC curve of class {0} (area = {1:0.2f})'
#     #          ''.format(i, roc_auc[i]))

#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
#     plt.legend(loc="lower right")
#     if flag_save:
#         plt.savefig('result_dl/{}/graph_ROC_{}.png'.format(dir, filename))
#     plt.clf()

if __name__ == "__main__":
    dirname =  'result'
    files = glob.glob('{}/*.csv'.format(dirname))
    for file in files[1:]:
        plt.figure(1)
        name = os.path.basename(file)
        print(name)
        # result = pd.read_csv(file)
        # print(result)
        # print(result.iloc[0])
        # print(result['mac_roc_auc'])
        
        data = list()
        f = open(file,'r')
        rea = csv.reader(f)
        for row in rea:
            data.append(row)
        f.close
        for idx, line in enumerate(data[1:]):
            if idx%2 == 0:
                temp = line
            else :
                if temp[-3] > line[-3]:
                    metric = 'a'
                else :
                    metric = 'f'
                date = line[0]
                name_model = line[1]
                name_data = line[2]
                name_type_dataset_ = line[5]
                # print(date, model, metric, name_data, name_type_dataset_)

                # read dataset
                X_train, X_val, X_test, y_train, y_val, y_test, num_ch = load_data_fin(name_data, type_set=name_type_dataset_)

                # read saved model
                data_size = 128
                if name_model == 'res50':                   
                    model = ResNet50(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=max, classes=2)
                elif name_model == 'effv2l':
                    model = EfficientNetV2L(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=max, classes=2, include_preprocessing=False)
                elif name_model == 'nasnetlarge':
                    model = NASNetLarge(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=None, classes=2)
                elif name_model == 'xception':
                    model = Xception(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=None, classes=2)
                elif name_model == 'inception':
                    model = InceptionV3(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=None, classes=2)
                elif name_model == 'dense':
                    model = DenseNet201(include_top=True, weights=None, input_shape=(data_size, data_size, num_ch), pooling=None, classes=2)

                set_optimizer = 'adam'
                model.compile(loss='categorical_crossentropy', metrics=['accuracy', F1Score(num_classes=2, average='macro')], optimizer=set_optimizer)

                dirname = f'result_dl/{date}_{name_model}_{name_data}_{name_type_dataset_}_adam'
                filename = f'weight_{name_model}_{name_data}_{name_type_dataset_}_adam_{metric}.h5'
                model.load_weights(f'{dirname}/{filename}')

                y_score = model.predict(X_test)
                n_classes = 2

                # roc_auc_macro, roc_auc_micro = plot_ROC(n_classes, y_test, y_score, dirname, '{}{}'.format(filename, name), flag_save=True)
                # draw graph

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

                # Plot all ROC curves
                colors = {"sp+psp+uv" : 'deeppink', "sp+psp":'red',
                           "sp":'aqua', "psp":'darkorange',
                             "sp+uv":'cornflowerblue', "psp+uv":'blue',
                               "rgb":'limegreen', "rgb+uv":'yellow',
                               "prgb":'violet', "prgb+uv":'green',
                               "rgb+prgb":'skyblue', "rgb+prgb+uv":'dodgerblue',
                                 }
                plt.plot(fpr["micro"], tpr["micro"],
                    label=f'{name_type_dataset_} ({roc_auc["micro"]:0.2f})',
                    color=colors[name_type_dataset_], linewidth=1)
                
                # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                # for i, color in zip(range(n_classes), colors):
                #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                #          label='ROC curve of class {0} (area = {1:0.2f})'
                #          ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        if True:
            plt.savefig('result_graph/graph_ROC_{}.png'.format(name_model))
        plt.clf()

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
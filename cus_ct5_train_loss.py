import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings('ignore')
import pickle
import datetime
import numpy as np
import shutil
import tensorflow as tf
import keras
from keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
from keras.metrics import Precision, Recall
from tensorflow_addons.optimizers import AdamW
from keras import layers, regularizers
from models.custom_ct5 import *

import random
from utils.helper import makedirs, writeCSV, sendMSG
from utils.helper_dl import plot_history, plot_PRC, plot_ROC, Datast3DLoader, split_data
from keras.callbacks import EarlyStopping
import argparse

class SaveBestSmoothF1(tf.keras.callbacks.Callback):
    def __init__(self, filepath_template, smoothing_window=5, min_train_f1=0.8, result_file='model_result.txt'):
        super().__init__()
        self.filepath_template = filepath_template
        self.smoothing_window = smoothing_window
        self.min_train_f1 = min_train_f1
        self.best_epoch = -1
        self.best_val_f1 = -np.inf
        self.best_train_f1 = -np.inf
        self.result_file = result_file
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        train_f1 = logs.get('f1_score')
        val_f1 = logs.get("val_f1_score")
        if val_f1 is not None:
            self.val_f1_scores.append(val_f1)

            if len(self.val_f1_scores) >= self.smoothing_window:
                smoothed_val_f1 = np.mean(self.val_f1_scores[-self.smoothing_window:])
                if smoothed_val_f1 > self.best_val_f1 and train_f1 >= self.min_train_f1:
                    self.best_val_f1 = val_f1
                    self.best_epoch = epoch
                    self.best_train_f1 = train_f1
                    self.model.save_weights(self.filepath_template)
                    # name = self.filepath_template.replace('.h5', '')
                    # filepath = f'{name}_{epoch}.h5'
                    # self.model.save_weights(filepath)
                    # print(f"\nModel saved at '{filepath}' with smoothed val_f1={self.best_val_f1:.4f} train_f1={self.best_train_f1:.4f} at epoch {self.best_epoch}.")

    def on_train_end(self, logs=None):
        if self.best_epoch != -1:
            # result_message = (f"Training complete. Best validation F1 score: {self.best_val_f1:.4f}, train F1 score: {self.best_train_f1:.4f} at epoch {self.best_epoch}\n")
            result_pkl = {
                        "train_f1": self.best_train_f1,
                        "val_f1": self.best_val_f1,
                        "epoch": self.best_epoch
            }

            with open(self.result_file, 'wb') as file_pi:
                pickle.dump(result_pkl, file_pi)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cpsc', help='Directory for data dir')
    parser.add_argument('--num_sample', type=int, default=4, help='Seed to split data')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--regular', type=bool, default=False, help='Regularizaion')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Select optimizer')
    parser.add_argument('--epoch', type=int, default=150, help='Training epoch')
    parser.add_argument('--gpu', default=0, help='Use GPU num')
    parser.add_argument('--model', type=str, default='res50', help='Path to saved model')
    parser.add_argument('--save_result', action='store_true', help='Flag to save the result.', default=True)
    parser.add_argument('--result_dir', type=str, default='./result', help='Path to saved model')
    parser.add_argument('--name_type_dataset', type=str, default='sp', help='Select the type of dataset combination')
    parser.add_argument('--comment', type=str, default='focal_early_stop_lr_decay', help='Comment for model')
    parser.add_argument('--mix_type', type=str, default='con', help='Comment for model')
    parser.add_argument('--flag_aug', type=bool, default=False, help='Augmentation')
    parser.add_argument('--model_size', type=str, default='tiny', help='Augmentation')
    return parser.parse_args()

def main(args):
    SEED = 42

    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    Model = {'cus_ct' : cus_ct}
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    msg = 'gpu{}_{}_{}_{}_{}_sample_{}'.format(args.gpu, date, args.dataset, args.model, args.name_type_dataset, args.num_sample)
    sendMSG('MSI_train', f'ST_{msg}', 'etc')
    path_dir = '/home/project/data_hdd0/result_dl'

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    ## load dataset
    # '20230525_all_size128_ac_calib'
    data_dir = f'data/{args.dataset}'
    path_to_csv = 'data/labels_{}.csv'.format(args.dataset)

    train_folds, val_folds, test_folds = split_data(seed=args.num_sample)
    train_seq, valid_seq, test_seq = Datast3DLoader.split(
            data_dir, path_to_csv, train_folds, val_folds, test_folds, args.batch_size, name_type_dataset='all')

    # TODO model selection
    filename = f'{args.model}_{args.name_type_dataset}_s{args.num_sample}'
    
    dirname = f'{date}_{filename}_{args.comment}'
    path_dir_final = f'{path_dir}/{dirname}'
    makedirs(f'{path_dir_final}')

    # model = Model[args.model](num_channel = num_ch)
    model = Model[args.model](num_class = 2, name_type_dataset = args.name_type_dataset, mix_type = args.mix_type, flag_aug= args.flag_aug)

    if args.regular:
        l2_lambda = 0.001
        l2_loss = lambda: tf.add_n([regularizers.l2(l2_lambda)(layer.kernel) for layer in model.layers if hasattr(layer, 'kernel')])

        # Add regularization loss to model's total loss
        model.add_loss(l2_loss)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=300, decay_rate=0.9, staircase=True)
    
    # model.compile(loss=tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.2), metrics=['accuracy', F1Score(num_classes=2, average='weighted')], optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(label_smoothing=0.2), metrics=['accuracy', F1Score(num_classes=2, average='weighted')], optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    
    logdir="logs/{}".format(dirname)


    # save_f = tf.keras.callbacks.ModelCheckpoint(f'{path_dir_final}/weight_{filename}_f.h5', monitor='val_f1_score', mode='max', save_best_only=True)
    # f'{path_dir_final}/weight_{filename}_f.h5'


    # best_weight_epoch_{{epoch:02d}}
    save_best_smooth_f1 = SaveBestSmoothF1(
                                                filepath_template=f"{path_dir_final}/weight_{filename}_f.h5",
                                                smoothing_window=5,
                                                # min_train_f1=0.5,
                                                min_train_f1=0.7, # changed by KSH, 2025.09.16.
                                                result_file=f"{path_dir_final}/result.pkl"
                                            )


    callback = [save_best_smooth_f1, EarlyStopping(monitor='val_loss', patience=50, mode='min')]

    history = model.fit(train_seq, batch_size=args.batch_size, epochs=args.epoch, validation_data=valid_seq, callbacks=callback)

    with open(f'{path_dir_final}/history_{filename}.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    plot_history(history, dir=path_dir_final, filename=filename)

    train_f1s = history.history.get(f"f1_score",None)
    train_accs = history.history.get(f"accuracy",None)
    val_accs = history.history.get(f"val_accuracy",None)
    val_f1s = history.history.get(f"val_f1_score",None)

    if os.path.exists(f'{path_dir_final}/result.pkl'):
        with open(file=f'{path_dir_final}/result.pkl', mode='rb') as f:
            model_result = pickle.load(f)

        # read file
        for name in ['_f']:
            if name == '_f':
                # idx_best = np.argmax(val_f1s)
                idx_best = model_result['epoch']
                best_val_f1 = model_result['val_f1']
                best_val_acc = val_accs[idx_best]
                best_train_acc = train_accs[idx_best]
                best_train_f1 = model_result['train_f1']

            model.load_weights(f'{path_dir_final}/weight_{filename}{name}.h5')

            # test model
            results_test = model.evaluate(test_seq, batch_size=args.batch_size)
            print("test loss, test acc, test f1_score :", results_test)

            with open(f'{path_dir_final}/history_{filename}{name}.txt', 'w') as f:
                f.write('loss, acc, f1\n{:.4f}, {:.4f}, {:.4f}'.format(results_test[0], results_test[1], results_test[2]))

            if args.save_result:
                if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir, exist_ok=True)

            y_score = model.predict(test_seq)
            n_classes = 2
            
            # save result
            roc_auc_macro, roc_auc_micro = plot_ROC(n_classes, test_seq.y, y_score, path_dir_final, '{}{}'.format(filename, name), flag_save=True)
            prc_auc_micro, precision_micro, recall_micro = plot_PRC(n_classes, test_seq.y, y_score, path_dir_final, '{}{}'.format(filename, name), flag_save=True)
        
            row = [date, args.model, args.dataset, args.optimizer, args.name_type_dataset, 
                    args.batch_size, args.num_sample, 
                    '{:.4f}'.format(best_train_acc), '{:.4f}'.format(best_train_f1),
                    '{:.4f}'.format(best_val_acc), '{:.4f}'.format(best_val_f1),
                        '{:.4f}'.format(results_test[0]), '{:.4f}'.format(results_test[1]),'{:.4f}'.format(results_test[2]),
                        '{:.4f}'.format(roc_auc_macro),
                            '{:.4f}'.format(roc_auc_micro), '{:.4f}'.format(prc_auc_micro)]
            writeCSV(row, filename=f'./result/result_{args.model}_{args.comment}.csv')
        graph_filename1 = f'{path_dir_final}/graph_total.png'
        shutil.copy2(graph_filename1, f'{path_dir}/2025_graphs/{args.model}_ep{dirname}_sample{args.num_sample}.png')
        
        file_path = f'{path_dir_final}/weight_{filename}{name}.h5'
        if os.path.exists(file_path):
            os.remove(file_path)

        sendMSG('MSI_train', 'ED_{}_f1_{:.4f}'.format(msg, results_test[2]), 'done')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    # data_with_channel = tf.expand_dims(data, axis=-1) 
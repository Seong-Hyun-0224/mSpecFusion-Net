import csv
import pickle
import numpy as np
from utils.helper import writeCSV

f = open('result_dl/result_effv2l_total.csv','r')
rdr = csv.reader(f)

for id, line in enumerate(rdr):
    # get name
    dirname = '{}_{}_{}_{}_{}'.format(line[0], line[1], line[2], line[5], line[3])
    filename = '{}_{}_{}_{}'.format(line[1], line[2], line[5], line[3])
    
    # load history
    dat = pickle.load(open('result_dl/{}/history_{}.bin'.format(dirname, filename), 'rb'))
    
    # update train / val , acc / f1
    # row = [date, model, dataname, optimizer, #ch,
            #  name, epoch, loss, tr_acc, tr_f1,
            #  v_acc, v_f1, te_acc, te_f1, mac_roc_auc,
            #  mic_roc_auc, mic_prc_auc]

    train_f1s = dat.get(f"f1_score",None)
    train_accs = dat.get(f"accuracy",None)
    val_accs = dat.get(f"val_accuracy",None)
    val_f1s = dat.get(f"val_f1_score",None)
    
    if id%2 == 0:
       idx_best = np.argmax(val_accs)
       best_val_acc = np.max(val_accs)
       best_val_f1 = val_f1s[idx_best]
       best_train_acc = train_accs[idx_best]
       best_train_f1 = train_f1s[idx_best]

    else :
       idx_best = np.argmax(val_f1s)
       best_val_f1 = np.max(val_f1s)
       best_val_acc = val_accs[idx_best]
       best_train_acc = train_accs[idx_best]
       best_train_f1 = train_f1s[idx_best]

    new_line = []
    for l in line:
       new_line.append(l)
    new_line[7] = '{:.4f}'.format(best_train_acc)
    new_line[8] = '{:.4f}'.format(best_train_f1)
    new_line[9] = '{:.4f}'.format(best_val_acc)
    new_line[10] = '{:.4f}'.format(best_val_f1)

    writeCSV(new_line, filename='result_dl/result_effv2l_total_update.csv')  

f.close()

# date, model, dataname, optimizer, ch, dataset, epoch, tr_acc, tr_f1,v_acc, v_f1, loss, te_acc, te_f1, mac_roc_auc,mic_roc_auc, mic_prc_auc

import pandas as pd

read_file = pd.read_csv (r'result_dl/result_effv2l_total_update.csv')
read_file.to_excel (r'effv2l.xlsx', index = None, header=True)
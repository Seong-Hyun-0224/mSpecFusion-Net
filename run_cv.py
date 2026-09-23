"""
Launcher for mSpecFusion-Net (MICCAI 2026).

Runs patient-wise 5-fold cross-validation sequentially on a single GPU.

Only the final model is released in this repository. The fusion-strategy
baselines (early/middle/late fusion) and the generic backbone comparisons
reported in Tables 3 and 5 of the paper are not included.
"""

import os
import datetime
import argparse

# ---------------------------------------------------------------------------
# Configuration reported in the paper
# ---------------------------------------------------------------------------
TRAIN_SCRIPT = 'cus_ct5_train_loss'   # mSpecFusion-Net (iterative fusion)
NAME_MODEL = 'cus_ct'
MODEL_SIZE = 'base'
BATCH_SIZE = 16
NUM_EPOCHS = 150
LR = 1e-4
NUM_FOLDS = 5                          # patient-wise 5-fold cross-validation
MIX_TYPE = 'con'
FLAG_AUG = False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run mSpecFusion-Net 5-fold cross-validation on a single GPU.')
    parser.add_argument('--gpu', default=0, type=int,
                        help='CUDA device index (selects the device only)')
    parser.add_argument('--dataset', default='20230525', type=str,
                        help='Dataset name under data/')
    parser.add_argument('--name_type_dataset', default='sp_4+psp_4+uv', type=str,
                        help='Modality combination. sp: Co-P MSI, psp: Cross-P MSI, '
                             'uv: autofluorescence. The _4 suffix applies PCA top-4 '
                             'band selection. Default is the paper configuration.')
    parser.add_argument('--comment', default='mspecfusion', type=str,
                        help='Tag appended to the result directory name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    for fold in range(NUM_FOLDS):
        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f'[fold {fold}/{NUM_FOLDS - 1}] {date} | {NAME_MODEL} | '
              f'{args.name_type_dataset} | batch {BATCH_SIZE}')
        print('##############################')

        os.system(
            f'python {TRAIN_SCRIPT}.py '
            f'--dataset {args.dataset} '
            f'--num_sample {fold} '
            f'--model {NAME_MODEL} '
            f'--gpu {args.gpu} '
            f'--name_type_dataset {args.name_type_dataset} '
            f'--comment {args.comment} '
            f'--batch_size {BATCH_SIZE} '
            f'--epoch {NUM_EPOCHS} '
            f'--lr {LR} '
            f'--regular False '
            f'--mix_type {MIX_TYPE} '
            f'--flag_aug {FLAG_AUG} '
            f'--model_size {MODEL_SIZE}'
        )
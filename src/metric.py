import torch.nn as nn
from pytorch_lightning.metrics.sklearns import AUROC, AveragePrecision

roc_auc = AUROC(average='macro')
average_precision = AveragePrecision(average='macro')

def get_auc(y_score, y_true):
    # for Validation sanity check:
    if y_true.shape[0] == 1:
        return 0,0
    else:
        roc_aucs  = roc_auc(y_score.flatten(0,1), y_true.flatten(0,1))
        pr_aucs = average_precision(y_score.flatten(0,1), y_true.flatten(0,1))
        return roc_aucs, pr_aucs
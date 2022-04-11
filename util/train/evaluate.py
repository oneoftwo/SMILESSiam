import sklearn


def count_learnable_parameters(model):
    num_param = sum(p.numel() \
            for p in model.parameters() if p.requires_grad)
    return num_param


def calc_precision(pred, true):
    from sklearn.metrics import precision_score
    a = precision_score(true, pred, zero_division=0)
    b = precision_score(true, pred, zero_division=1)
    if (a == 0) and (b==1):
        return -1
    else:
        return a


def calc_recall(pred, true):
    from sklearn.metrics import recall_score
    return recall_score(true, pred)


def calc_roc_auc(score, true):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(true, score)


def calc_accuracy(pred, true):
    from sklearn.metrics import accuracy_score
    return accuracy_score(true, pred)


def calc_prc_auc(score, true):
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    p, r, ths = precision_recall_curve(true, score)
    asdf = auc(r, p)
    return asdf


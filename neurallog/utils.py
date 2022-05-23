from sklearn.metrics import precision_recall_fscore_support, classification_report

def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.
    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list
    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1


def report(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.
    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list
    Returns
    -------
        Precision, Recall and F1 for each class
    """
    r = classification_report(y_true, y_pred)
    return r
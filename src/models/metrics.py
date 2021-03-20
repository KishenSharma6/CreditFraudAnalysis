from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Evaluation:
    pass
def classification_metrics(predictions, actual):
    """Return dictionary containing accuracy, precision, recall, and f1 score for 
    classification tasks.

    Args:
        predictions (array): prediction values
        actual (array): actual value

    Returns:
        [dictionary]: returns dictionary containing classification evaluation metrics.
    """
    accuracy= round(accuracy_score(actual, predictions),3)
    precision= round(precision_score(actual, predictions),3)
    recall= round(recall_score(actual, predictions),3)
    f1score= round(f1_score(actual, predictions),3)
    scores= {'accuracy':accuracy,
             'precision':precision,
             'recall':recall,
             'f1_score':f1score
            }
    return scores
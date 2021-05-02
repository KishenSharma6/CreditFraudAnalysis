from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, fbeta_score, plot_confusion_matrix


class Evaluation:
    def __init__(self, predictions, actual):
        self.predictions= predictions
        self.actual= actual
    
    def classification_metrics(self, beta= 1.0):
        """Return dictionary containing accuracy, precision, recall, and f1 score for 
        classification tasks.

        Args:
            beta (float): Determines the weight of recall in the combined f score.

        Returns:
            [dictionary]: returns dictionary containing classification evaluation metrics.
        """
        accuracy= round(accuracy_score(self.actual, self.predictions),3)
        precision= round(precision_score(self.actual, self.predictions),3)
        recall= round(recall_score(self.actual, self.predictions),3)
        fScore= round(fbeta_score(self.actual, self.predictions, beta = beta),3)
        scores= {'accuracy':accuracy,
                'precision':precision,
                'recall':recall,
                'fbeta_score':fScore
                }
        return scores

    def confusion_matrix(self):
        return confusion_matrix(self.actual, self.predictions)
    
    def confustion_matrix_plot(self, estimator):
        pass
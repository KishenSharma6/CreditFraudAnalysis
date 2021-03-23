from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
print('hi')
class Models:
    def __init__(self, data, target):
        self.data= data
        self.target= target

    def initialize(self):
        logReg= LogisticRegression(penalty='l2', C=1.0, random_state= 24)
        ranForest= RandomForestClassifier(n_estimators= 100, random_state= 24,  )

from sklearn.metrics import r2_score



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

class Models:
    def __init__(self, data, target):
        self.data= data
        self.target= target
    
    def sample_data(self, n):
        """Create sample of initial data for testing purposes

        Args:
            n (integer): Number of random rows to generate.
        
        Returns: Dataframe containing sample from data

        """
        return self.data.sample(n=n)

    def split_data(self, test_size):
        """Split data into training/test sets and assigns them to object

        Args:
            test_size (float): % of data you would like to be reserved for the test set
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, random_state= 24, test_size= test_size)
        self.X_train
        self.X_test
        self.y_train
        self.y_test


    def initialize(self):
        logReg= LogisticRegression(penalty='l2', C=1.0, random_state= 24)
        ranForest= RandomForestClassifier(n_estimators= 100, random_state= 24,  )

    def over_sampling(self):
        """Applies SMOTE's oversampling technique, prints the results of the transformation, and assigns
        transformed data to object.
        """
        sm= SMOTE(random_state= 24)
        X_train_sm, y_train_sm= sm.fit_resample(self.X_train, y_train)
        print('Shape of X_train prior to SMOTE: %s' % (self.X_train.shape))
        print('Shape of X_train after SMOTE: %s' % (self.X_train_sm.shape))
        print('Class balance after SMOTE: \n%s' % (y_train_sm.shape.value_counts(normalize= True) * 100))
        self.X_train_SMOTE= X_train_sm
        self.y_train_SMOTE= y_train_sm

    
    def under_sampling(self, X_train, y_train):
        pass
    
    def confustion_matrix_plot(self, estimator):
        #for each iteration of initialize, print out a confusion matrix for the base models
        pass
    
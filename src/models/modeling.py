import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Data:
    def __init__(self, data, target):
        self.data= data
        self.target= target
    
    def sample_data(self, n):
        """Create sample of initial data for testing purposes

        Args:
            n (integer): Number of random rows to generate.
        
        Returns: Dataframe containing sample from data

        """
        self.data_sample= self.data.sample(n=n)
    
    def about_sample(self):
        assert hasattr(self, 'data_sample')== True, "Object does not have sample data. Use method data_sample()"
        return self.data_sample.describe()

    def split_data(self, test_size):
        """Split data into training/test sets and assigns them to object. Print X_train, y_train, X_test,
        y_test shapes

        Args:
            test_size (float): % of data you would like to be reserved for the test set
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, random_state= 24, test_size= test_size)
        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test
        

    
    def over_sampling(self):
        """Applies SMOTE's oversampling technique, prints the results of the transformation, and assigns
        transformed data to object.
        """
        assert hasattr(self, 'X_train') == True, "Object does not contain X_train attribute. Run method split_data() to create train/test splits"
        
        # sm= SMOTE(random_state= 24)

        # X_train_sm, y_train_sm= sm.fit_resample(self.X_train, self.y_train)
        
        # print('Shape of X_train prior to SMOTE: %s' % (self.X_train.shape))
        # print('Shape of X_train after SMOTE: %s' % (self.X_train_sm.shape))
        # print('Class balance after SMOTE: \n%s' % (y_train_sm.shape.value_counts(normalize= True) * 100))
        # self.X_train_SMOTE= X_train_sm
        # self.y_train_SMOTE= y_train_sm

    
    def under_sampling(self, X_train, y_train):
        pass


class Models:
    def __init__(self, X_train, y_train):
        self.X_train= X_train
        self.y_train= y_train

    def base_model_evaluation(self, preprocessor, models, scoring, names):
        #Create preprocessor to process numerical and cat features
        for model, name in zip(models, names):
            clf= Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', model)])
            cv_results= cross_validate(estimator= clf, X= self.X_train, y= self.y_train, 
                                       cv= 5, scoring=scoring)
            return cv_results
            
            
           

    def confustion_matrix_plot(self, estimator):
        #for each iteration of initialize, print out a confusion matrix for the base models
        pass
    
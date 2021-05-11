from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler


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
        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test

    def standardize_data(self, cols):
        """Standardize X_train and X_test of object

        Args:
            cols (List): List of columns that contain data you would like to apply StandardScaler() to.
        """
        try:
            training_data= self.X_train.copy()
            test_data= self.X_test.copy()
            
            training_to_transform= training_data[cols]
            training_remainder= training_data.drop(cols, axis= 1)

            test_to_transform= test_data[cols]
            test_remainder= test_data.drop(cols, axis= 1)
            
            scaler= StandardScaler()
            
            X_train_transformed= scaler.fit_transform(training_to_transform)
            X_test_transformed= scaler.transform(test_to_transform)
            
            self.X_train_transformed= pd.concat([pd.DataFrame(X_train_transformed), training_remainder])
            self.X_test_transformed= pd.concat([pd.DataFrame(X_test_transformed),test_remainder])

            print('X_train and X_test attributes have been successfully standardized')
        except AttributeError:
            print('ERROR Data needs to be split. Run method split_data on object')

    def base_model_evaluation(self):
        logReg= LogisticRegression(penalty='l2', C=1.0, random_state= 24)
        ranForest= RandomForestClassifier(n_estimators= 100, random_state= 24)
        gradBoost= GradientBoostingClassifier(random_state= 24)
        adaBoost= AdaBoostClassifier(random_state= 24)
        knn= KNeighborsClassifier(n_jobs= -1)
        naiveBayes= GaussianNB()

        models= [logReg, ranForest, gradBoost, adaBoost, knn, naiveBayes]
        names= ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'ADA Boost', 'KNN', 'Naive Bayes']
        scoring= {'Recall':'recall', 'F2': make_scorer(fbeta_score, beta= 2)}
        
        try:
            for model, name in zip(models, names):
                cv_results= cross_validate(estimator= model, X= self.X_train, y= self.y_train, 
                                       cv= 5, scoring=scoring)
                print('===========================')
                print('%s Performance:\n' % (name))
                print('Best CV Recall Score: %s F2 Score: %s' % (cv_results['test_Recall'], cv_results['test_F2']))
                print('Average CV Recall Score: %s F2 Score: %s' % (cv_results['test_Recall'], cv_results['test_F2']))
                print('Recall SD: %s F2 SDL %s' % (cv_results['test_Recall'], cv_results['test_F2']))
                print('===========================')
        except AttributeError:
            print('ERROR. Object does not have train/test splits. Run method split_data() first.')

    def over_sampling(self):
        """Applies SMOTE's oversampling technique, prints the results of the transformation, and assigns
        transformed data to object.
        """
        print('Be sure to add a type test')
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
    
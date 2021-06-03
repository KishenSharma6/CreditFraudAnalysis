import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

class Stats_Tests:
    def __init__(self, data):
        self.data= data

    def add_constant(self):
        """Add constant to self.data to be later used in VIF_analysis
        """
        assert 'const' not in self.data.columns, 'Constant already exists in self.data!'
        self.data= self.data.assign(const=1)
        return "Constant column appended to self.data"

    def VIF_analysis(self):
        """Apply VIF analysis to data

        Returns:
            Series: Returns Pandas series containing VIF scores
        """
        #Make sure constant is present in data
        assert 'const' in self.data.columns, 'Data does not contain constant. Use add_constant() method to append constant column to self.data'
        
        return pd.Series([variance_inflation_factor(self.data.values, i) 
               for i in range(self.data.shape[1])], 
              index=self.data.columns)

    def drop_cols(self, columns):
        """Drops requested columns.

        Args:
            columns ([list]): List of strings containing columns names you would like dropped from self.data
        """
        self.data= self.data.drop(columns= columns, axis=1)
        return 'Requested columns have been removed.'
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class TransformAle(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data.fillna(data["NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO"].mean(axis='columns')
        return data

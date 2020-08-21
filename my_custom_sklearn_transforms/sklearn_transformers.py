from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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
        ct = ColumnTransformer(
            [("1t1",'passthrough', slice(0,4)),
             ("1t2",SimpleImputer(
                missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
                strategy='mean',  # a estratégia escolhida é a alteração do valor faltante por uma constante
                verbose=0,
                copy=True
            ), [4,5,6,7,10]),
             ("1t3",SimpleImputer(
                missing_values=np.nan,  # os valores faltantes são do tipo ``np.nan`` (padrão Pandas)
                strategy='constant',  # a estratégia escolhida é a alteração do valor faltante por uma constante
                fill_value=1,
                verbose=0,
                copy=True
            ), [8]),
            ("1t4", 'passthrough', [9,10,11,12])])

        ct2 = ColumnTransformer(
            [("2t1",'passthrough', slice(0,8)),
             ("2t2","drop", [8]),
             ("2t3",'passthrough', slice(9,14))])

        ct3= ColumnTransformer(
            [("3t1",'passthrough', slice(0,4)),
             ("3t2",SimpleImputer(
                missing_values=0,
                strategy='mean',
                verbose=0,
                copy=True
            ), slice(4,8)),
             ("3t3",'passthrough', slice(8,13))])
        
        data= pd.DataFrame.from_records(
                data=ct3.fit_transform(ct2.fit_transform(ct.fit_transform(data))),  # o resultado SimpleImputer.transform(<<pandas dataframe>>) é lista de listas
                columns=data.columns  # as colunas originais devem ser conservadas nessa transformação
                )
        return data

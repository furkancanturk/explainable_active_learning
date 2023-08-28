from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class LinearClassifier(LogisticRegression):

    def __init__(self, **kwargs):
        super(LinearClassifier, self).__init__(**kwargs)
        
    def explain(self, X: np.array, y: np.array, raw_data:pd.DataFrame) -> pd.Series:
        return pd.Series((self.coef_[0].ravel() > 0).astype(int), index=raw_data.columns)
        
    def fit(self, X, y, *args, **kwargs):
        super(LinearClassifier, self).fit(X,y)
from typing import Optional
import pandas as pd
import numpy as np
from typing import Tuple

class UncertaintyBasedSampling():

    metric_names = ["entropy","least_confidence","margin_confidence","confusion_confidence","random"]
   
    @staticmethod
    def entropy(pdist):
        if len(pdist.shape) == 1:
            pdist = pdist.reshape(1,pdist.shape[0])
        return -1 * np.sum(pdist * np.log2(pdist + 1e-5), axis=1) / np.log2(pdist.shape[1])
    
    @staticmethod
    def least_confidence(pdist):
        if len(pdist.shape) == 1:
            pdist = pdist.reshape(1,pdist.shape[0])
        return (pdist.shape[1] / (pdist.shape[1] - 1)) * (1 - pdist.max(axis=1))
    
    @staticmethod
    def margin_confidence(pdist):
        if len(pdist.shape) == 1:
            pdist = pdist.reshape(1,pdist.shape[0])
        return 1 - (pdist.max(axis=1) - np.sort(pdist + 1e-5, axis=1)[:,-2])
    
    @staticmethod
    def random(pdist):
        if len(pdist.shape) == 1:
            pdist = pdist.reshape(1,pdist.shape[0])
        return np.random.rand(pdist.shape[0])
    
    @staticmethod
    def confusion_confidence(pdist):
        if len(pdist.shape) == 1:
            pdist = pdist.reshape(1,pdist.shape[0])
        return np.sum(1 + pdist - pdist.max(axis=1), axis=1)

    def __init__(self, metric:str):
        self.uncertainty_func = getattr(self, metric) #getting corresponding metric function belonging the class 

    def __str__(self) -> str:
        return f"UncertaintyBasedSampling(metric={str(self.uncertainty_func)})"

    def get_sample(self, data, size, model, return_uncertainty=False, *args, **kwargs):

        self.uncertainty_scores = pd.Series(self.uncertainty_func(model.predict_proba(data.values)), data.index).sort_values(ascending=False) 
        sample = data.loc[self.uncertainty_scores[:size].index]
        
        if return_uncertainty:
            return sample, self.uncertainty_scores[:size]
        
        return sample

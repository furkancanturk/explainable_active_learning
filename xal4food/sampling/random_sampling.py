import numpy as np
import pandas as pd
import random

class RandomSampling():
    def __init__(self, random_state):
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    def get_sample(self, data, size, *args, **kwargs):
        
        if type(data) is pd.DataFrame:
            return data.sample(size)
        
        random_idx = np.random.choice(data.shape[0], replace=False, size=size)
        return data[random_idx]

    def fit(self, *args, **kwargs):
        pass
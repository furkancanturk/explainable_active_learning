import numpy as np
import pandas as pd
from typing import List, Dict, Union

class SyntheticUser():
    
    def __init__(self, user_name: str, like_threshold: float, preferences: Dict, ing_scores: pd.Series, random_state=42, **kwargs):
        self.user_name = user_name
        self.like_threshold = like_threshold
        self.preferences = preferences
        self.random_state = random_state
        self.feature_scores = ing_scores
        self.ineffective_features = self.feature_scores[self.feature_scores.isna()].index.tolist()
    
    def set_labels(self, ingredients_df:pd.DataFrame, food_class_bounds:list=None):
        
        effective_features = list(set(ingredients_df.columns) - set(self.ineffective_features))
        #Utility function: Food utility is the mean of the ingredient scores.
        input_data = ingredients_df[effective_features]
        mean_scores = np.dot(input_data.values, self.feature_scores[effective_features].values) / input_data.sum(axis=1).values
        self.food_scores = pd.Series(mean_scores, index=ingredients_df.index)
    
        if food_class_bounds is None:
            self.food_class_bounds = [self.like_threshold, 10.1]
        else:
            self.food_class_bounds = food_class_bounds
        self.food_labels = pd.cut(self.food_scores, [0]+self.food_class_bounds, labels=[0,1]).astype(int)
        #self.food_labels = (self.food_scores > self.food_scores.mean()).astype(int)

    def get_labels(self, sample_idx: list, target_label=None, min_size=None):
        if target_label is None:
            return self.food_labels.loc[sample_idx]
        else:
            sample_labels = self.food_labels.loc[sample_idx]
            target_sample_labels = sample_labels[sample_labels == target_label]
            if min_size < len(target_sample_labels):
                size = np.random.randint(min_size, len(target_sample_labels))
                return target_sample_labels.sample(size)
            else:
                return target_sample_labels
            
    def get_preference(self, features: list) -> pd.Series:
        pref = (self.feature_scores[features] > self.food_scores.mean()).astype(int)
        pref[self.ineffective_features] = 2 #indicator for ineffective features 
        return pref

def get_feedback_from_synthetic_user(user: SyntheticUser, item: Union[pd.Series, pd.DataFrame], explanation: dict[str, list], reasoning_level ='local', *kwargs):
  
    item = item.squeeze()
    item_idx = item.name
    item_score = user.food_scores[item_idx]

    food_label = user.food_labels[item_idx]
    user_reasoning = pd.Series(np.nan, explanation['preference_factors'].keys())
    
    for feature, _ in explanation['preference_factors'].items():
        
        if feature not in user.ineffective_features:
            feature_score = user.feature_scores[feature]

            if reasoning_level == 'global':
                feature_effect = (feature_score > user.food_scores.mean()).astype(int)
            else: # i.e. preference_factor is 'local'
                feature_effect = (feature_score > item_score).astype(int)
        else:
            feature_effect = 2

        user_reasoning[feature] = feature_effect  
    
    feedback = dict(true_label=food_label, user_reasoning=user_reasoning)
 
    return feedback
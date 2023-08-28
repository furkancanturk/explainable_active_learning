import random
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from copy import deepcopy
from time import time
from typing import Union
from similarity_learning import augment_labeled_data
from oracle.user import SyntheticUser, get_feedback_from_synthetic_user
from data_builder import *

class ActiveLearning():

    def __init__(self, 
                model, 
                user: SyntheticUser, 
                sampling_strategy, 
                dataset: Union[FoodDataset, FoodDataset2],
                labels: pd.Series, 
                pseudo_labeled_idx: list,   
                unlabeled_idx: list,      
                target_idx: list,      
                prior_prob: float, 
                metric, 
                use_user_feedback: bool,
                explainer):

        self.model = deepcopy(model)
        self.user = deepcopy(user)
        self.dataset = deepcopy(dataset)
        self.data_transformer = deepcopy(self.dataset.data_transformer)
        self.sampling_strategy = deepcopy(sampling_strategy)
        self.X = None
        self.labels = deepcopy(labels)
        self.pseudo_labeled_idx = deepcopy(pseudo_labeled_idx)
        self.unlabeled_idx = deepcopy(unlabeled_idx)
        self.target_idx = deepcopy(target_idx)
        self.selected_sample_idx = []
        self.true_labeled_idx = list(set(self.labels.index) - set(self.pseudo_labeled_idx))
        self.prior_prob = prior_prob
        self.metric = metric
        self.use_user_feedback = use_user_feedback
        self.explainer = deepcopy(explainer)
        self.synthetic_sample = np.array([]).reshape(0, self.dataset.data.shape[1])
        self.synthetic_labels = np.array([])
        self.explanations = pd.Series(None, index=self.dataset.data.index, dtype=object)
        self.user_feedbacks = pd.Series(None, index=self.dataset.data.index, dtype=object)
        self.positive_features = set()
        self.negative_features = set()
        self.ineffective_features = set()
        self.labeled_sample_size_list = []
        self.scores = []
        self.pref_acc = []
        self.step_times = []
        self.step_counter = 0
        np.random.seed(dataset.random_state)
        random.seed(dataset.random_state)

    def step(self, batch_size: int, candidate_idx: list, augment_batches: bool, verbose=True):
     
        st = time()

        if self.step_counter == 0:
            self.update_model()
            self.score_model(self.dataset.data.loc[self.unlabeled_idx])

        new_sample = self.sampling_strategy.get_sample(self.X.loc[candidate_idx], batch_size, self.model, similarity_df=self.dataset.similarity_df)
         
        for idx, item in new_sample.iterrows():
            explanation = self.generate_explanation(idx=idx, new_sample=new_sample)        
            user_feedback = self.interact_with_user(item, explanation)
            self.explanations[idx] = explanation
            self.user_feedbacks[idx] = user_feedback
            self.labels[idx] = user_feedback['true_label']
        
        self.update_idx(new_sample.index)
        
        assert len( set(self.true_labeled_idx) & set(self.pseudo_labeled_idx)) == 0

        if augment_batches:
        
            labels, pseudo_labeled_idx = augment_labeled_data(self.dataset.similarity_df,
                                                                self.X.loc[self.target_idx], 
                                                                self.X.loc[self.labels.index], 
                                                                self.labels,
                                                                self.pseudo_labeled_idx,
                                                                self.prior_prob)

            if len(pseudo_labeled_idx) > 0:
                self.labels, self.pseudo_labeled_idx = labels, pseudo_labeled_idx
        
        if self.use_user_feedback:
            synthetic_X, synthetic_y = self.evaluate_feedback(new_sample, self.explanations[new_sample.index], self.user_feedbacks[new_sample.index])  
            self.update_model(synthetic_X, synthetic_y)
        else:
            self.update_model()

        score = self.score_model(self.dataset.data.loc[self.unlabeled_idx])
        
        self.step_counter += 1
        step_time = round(time() - st, 2)
        self.step_times.append(step_time)
        
        if verbose:
            print(f">> Step-{self.step_counter}: {self.metric.__name__}: {round(score,3)} | Labeled Sample size: {len(self.labels)} | Elapsed Time: {step_time}\n"+"--"*50)
                 
        return score
    
    def update_idx(self, new_labeled_idx: list):
        self.selected_sample_idx += list(new_labeled_idx)
        self.labeled_sample_size_list.append(len(new_labeled_idx))
        self.true_labeled_idx += list(new_labeled_idx)
        self.target_idx = list(set(self.target_idx) - set(new_labeled_idx))
        self.unlabeled_idx = list(set(self.unlabeled_idx) - set(new_labeled_idx))
        self.pseudo_labeled_idx = list(set(self.pseudo_labeled_idx) - set(new_labeled_idx))

    def score_model(self, raw_data: pd.DataFrame):
            
        X = self.data_transformer.transform(raw_data)
        X[list(self.ineffective_features)] = 0
        y_pred = self.model.predict(X)
        y_true = self.user.get_labels(raw_data.index)
        score = self.metric(y_true=y_true, y_pred=y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        self.scores.append([score, mcc])

        model_pref = self.explainer.explain(X.values, self.labels.values, raw_data)
        model_pref[list(self.ineffective_features)] = 2 #ineffective features
        user_pref = self.user.get_preference(raw_data.columns)
        pref_acc = (user_pref == model_pref).mean()
        self.pref_acc.append(pref_acc)

        return score
    
    def update_model(self, synthetic_X: np.ndarray = None, synthetic_y: np.ndarray = None):
  
        if type(synthetic_X) is np.ndarray: #i.e. synthetic_X is not None 
      
            self.synthetic_sample = np.vstack((self.synthetic_sample, synthetic_X))
            self.synthetic_labels = np.hstack((self.synthetic_labels, synthetic_y))
            X_augmented = np.vstack((self.synthetic_sample, self.dataset.data.loc[self.dataset.train_idx].values))
            self.data_transformer.fit(X_augmented)
        
        self.X = self.data_transformer.transform(self.dataset.data.loc[self.dataset.train_idx])
        
        X_train = self.X.loc[self.labels.index].values
        y_train = self.labels.values
    
        if len(self.synthetic_sample) > 0 :
            transformed_synthetic_sample = self.data_transformer.transform(self.synthetic_sample)
            X_train = np.vstack((X_train, transformed_synthetic_sample))
            y_train = np.hstack((y_train, self.synthetic_labels))
   
        X_train = pd.DataFrame(X_train, columns=self.X.columns)
        X_train[list(self.ineffective_features)] = 0
        self.model.fit(X_train.values, y_train)

        if type(self.model) == type(self.explainer):
            self.explainer = self.model
    
    def interact_with_user(self, item: Union[pd.Series, pd.DataFrame], explanation: dict[str, list], reasoning_level='global'):
    
        user_feedback = get_feedback_from_synthetic_user(self.user, item, explanation, reasoning_level)

        return user_feedback

    def evaluate_feedback(self, sample: pd.DataFrame, explanations: list[dict[str,list]], feedbacks: list[dict[str,list]], synthetic_coef=20):
            
        def generate_synthetic_samples(item, feature_labels: pd.Series, already_created:bool, synthetic_coef: int, is_correct: bool):
   
            new_X = []
            new_Y = []

            for feature, effect in feature_labels.items():

                if int(effect) == 2: # ineffective feature
                    self.ineffective_features.add(feature)
                    continue
                
                if feature in self.ineffective_features:
                    self.ineffective_features.discard(feature)

                if feature in already_created:
                    continue

                if is_correct:
                    if feature in self.positive_features | self.negative_features:
                        continue

                if int(effect) == 1:
                    self.positive_features.add(feature)
                elif int(effect) == 0:
                    self.negative_features.add(feature)
                else:
                    raise Exception(f"Synthetic sample can be generated only for features with positive(1) or negative(0) labels, but current feature label is {effect}")

                new_x = pd.Series(0.0, index=item.index)
                new_x[feature] = item[feature]
                new_y = int(effect)
                    
                for _ in range(synthetic_coef):
                    new_X.append(new_x.values.reshape(1,-1))
                    new_Y.append(new_y)

                already_created.append(feature)

            if new_X:
                new_X = np.concatenate(new_X, axis=0)
          
            return new_X, new_Y

        new_X = []
        new_y = []
        already_created = []

        for idx, _ in sample.iterrows():
            item = self.dataset.data.loc[idx]
            
            feedback = feedbacks[idx]["user_reasoning"]
            explanation = explanations[idx]["preference_factors"]
            correction = feedback == explanation
            feedback_for_correct = feedback[correction]
            feedback_for_incorrect = feedback[~correction]

            if len(feedback_for_correct) > 0:
                new_X_, new_y_ = generate_synthetic_samples(item, feedback_for_correct, already_created, synthetic_coef, True)   
                if len(new_X_) > 0:
                    new_X.append(new_X_)
                    new_y.extend(new_y_)
            
            if len(feedback_for_incorrect) > 0:
                new_X_, new_y_ = generate_synthetic_samples(item, feedback_for_incorrect, already_created, synthetic_coef, False)   
                if len(new_X_) > 0:
                    new_X.append(new_X_)
                    new_y.extend(new_y_)

        if len(new_X) > 0:      
            new_X = np.concatenate(new_X, axis=0)
            new_y = np.array(new_y)
            
        return new_X, new_y

    def generate_explanation(self, idx: Union[int, float], new_sample: pd.DataFrame, *args, **kwargs):
        train_idx = list(set(self.labels.index) - set(new_sample.index))
        
        item = self.dataset.data.loc[idx]
        item_features = list(item[item != 0].index)
        x = self.X.loc[idx]

        pred_label = self.model.predict(x.values.reshape(1,-1))[0]

        train_df = self.dataset.data.loc[train_idx]
        all_feature_weights = self.explainer.explain(self.X.loc[train_idx].values, self.labels[train_idx].values, train_df)
        item_feature_weights = pd.Series(all_feature_weights, self.dataset.data.columns)[item_features]
        important_feature_weights = item_feature_weights[item_feature_weights.abs() > 1e-8] 
        
        preference_factors = (important_feature_weights > 0).astype(int)
        
        ineffective_features = item_feature_weights.drop(important_feature_weights.index).index
        
        for feature_name in ineffective_features:
            preference_factors[feature_name] =  2

        explanation = {'recommendation_label': float(pred_label), "preference_factors":preference_factors}

        return explanation
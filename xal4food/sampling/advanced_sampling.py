from sampling.uncertainty_sampling import UncertaintyBasedSampling
import numpy as np
import pandas as pd
from copy import deepcopy

class ClusteredUncertaintySampling():

    def __init__(self, diversity_sampling, uncertainty_sampling, fit_clusters):
        self.diversity_sampling = deepcopy(diversity_sampling)
        self.uncertainty_sampling = deepcopy(uncertainty_sampling)
        self.fit_clusters = fit_clusters
    
    def __str__(self) -> str:
        return f"ClusteredUncertaintySampling(clustering={type(self.diversity_sampling.model).__name__}, uncertainty={self.uncertainty_sampling.uncertainty_func.__name__})"
    
    def get_sample(self, data, size,  model, *args, **kwargs) -> pd.DataFrame:

        assert data.shape[0] >= size

        if self.fit_clusters:
            self.diversity_sampling.fit(data)
        
        clusters = self.diversity_sampling.get_clusters(data)
        
        n_clusters = len(clusters)
        if size < n_clusters:
            subsize, remainder = 1, 0
        else:
            subsize, remainder = divmod(size, n_clusters)

        sample = pd.DataFrame([])
        uncertainty_scores = pd.Series([], dtype=float)
        
        for cluster in clusters: 
    
            subsample, uncertainty = self.uncertainty_sampling.get_sample(cluster, subsize, model, return_uncertainty=True)
           
            sample = pd.concat([sample, subsample], axis=0)
            uncertainty_scores = pd.concat([uncertainty_scores, uncertainty], axis=0)

        if remainder > 0:
            subsample = data.drop(sample.index).sample(remainder)
            
            pdist_vals = model.predict_proba(subsample.values)
            uncertainty = pd.Series(self.uncertainty_sampling.uncertainty_func(pdist_vals), index=subsample.index)
            
            sample = pd.concat([sample, subsample], axis=0)
            uncertainty_scores = pd.concat([uncertainty_scores, uncertainty], axis=0)
      
        if size < n_clusters:
   
            most_uncertain_idx = uncertainty_scores.sort_values(ascending=False)[:size].index
            sample = sample.loc[most_uncertain_idx]
  
        assert len(sample) == size
        
        return sample


class MostUncertainCluster():

    def __init__(self, diversity_sampling, uncertainty_metric, fit_clusters):
        self.diversity_sampling = deepcopy(diversity_sampling)
        self.uncertainty_func = getattr(UncertaintyBasedSampling, uncertainty_metric)
        self.n_clusters = self.diversity_sampling.model.n_clusters
        self.fit_clusters = fit_clusters 

    def __str__(self) -> str:
        return f"MostUncertainCluster(clustering={type(self.diversity_sampling.model).__name__}, uncertainty_metric={self.uncertainty_func.__name__})"

    def get_sample(self, data, size, model, *args, **kwargs) -> pd.DataFrame:

        assert data.shape[0] >= size

        if self.fit_clusters:
            self.diversity_sampling.fit(data)
        
        clusters = self.diversity_sampling.get_clusters(data)
        sample_cluster = None
        highest_uncert = 0

        for cluster in clusters:
            if len(cluster) < size:
                continue

            pdist_vals = model.predict_proba(cluster)
            avg_uncert_score = max(0, np.mean(self.uncertainty_func(pdist_vals)))
       
            if avg_uncert_score > highest_uncert:
                highest_uncert = avg_uncert_score
                sample_cluster = cluster
            
        pdist_vals = model.predict_proba(sample_cluster)
        indices = np.argsort(self.uncertainty_func(pdist_vals))[-size:]
        sample = sample_cluster.iloc[indices]
        
        return sample

class UncertaintyClusteredSampling():

    def __init__(self, diversity_sampling, uncertainty_sampling, fit_clusters, data_proportion, random_state):
        self.diversity_sampling = deepcopy(diversity_sampling)
        self.uncertainty_sampling = deepcopy(uncertainty_sampling)
        self.fit_clusters = fit_clusters
        self.data_proportion = data_proportion
        self.random_state = random_state
    
    def __str__(self) -> str:
        return f"UncertaintyClusteredSampling(clustering={type(self.diversity_sampling.model).__name__}, uncertainty={self.uncertainty_sampling.uncertainty_func.__name__})"
    
    def get_sample(self, data, size, model, *args, **kwargs) -> pd.DataFrame:
        
        assert data.shape[0] >= size

        large_sample_size = int(len(data)*self.data_proportion) 
        large_sample, _ = self.uncertainty_sampling.get_sample(data, large_sample_size, model, return_uncertainty=True)
        
        if self.fit_clusters:
            self.diversity_sampling.fit(large_sample)
        
        sample = self.diversity_sampling.get_sample(large_sample, size)
        
        return sample
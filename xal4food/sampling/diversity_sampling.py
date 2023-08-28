import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from time import time
from sklearn.metrics import pairwise_distances

class ClusterBasedSampling():
    fitted_data = None
    model = None
    train_clusters = None
    
    def __init__(self, clustering_model, embedder=None, metric='jaccard', outlier_proportion=0 , random_state=42):
        self.model = clustering_model
        self.embedder = embedder
        self.metric = metric
        self.random_state = random_state
        self.outlier_proportion = outlier_proportion
    
    def __str__(self) -> str:
        return f"ClusterBasedSampling(clustering={self.model})"

    def fit(self, data):
        
        if self.embedder:
            data_ = self.embedder.transform(data)
        else:
            data_  = data
        
        print(f">> Clustering data {data_.shape} using {self.model}...")    
        st = time()
        self.model.fit(data_)
        et = time()
        print(f">> Clustering completed in {round(et-st, 1)} secs.")
       
    def get_clusters(self, data: pd.DataFrame):
        if self.embedder:
            data_ = self.embedder.transform(data)
        else:
            data_  = data
        try:
            cluster_labels = self.model.predict(data_)
            clusters =  [data[cluster_labels == c_id] for c_id in np.sort(np.unique(cluster_labels))]  
        except:
            dist_mt = pairwise_distances(np.array(data_), np.array(self.model.cluster_centers_), metric='euclidean')
            cluster_labels = [np.argmin(dist_arr) for dist_arr in dist_mt]
            clusters =  [data[cluster_labels == c_id] for c_id in np.sort(np.unique(cluster_labels))]  
        return clusters
            
    def get_sample(self, data: pd.DataFrame, size, *args, **kwargs) -> pd.DataFrame:
        if self.embedder:
            data_ = pd.DataFrame(self.embedder.transform(data), index = data.index)
        else:
            data_  = data
        
        cluster_labels = self.model.predict(data_)

        centroid_idx = []
        outlier_idx = []
        cluster_sample_idx = []
        cluster_sizes = {}

        for c_id in np.sort(np.unique(cluster_labels)):
            cluster =  data_[cluster_labels == c_id]

            n_sample = int(size * cluster.shape[0]/data_.shape[0])
            n_outliers = int(n_sample * self.outlier_proportion) 

            c_idx = self.nearest_to_point(cluster, self.model.cluster_centers_[c_id], self.metric)
            o_idx = self.farthest_point(cluster, self.model.cluster_centers_[c_id], self.metric, n_outliers) if n_outliers > 0 else []
 
            cluster_ = cluster.drop(np.unique(c_idx + o_idx))
            s_idx = cluster_.sample(n_sample).index.tolist() if cluster_.shape[0] > 0 else []
            
            centroid_idx += c_idx
            outlier_idx += o_idx
            cluster_sample_idx += s_idx
            cluster_sizes[c_idx[0]] = cluster.shape[0]

        if size < len(centroid_idx):
            centroid_idx = pd.Series(cluster_sizes)[centroid_idx].sort_values(ascending=False).iloc[:size].index.tolist()
            chosen_outlier_idx = []
        else:
            n_outliers = min(size - len(centroid_idx), int(size * self.outlier_proportion))
            chosen_outlier_idx = np.random.choice(outlier_idx, n_outliers, replace=False).tolist()
            
        remaining_sample_size = size - len(centroid_idx) - len(chosen_outlier_idx)
        chosen_cluster_sample_idx = np.random.choice(cluster_sample_idx, remaining_sample_size, replace=False).tolist()
        
        #When n_clusters is high, some single observations can be a cluster alone. 
        #A cluster of a single instance yields c_idx = o_idx. Therefore, np.unique should be used below.
        sample_idx = np.unique(centroid_idx + chosen_outlier_idx + chosen_cluster_sample_idx) 
        sample = data.loc[sample_idx].copy()
        
        print(f">> A diverse sample at size of {sample.shape[0]} given.")
        return sample
    
    def nearest_to_point(self, data:pd.DataFrame, point:np.array, metric, n_closest=1):

        dist = pairwise_distances(X=data.values, Y=point.reshape(1,-1), metric=metric).ravel()
        indices = np.argsort(dist)[:n_closest]
 
        return data.iloc[indices].index.tolist()

    def farthest_point(self, data:pd.DataFrame, point:np.array, metric, n_farthest=1):

        dist = pairwise_distances(X=data.values, Y=point.reshape(1,-1), metric=metric).ravel()
        indices = np.argsort(dist)[-n_farthest:]
        
        return data.iloc[indices].index.tolist()
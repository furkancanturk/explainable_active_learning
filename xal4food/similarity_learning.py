import pandas as pd
import numpy as np
from time import time
from typing import Union
from data_builder import *
from sklearn.metrics import pairwise_distances
from sklearn.semi_supervised import LabelSpreading
from sampling.uncertainty_sampling import UncertaintyBasedSampling


def calculate_similarity(dataset: Union[FoodDataset, FoodDataset2], idx1: list, idx2: list, distance_metric = 'jaccard', alpha = 1): 
    print(f">> Similarity matrix {(len(idx1), len(idx1))} computing...")

    # the below code culd be utilized if continuous features are included in the dataset
    #alpha = 1 / (dataset.data[dataset.used_cat_features].sum(axis=1).mean() / len(dataset.used_cont_features))

    alpha = 1

    st = time()
    cat_dist = pairwise_distances(X = dataset.data.loc[idx1, dataset.used_cat_features].values,
                                  Y = dataset.data.loc[idx2, dataset.used_cat_features].values, 
                                  metric = distance_metric, n_jobs=-1)

    # the below code culd be utilized if continuous features are included in the dataset
    # cont_dist = pairwise_distances(X = dataset.data.loc[idx1, dataset.used_cont_features].values,
    #                                 Y = dataset.data.loc[idx2, dataset.used_cont_features].values, 
    #                                 metric = 'cosine', n_jobs=-1)

    distance_mt = alpha * cat_dist # + (1-alpha) * cont_dist
    similarity_df = pd.DataFrame(1-distance_mt, index=idx1, columns=idx2).sort_index(axis=0).sort_index(axis=1)
    
    et = time()

    print(f">> Similarity matrix computed in {et-st} secs.")   

    return similarity_df
    

def augment_labeled_data(similarity_df: pd.DataFrame, unlabeled_data: pd.DataFrame, labeled_sample:pd.DataFrame, labels: pd.Series, pseudo_labeled_idx: list, prior_prob=0.5):
    
    st = time()

    true_labeled_idx = list(set(labels.index) - set(pseudo_labeled_idx))
    true_labels = labels[true_labeled_idx]

    if true_labels.mean() in [0.0, 1.0]:
        print("All true labels are ", true_labels.mean(),  "(Data augmentation cannot be applied.)")
        return labels, pseudo_labeled_idx

    ls, pseudo_labels = label_spreading(unlabeled_data, labeled_sample, labels,\
                                        pseudo_labeled_idx, similarity_df,
                                        prior_prob)
    
    new_pseudo_labeled_idx = list(set(pseudo_labels.index) - set(true_labeled_idx))
    new_pseudo_labels = pseudo_labels[new_pseudo_labeled_idx]

    new_labels = pd.concat([labels[true_labeled_idx],
                            new_pseudo_labels]).sort_index()
 
    print("Augmented labels mean:", round(new_labels.mean(), 3), "| Data Augmentation Time:", round(time()-st, 2) )

    return new_labels, new_pseudo_labeled_idx

def label_spreading(unlabeled_data: pd.DataFrame, labeled_sample: pd.DataFrame, labels: pd.Series, pseudo_labeled_idx: list, similarity_df: pd.DataFrame, prior_prob=0.5):
    
    def kernel(idx1: list, idx2: list):
        distance_matrix = 1 - similarity_df.loc[idx1[:, 0], idx2[:, 0]].values
        return np.exp( -1 * distance_matrix * 2 / distance_matrix.std())
    
    common_idx = set(unlabeled_data.index) & set(labeled_sample.index)
    all_data = pd.concat((unlabeled_data.drop(common_idx), labeled_sample), axis=0)
    data_idx = np.array(list(all_data.index)).reshape(-1,1)
    label_series = pd.Series(-1, index=all_data.index)

    label_series[labels.index] = labels
    label_series[pseudo_labeled_idx] = -1

    ls = LabelSpreading(kernel=kernel, alpha=0.25, max_iter=200, n_jobs=-1)
    ls.fit(data_idx, label_series)
    
    pred = np.where(ls.label_distributions_[:, 1] > prior_prob, 1, 0)
    uncertanities = UncertaintyBasedSampling.entropy(ls.label_distributions_)
    mask = np.max(ls.label_distributions_,axis=1) > uncertanities
    pseudo_labeled_data = all_data[mask]
    pseudo_labels = pd.Series(pred[mask], index=pseudo_labeled_data.index)

    return ls, pseudo_labels


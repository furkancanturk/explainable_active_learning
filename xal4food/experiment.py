from sampling.diversity_sampling import ClusterBasedSampling
from sampling.uncertainty_sampling import UncertaintyBasedSampling
from sampling.advanced_sampling import *
from sampling.random_sampling import RandomSampling
from active_learning import ActiveLearning
from similarity_learning import *
from linear_preference_model import *
from oracle.user import *
from data_builder import *

from collections import OrderedDict
import pickle
import os
import random

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn_extra.cluster import KMedoids

import warnings
warnings.filterwarnings("ignore")

uncertainty_metrics = UncertaintyBasedSampling.metric_names

dir = os.path.abspath(os.path.dirname(__file__))
data_dir = dir+"/data/"
oracle_dir = dir+"/oracle/"
log_dir = dir+"/log/"



def experiment(kargs):

    dataset_code, test_size, min_feature_freq, diversity_sampling, diverse_n_clusters, \
    diverse_data_size, init_sample_size, augment_init_sample, augment_batches, refit_clusters, \
    use_user_feedback, u_metric, n_steps, target_metric, \
    model_params, random_state, user_names, strategy_names = kargs.values()

    #for reproducability
    random.seed(random_state)
    np.random.seed(random_state) 

    #----------------------------------------------------------------------------------------------------------------
    # STEP 0: DATA PREPROCESSING
    #----------------------------------------------------------------------------------------------------------------
    print(">>>>> STEP-0")

    data_features = ['ingredients']
    food_dt_obj_file = f"fooddataset{dataset_code}_{random_state}_{test_size}_{diverse_data_size}_{min_feature_freq}_{diversity_sampling}.pkl"
    dataset = get_dataset(dataset_code, random_state, test_size, min_feature_freq, data_features, food_dt_obj_file)

    embedder = None#TruncatedSVD(n_components=200, random_state=random_state)
    data_transformer = DataTransformer(test_size, random_state)
    dataset.split()
    dataset.preprocess(data_transformer=data_transformer)

    # if data_transformer.embedder is None and not embedder is None:
    #     embedder.fit(dataset.X_train)

    init_unlabeled_data = dataset.data.loc[dataset.train_idx]
    input_size = init_unlabeled_data.shape[1]

    #----------------------------------------------------------------------------------------------------------------
    # STEP 1: DIVERSE SET GENERATION
    #----------------------------------------------------------------------------------------------------------------
    print(">>>>> STEP-1")

    if diverse_data_size == 'all':
        diverse_data_size = init_unlabeled_data.shape[0]

    if dataset.similarity_df.shape[0] != diverse_data_size:
        
        if diversity_sampling == 'cluster_based':
            save_file = log_dir+f"KMedoidsSampling_target_{dataset_code}_{min_feature_freq}_{diverse_n_clusters}.pkl"
        else:
            save_file = log_dir+f"Random_target_{dataset_code}_{random_state}_{min_feature_freq}_{diverse_n_clusters}.pkl"
        
        if not os.path.exists(save_file):
            if diversity_sampling == 'cluster_based':
                clustering1 = KMedoids(n_clusters=diverse_n_clusters, metric='jaccard', init='k-medoids++', random_state=42, max_iter=100)
                diversity_strategy_ = ClusterBasedSampling(clustering1, embedder, outlier_proportion=0, random_state=random_state)
            else:
                diversity_strategy_ = RandomSampling(random_state)
            diversity_strategy_.fit(init_unlabeled_data.values)
            
            with open(save_file, 'wb') as f:
                pickle.dump(diversity_strategy_, f)
            
        else:
            with open(save_file, 'rb') as f:
                diversity_strategy_ = pickle.load(f)

        init_target_data = diversity_strategy_.get_sample(init_unlabeled_data, diverse_data_size)

        dataset.similarity_df = calculate_similarity(dataset, init_target_data.index, init_target_data.index, 'jaccard')
        
        with open(data_dir+food_dt_obj_file, 'wb') as f:
            pickle.dump(dataset, f)

    else:
        init_target_data = init_unlabeled_data.loc[dataset.similarity_df.index]

    #----------------------------------------------------------------------------------------------------------------
    # STEP 2: SELECTING A DIVERSE SAMPLE
    #----------------------------------------------------------------------------------------------------------------
    print(">>>>> STEP-2")
    clustering2 = KMedoids(n_clusters=init_sample_size, metric='jaccard', init='k-medoids++', random_state=random_state)

    if diversity_sampling == 'cluster_based':
        save_file = f"KMedoidsSampling_init_{dataset_code}_{random_state}_{min_feature_freq}_{init_sample_size}.pkl"
    else:
        save_file = f"Random_init_{dataset_code}_{random_state}_{min_feature_freq}_{init_sample_size}.pkl"

    if not os.path.exists(log_dir+save_file):
        
        if diversity_sampling == 'cluster_based':
            diversity_strategy = ClusterBasedSampling(clustering2, embedder, outlier_proportion=0, random_state=random_state)
        else:
            diversity_strategy = RandomSampling(random_state)
        
        diversity_strategy.fit(init_target_data)
        with open(log_dir+save_file, 'wb') as f:
            pickle.dump(diversity_strategy, f)

    else:
        print("EXITS:", save_file)
        with open(log_dir+save_file, 'rb') as f:
            diversity_strategy = pickle.load(f)

    init_diverse_sample_file = "init_diverse_sample_" + food_dt_obj_file
    
    if not os.path.exists(log_dir+init_diverse_sample_file):            
        init_diverse_sample = diversity_strategy.get_sample(init_target_data, init_sample_size, similarity_df=dataset.similarity_df)
        with open(log_dir+init_diverse_sample_file, 'wb') as f:
            pickle.dump(init_diverse_sample, f)
    
    else:
        print("EXISTS:", init_diverse_sample_file)
        with open(log_dir+init_diverse_sample_file, 'rb') as f:
            init_diverse_sample = pickle.load(f)
            f.close()

    #----------------------------------------------------------------------------------------------------------------
    # STEP 2: LABELING THE DIVERSE SAMPLE BY USER
    #----------------------------------------------------------------------------------------------------------------
  
    for user_name in user_names:
        
        with open(oracle_dir+f"synthetic_users/{user_name}_{dataset_code}_{random_state}_{min_feature_freq}.pkl", "rb") as f:
            user = pickle.load(f)   
            user.set_labels(dataset.data)
            user_pos = round(user.food_labels.mean(),3)
            print("User preference positivity:", user_pos)

        init_labels = user.get_labels(init_diverse_sample.index)
        init_labeled_sample = init_target_data.loc[init_labels.index]
        prior_prob = init_labels.mean()
        
        if prior_prob in [0.0, 1.0]:
            extra_sample = init_target_data.sample(init_diverse_sample.shape[0])
            extra_init_labels = user.get_labels(extra_sample.index)
            init_labels = pd.concat((init_labels, extra_init_labels))
        init_labeled_sample = init_target_data.loc[init_labels.index]
        prior_prob = init_labels.mean()
        print("prior_prob of init_labels:", prior_prob)
        target_data = init_target_data.drop(init_diverse_sample.index)
        unlabeled_data = init_unlabeled_data.drop(init_diverse_sample.index)

        #----------------------------------------------------------------------------------------------------------------
        # STEP 3.1: LABELED DATA AUGMENTATION
        #----------------------------------------------------------------------------------------------------------------
        if augment_init_sample:
            init_target_similarity_df = dataset.similarity_df.loc[init_target_data.index, init_target_data.index]
            labels, pseudo_labeled_idx = augment_labeled_data(
                                                            init_target_similarity_df, 
                                                            target_data,
                                                            init_labeled_sample,
                                                            init_labels,
                                                            [],
                                                            prior_prob)
        else:
            labels = init_labels
            pseudo_labeled_idx = []
            pseudo_cm0 = None

        for model_name, params in model_params.items():
            
            print(">>>>> STEP-3.1", user_name, model_name)

            model = LinearClassifier(**params) 
            
            true0 = user.get_labels(unlabeled_data.index)
            X_train = dataset.data_transformer.transform(init_labeled_sample.values)
            X_test = dataset.data_transformer.transform(unlabeled_data.values)
            model.fit(X_train, init_labels.values,  None, None, target_metric)
    
            pred0 = model.predict(X_test)
            s0 = target_metric(true0, pred0)
            mcc0 = matthews_corrcoef(true0, pred0)
            
            #----------------------------------------------------------------------------------------------------------------
            # STEP 4: ACTIVE LEARNING WITH EXPLANATION-FEEDBACK INTERACTIONS
            #----------------------------------------------------------------------------------------------------------------
        
            uncertainty_strategy = UncertaintyBasedSampling(u_metric) 
            strategies = {
                        'Uncertainty': uncertainty_strategy,
                        'MostUncertainCluster': MostUncertainCluster(diversity_strategy, u_metric, refit_clusters), 
                        'ClusteredUncertainty': ClusteredUncertaintySampling(diversity_strategy, uncertainty_strategy, refit_clusters),
                        'UncertaintyClustered': UncertaintyClusteredSampling(diversity_strategy, uncertainty_strategy, refit_clusters, 0.25, random_state),
                        'Random': RandomSampling(random_state)
                        }
    
            for strategy_name in strategy_names:
                print(">>>>> STEP-4", random_state, user_name, model_name, strategy_name)
                
                sampling_strategy = strategies[strategy_name]

                for iuf in use_user_feedback:
                    
                    file_name = f"{random_state}_{init_sample_size}_{diversity_sampling}_{user_name}_{model_name}_{strategy_name}_{iuf}.pkl"
                    
                    if os.path.exists(log_dir+file_name):
                        print("EXISTS: "+ file_name)
                        continue
                    
                    act_l = ActiveLearning(model, 
                                            user, 
                                            sampling_strategy, 
                                            dataset, 
                                            labels,
                                            pseudo_labeled_idx,
                                            unlabeled_data.index.tolist(),
                                            target_data.index.tolist(),
                                            prior_prob,
                                            target_metric,
                                            use_user_feedback=iuf,
                                            explainer=model)

                    act_l.user_positivity = user_pos
                    act_l.prior_prob = prior_prob
                    act_l.scores = [[s0, mcc0]]

                    for i in range(n_steps):
                        candidates = act_l.target_idx
                        batch_size = 1 if i < 5 else 5
                        act_l.step(batch_size, candidates, augment_batches, verbose=True)

                    #deleting not used elements for the result analysis
                    to_del = ['user', 'dataset', 'X', 'synthetic_sample', 'synthetic_labels'] 
                    for attr in to_del:
                        setattr(act_l, attr, None)
    
                    with open(log_dir+file_name, 'wb') as f:
                        pickle.dump(act_l, f)
                        f.close()


if __name__ == '__main__':

    strategy_lst1 = ['ClusteredUncertainty', 'Uncertainty', 'MostUncertainCluster', 'UncertaintyClustered']
    strategy_lst2 = ["Random"]

    user_names = ['user1', 'user2', 'user3',  'sportive_user1', 'unhealthy_user1', 'vegan_user2', 'elder_user2', 'sportive_user3', 'unhealthy_user3', 'random']

    params = {"dataset_code": 1,
            "test_size": 0.0,
            "min_feature_freq": 0.001,
            "diversity_sampling": "cluster_based", #change it to "random" if strategy is "Random"
            "diverse_n_clusters": 100, 
            "diverse_data_size": 5000,
            "init_sample_size": 30,
            "augment_init_sample": True,
            "augment_batches": True,
            "refit_clusters": False,
            "use_user_feedback": [True, False],
            "u_metric":'least_confidence',
            "n_steps": 50,
            "target_metric": f1_score,
            "model_params" : {'lr':{ 'C':0.1,  'n_jobs':4, 'max_iter':1000}}
            }
    
    params_lst = []

    for random_state in range(5):      
        params["random_state"] = int(random_state)
        for u in user_names:
            p = params.copy()
            list_of_tuples = [(key, val) for key, val in p.items()]
            p = OrderedDict(list_of_tuples)

            p['user_names'] = [u]
            p['strategy_names'] = lst1
            params_lst.append(p)  
    
    with Pool(5) as p:
        p.map(experiment, params_lst)
import os
import ast
import pickle
import itertools
from time import time
from pathlib import Path
from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

import nltk    
try:
    nltk.data.find('corpora/stopwords.zip')
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
except:
    nltk.download("stopwords", nltk.data.path[0])
    nltk.download("wordnet", nltk.data.path[0])
    nltk.download("omw-1.4", nltk.data.path[0])
    nltk.download('averaged_perceptron_tagger', nltk.data.path[0])

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag_sents
from nltk.corpus import stopwords

dir = os.path.abspath(os.path.dirname(__file__))
data_dir = dir+"/data/"

def get_dataset(dataset_code: int, random_state: int, test_size: float, min_cat_feature_freq: float, features_to_use: list, file_name: str, 
                drop_rare_foods = True, base = False):
    
    if not os.path.exists(data_dir+file_name) or base:

        base_dt_file_name = f"fooddataset{dataset_code}_{min_cat_feature_freq}_base.pkl"
        print(data_dir+base_dt_file_name)
        if not os.path.exists(data_dir+base_dt_file_name):
            
            print(f">> Generating Base FoodDataset{dataset_code}...")
            
            if dataset_code == 1:
                dataset = FoodDataset(features_to_use, test_size, random_state)
            elif dataset_code == 2:
                dataset = FoodDataset2(features_to_use, test_size, random_state)
            elif dataset_code == 3:
                dataset = FoodDataset3(features_to_use, test_size, random_state)
            else:
                raise Exception("Invalid dataset_code:{dataset_code}. It must be one of [1,2,3].")
            
            if min_cat_feature_freq > 0:
                feature_freq = dataset.data[dataset.used_cat_features].mean()
                rare_features = feature_freq[feature_freq < min_cat_feature_freq].index.tolist()
                dataset.data.drop(rare_features, axis=1, inplace=True)
                print(f">> {len(rare_features)} rare features are dropped.")

                # Dropping foods if it has no feature after dropping rare features
                dataset.data = dataset.data[dataset.data.sum(axis=1) > 0]
                dataset.ingredients_df = dataset.ingredients_df.loc[dataset.data.index]
                dataset.used_cat_features = dataset.data.columns.tolist()
            
            if drop_rare_foods:
                rare_idx = get_rare_food_idx(dataset)
                dataset.data.drop(rare_idx, inplace=True)
                dataset.ingredients_df.drop(rare_idx, inplace=True)
                print(f">> {len(rare_idx)} rare instances are dropped.")

            with open(data_dir+base_dt_file_name, 'wb') as f:
                pickle.dump(dataset, f)

        else:
            print(f">>Base FoodDataset{dataset_code} found in "+ data_dir+base_dt_file_name)
            with open(data_dir+base_dt_file_name, 'rb') as f:
                dataset = pickle.load(f)
            
    else:
        print(f">> FoodDataset{dataset_code} with similarity matrix found in "+data_dir+file_name)
        with open(data_dir+file_name, 'rb') as f:
            dataset = pickle.load(f)

    return dataset

def concat_all_features_as_list(dataset):
    log_nutrients = (dataset.data[dataset.nutrient_names] + 0.001).apply(np.log)
    q_vals = [20, 40, 60, 80, 100]
    quantile_df = pd.DataFrame(np.array([np.percentile(log_nutrients, q, axis=0) for q in q_vals]), index=q_vals, columns=dataset.nutrient_names)

    cat_names = ['very low', 'low', 'normal', 'high', 'too']
    cat_nutrient = pd.DataFrame()
    for nutrient in dataset.nutrient_names:
        cat_nutrient[nutrient] = pd.cut(log_nutrients[nutrient], [-9999]+list(quantile_df[nutrient]), labels=[cat + " " + nutrient for cat in cat_names]).astype(str)#.replace({'nan':'no_'+nutrient})

    dataset.data['nutrient_levels'] = cat_nutrient.apply(list, axis=1)
    concat_feature_lst = dataset.data.apply(lambda row: [row['tag']] + row['nutrient_levels'] + list(row['ingredients']), axis=1)
    return concat_feature_lst


def get_rare_food_idx(dataset):
    ingredient_freq = dataset.ingredients_df.mean()
    food_ingredient_freq = dataset.ingredients_df.astype(int) * ingredient_freq
    food_freq = list(map(lambda row: np.mean(list(filter(None, row))), food_ingredient_freq.values))
    food_freq = pd.Series(food_freq, dataset.data.index)
    f_mean, f_std = food_freq.describe()[['mean', 'std']]
    rare_food_idx = list(food_freq[food_freq<f_mean - 2*f_std].index)
    return rare_food_idx

class FoodDataset(): # FoodKG dataset

    nutrient_names = ['poly_fat', 'mono_fat', 'sat_fat', 'carbs', 'protein', 'cals']
    feature_names = ['tag'] + nutrient_names
    categorical_features = ['tag', 'ingredients']
    continuous_features = list(set(feature_names) - set(categorical_features))
    meta_features = ["dish_name"]
    data = pd.DataFrame()
    similarity_df = pd.DataFrame()
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    train_idx = []
    test_idx = []
    
    def __init__(self, features_to_use, test_size = 0.0, random_state = 42):
        self.random_state = random_state
        self.test_size = test_size
        
        pkl_file_name = 'foodkg_dfs_dict.pkl'
        if os.path.exists(data_dir+pkl_file_name):
            with open(data_dir+pkl_file_name, 'rb') as f:
                dfs = pickle.load(f)
        else:
            raw = self.load_and_parse_raw_data() # taking ~90 secs
            ing_df = self.encode_ingredients(raw.ingredients) #taking ~360 secs
            raw, ing_df = self.drop(raw, ing_df)
            dfs = dict(raw=raw, ingredients_df=ing_df)
            
            with open(data_dir+pkl_file_name, 'wb') as f:    
                pickle.dump(dfs, f)
        
        self.raw = dfs['raw']
        self.ingredients_df = dfs['ingredients_df'].astype(int)
        # updating the ingredients column of the raw dataframe since some similar ingredients are discarded
        #self.raw.ingredients = self.ingredients_df.apply(lambda row: set(row.iloc[np.nonzero(row.values)].index), axis=1)  
        self.all_ingredients = sorted(list(self.ingredients_df.columns))

        self.set_data_features(features_to_use)

    def set_data_features(self, feature_list):
        
        self.used_cont_features = []
        self.used_cat_features = []
       
        if "nutritions" in feature_list:
            self.used_cont_features += self.nutrient_names

        self.data = pd.DataFrame(self.raw[self.used_cont_features], index=self.raw.index)

        if 'tag' in feature_list:
            tag_df = pd.get_dummies(self.raw['tag'])
            self.data = pd.concat((self.data, tag_df), axis=1)
            self.used_cat_features += tag_df.columns.tolist()

        if 'ingredients' in feature_list:
            self.data = pd.concat((self.data, self.ingredients_df), axis=1)
            self.used_cat_features += self.ingredients_df.columns.tolist()

    def load_and_parse_raw_data(self):
        
        file_name = 'recipe_kg.json'
        with open(dir+f"/data/{file_name}", 'r') as f:
            recipes = f.readlines()
        recipes = list(map(ast.literal_eval, recipes))

        food_lst = []

        for j in range(len(recipes)):
            tagged_dishes = list(recipes[j].values())[0]
            tag = tagged_dishes['name'][0]
            dishes = tagged_dishes['neighbors']['tagged_dishes']

            for dish_dict in dishes:
                dish = list(dish_dict.values())[0]
                name = dish['name'][0]
                p_fat = float(dish['neighbors']['polyunsaturated fat'][0])
                m_fat = float(dish['neighbors']['monounsaturated fat'][0])
                s_fat = float(dish['neighbors']['saturated fat'][0])
                carb = float(dish['neighbors']['carbohydrates'][0])
                prot = float(dish['neighbors']['protein'][0])
                cals = float(dish['neighbors']['calories'][0])

                ingredients = dish['neighbors']['contains_ingredients']
                ingredients_lst = []
                for i in ingredients:
                    ingredient = list(i.values())[0]['name'][0]
                    ingredients_lst.append(ingredient)
                food_lst.append([tag, name, p_fat, m_fat, s_fat, carb, prot, cals, sorted(list(set(ingredients_lst)))])

        #converting food list to dataframe
        df = pd.DataFrame(food_lst, columns=['tag', 'dish_name', 'poly_fat', 'mono_fat', 'sat_fat', 'carbs', 'protein', 'cals', 'ingredients'])

        #dropping foods which all nutrient values are zero
        no_nutrient = df[df[self.nutrient_names].sum(axis=1) == 0]
        df.drop(no_nutrient.index, inplace=True)
        df.reset_index(drop = True, inplace=True)
        
        #same foods with multiple tags exists which causes duplicate rows in the dataset
        df.drop_duplicates('dish_name', ignore_index=True, inplace=True)

        return df

    def encode_ingredients(self, ingredients):
    
        wnl = WordNetLemmatizer()
        mlb = MultiLabelBinarizer()
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

        splitted_ings = ingredients.apply(lambda row: [[wnl.lemmatize(word.lower()) for word in words.split()] for words in row if words])
        concat_ings =  splitted_ings.apply(lambda row : list(itertools.chain.from_iterable(row)))

        vectorizer.fit_transform(concat_ings.apply(" ".join))
        valid_words = vectorizer.get_feature_names_out()
        valid_words = valid_words[(valid_words >= "a") & (valid_words <= "z")]
        valid_words = [x for x in valid_words if len(x) > 2]

        filtered_ings1 = splitted_ings.apply(lambda row: [[word for word in words if word in valid_words] for words in row])

        ing_word_tags = list(map(pos_tag_sents, filtered_ings1.values)) 

        noun_ings = [ [[ word_tag_tuple[0] for word_tag_tuple in tag_tuple_lst if word_tag_tuple[1] == 'NN' ]for tag_tuple_lst in ing_lst] for ing_lst in ing_word_tags]
        noun_ings = pd.Series(noun_ings, filtered_ings1.index)

        concat_noun_ings = noun_ings.apply(lambda row : list(itertools.chain.from_iterable(row)))

        encoded_ings = vectorizer.fit_transform(concat_noun_ings.apply(" ".join))

        ing_freq = pd.Series(encoded_ings.toarray().mean(axis=0), index=vectorizer.get_feature_names_out())

        filtered_ings2 = noun_ings.apply(lambda row: set([words[ing_freq.loc[words].argmax()] for words in row if words]))
        ing_df = pd.DataFrame(mlb.fit_transform(filtered_ings2),columns=mlb.classes_)

        return ing_df

    def drop(self, recipes, ing_df):
        #Dropping instances with one ingredient
        ing_df = ing_df[ing_df.sum(axis=1) > 1].sort_index()
        recipes = recipes.loc[ing_df.index].sort_index()

        #Dropping instances with negative nutritional values
        mask = recipes[self.nutrient_names].apply(lambda x: x<0, axis=1).any(axis=1)
        to_drop = recipes[mask].index
        recipes.drop(to_drop, inplace=True)
        ing_df.drop(to_drop, inplace=True)

        #Resetting indicies
        recipes.reset_index(drop=True, inplace=True)
        ing_df.reset_index(drop=True, inplace=True)
        
        #updating the ingredients column of the data dataframe since some similar ingredients are discarded
        recipes.ingredients = ing_df.apply(lambda row: set(row.iloc[np.nonzero(row.values)].index), axis=1)  

        #converting binary ing_df to boolean for less storage
        ing_df = ing_df.astype(bool)

        #removing empty columns
        empty_cols = ing_df.columns[ing_df.sum(axis=0)<1]
        ing_df.drop(empty_cols, axis=1, inplace=True)

        return recipes, ing_df

    def split(self):
        if self.test_size > 0:
            self.train_idx, self.test_idx = train_test_split(self.data.index.tolist(), test_size=self.test_size, random_state=self.random_state)
        else:
            self.train_idx = self.data.index.tolist()
            self.test_idx = []
        print(f">> Dataset splitted with test ratio of {self.test_size} into train({len(self.train_idx)}) and test({len(self.test_idx)}) datasets.")
        return self.train_idx, self.test_idx

    def preprocess(self, data=None, data_transformer=None):
        if data is None and data_transformer is None:
            raise Exception("Either data or data_transfomer must be provided.")
        
        if data_transformer:
            self.data_transformer = data_transformer
        
        if data is None:
            data = self.data

        self.data_transformer.fit(data)
        self.X_train = self.transform(self.data.loc[self.train_idx])
        if len(self.test_idx) > 0:
            self.X_test = self.transform(self.data.loc[self.test_idx])
    
    def transform(self, x):
        return self.data_transformer.transform(x)
    

class FoodDataset2(): # CulinaryDB dataset

    nutrient_names = ['calcium', 'carbohydrates', 'cholesterol', 'fat', 'fiber', 'folate', 'iron', 'magnesium', \
                      'niacin', 'potassium', 'protein', 'saturatedFat', 'sodium', 'sugars', 'thiamin', 'vitaminA', 'vitaminB6', 'vitaminC']
    feature_names = ['calories', 'caloriesFromFat', 'total_prep_time', 'ingredients', 'cuisine'] + nutrient_names
    categorical_features = ['ingredients', 'cuisine']
    continuous_features = list(set(feature_names) - set(categorical_features))
    meta_features = ["recipe_id", "recipe_name", "image_url"]
    data = pd.DataFrame()
    similarity_df = pd.DataFrame()
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    train_idx = []
    test_idx = []
    
    def __init__(self, features_to_use, test_size = 0.0, random_state = 42):
        self.random_state = random_state
        self.test_size = test_size
        
        pkl_file_name = 'culinarydb_dfs_dict.pkl'
        if os.path.exists(data_dir+pkl_file_name):
            with open(data_dir+pkl_file_name, 'rb') as f:
                dfs = pickle.load(f)
        else:
            raw = self.load_and_parse_raw_data()
            ing_df = self.encode_ingredients(raw.ingredients)
            raw, ing_df = self.drop(raw, ing_df)
            dfs = dict(raw=raw, ingredients_df=ing_df)
            
            with open(data_dir+pkl_file_name, 'wb') as f:    
                pickle.dump(dfs, f)
       
        self.raw = dfs['raw']
        self.ingredients_df = dfs['ingredients_df'].astype(int)
        # updating the ingredients column of the raw dataframe since some similar ingredients are discarded
        #self.raw.ingredients = self.ingredients_df.apply(lambda row: set(row.iloc[np.nonzero(row.values)].index), axis=1)  
        self.all_ingredients = sorted(list(self.ingredients_df.columns))

        self.set_data_features(features_to_use)

    def set_data_features(self, feature_list):

        self.used_cont_features = []
        self.used_cat_features = []
       
        if "nutritions" in feature_list:
            self.used_cont_features += self.nutrient_names

        if "calories" in feature_list:
            self.used_cont_features += ['calories', 'caloriesFromFat']

        if "total_prep_time" in feature_list:
            self.used_cont_features += ["total_prep_time"]

        self.data = pd.DataFrame(self.raw[self.used_cont_features], index=self.raw.index)

        if "cuisine" in feature_list:
            cuisine_df = pd.get_dummies(self.raw['cuisine'])
            self.data = pd.concat((self.data, cuisine_df), axis=1)
            self.used_cat_features += cuisine_df.columns.tolist()

        if 'ingredients' in feature_list:
            self.data = pd.concat((self.data, self.ingredients_df), axis=1)
            self.used_cat_features += self.ingredients_df.columns.tolist()

    def load_and_parse_raw_data(self):

        core_data = pd.read_csv(data_dir+"core-data_recipe.csv")
        cuisine_data = pd.read_csv(data_dir+"01_Recipe_Details.csv")
        cuisine_data.columns = cuisine_data.columns.str.replace('Title', 'recipe_name')

        core_data["recipe_name"] = core_data["recipe_name"].str.lower()
        cuisine_data["recipe_name"] = cuisine_data["recipe_name"].str.lower()

        df = core_data.merge(cuisine_data[["recipe_name", "Cuisine"]], on='recipe_name', how='left')
        df.dropna(inplace=True)

        #same foods with same recipe names dropped if exists
        df.drop_duplicates('recipe_name', ignore_index=True, inplace=True)

        df['cooking_directions'] = df['cooking_directions'].apply(ast.literal_eval).apply(lambda x: x['directions'].splitlines())
        df['total_prep_time'] = df['cooking_directions'].apply(lambda x: x[x.index('Ready In')+1].split() if 'Ready In' in x else np.nan)
        df.dropna(inplace=True)
        df['total_prep_time'] = df['total_prep_time'].apply(lambda lst: int(lst[0]) * 60 + int(lst[2]) if len(lst) == 4 else int(lst[0]))
        df['nutritions'] = df['nutritions'].apply(ast.literal_eval)
        
        nutrients_df = df['nutritions'].apply(lambda nutrition_dict: {name: nutrition["amount"] for name, nutrition in nutrition_dict.items()})

        df = pd.concat((df.reset_index(drop=True), pd.DataFrame.from_records(nutrients_df)), axis=1)

        return df

    def encode_ingredients(self, ingredients):

        wnl = WordNetLemmatizer()
        mlb = MultiLabelBinarizer()
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
        
        splitted_ings = ingredients.str.lower().str.replace(":", "^").str.split("^").apply(lambda row: [[wnl.lemmatize(word) for word in words.split()] for words in row if words])
        concat_ings =  splitted_ings.apply(lambda row : list(itertools.chain.from_iterable(row)))

        vectorizer.fit_transform(concat_ings.apply(" ".join))
        valid_words = vectorizer.get_feature_names_out()
        valid_words = valid_words[(valid_words >= "a") & (valid_words <= "z")]
        valid_words = [x for x in valid_words if len(x) > 2]

        filtered_ings1 = splitted_ings.apply(lambda row: [[word for word in words if word in valid_words] for words in row])

        ing_word_tags = list(map(pos_tag_sents, filtered_ings1.values)) #TAKES 30+ SECS FOR 15K FOODS

        noun_ings = [ [[ word_tag_tuple[0] for word_tag_tuple in tag_tuple_lst if word_tag_tuple[1] == 'NN' ]for tag_tuple_lst in ing_lst] for ing_lst in ing_word_tags]
        noun_ings = pd.Series(noun_ings, filtered_ings1.index)

        concat_noun_ings = noun_ings.apply(lambda row : list(itertools.chain.from_iterable(row)))

        encoded_ings = vectorizer.fit_transform(concat_noun_ings.apply(" ".join))

        ing_freq = pd.Series(encoded_ings.toarray().mean(axis=0), index=vectorizer.get_feature_names_out())

        filtered_ings2 = noun_ings.apply(lambda row: set([words[ing_freq.loc[words].argmax()] for words in row if words])) #TAKES 45+ SECS FOR 15K FOODS
        ing_df = pd.DataFrame(mlb.fit_transform(filtered_ings2),columns=mlb.classes_)

        return ing_df

    def drop(self, recipes, ing_df):
        #selected_ings = ing_df_.columns[ing_df_.mean() >= 10**-3]
        #ing_df = ing_df_[ing_df_[selected_ings].sum(axis=1) >= 2][selected_ings]
        #recipes = recipes.loc[ing_df.index].reset_index(drop=True)
      
        #Dropping instances with one ingredient
        ing_df = ing_df[ing_df.sum(axis=1) > 1].sort_index()
        recipes = recipes.loc[ing_df.index].sort_index()

        #Dropping instances with negative nutritional values
        mask = recipes[self.nutrient_names].apply(lambda x: x<0, axis=1).any(axis=1)
        to_drop = recipes[mask].index
        recipes.drop(to_drop, inplace=True)
        ing_df.drop(to_drop, inplace=True)

        #Resetting indicies
        recipes.reset_index(drop=True, inplace=True)
        ing_df.reset_index(drop=True, inplace=True)

        #removing empty columns
        empty_cols = ing_df.columns[ing_df.sum(axis=0)<1]
        ing_df.drop(empty_cols, axis=1, inplace=True)

        #converting binary ing_df to boolean for less storage
        ing_df = ing_df.astype(bool)

        return recipes, ing_df

    def split(self):
        if self.test_size > 0:
            self.train_idx, self.test_idx = train_test_split(self.data.index.tolist(), test_size=self.test_size, random_state=self.random_state)
        else:
            self.train_idx = self.data.index.tolist()
            self.test_idx = []
        print(f">> Dataset splitted with test ratio of {self.test_size} into train({len(self.train_idx)}) and test({len(self.test_idx)}) datasets.")
        return self.train_idx, self.test_idx

    def preprocess(self, data=None, data_transformer=None):
        if data is None and data_transformer is None:
            raise Exception("Either data or data_transfomer must be provided.")
        
        if data_transformer:
            self.data_transformer = data_transformer
        
        if data is None:
            data = self.data

        self.data_transformer.fit(data)
        self.X_train = self.transform(self.data.loc[self.train_idx])
        if len(self.test_idx) > 0:
            self.X_test = self.transform(self.data.loc[self.test_idx])
    
    def transform(self, x):
        return self.data_transformer.transform(x)

   
class FoodDataset3(): #diyetkolik.com dataset

    nutrient_names = ["Karbonhidrat (g)", "Protein (g)", "Yağ (g)", "Lif"]
    feature_names = ['category', 'ingredients', 'cuisine', 'total_prep_time', 'calories'] + nutrient_names
    categorical_features = ['category', 'ingredients', 'cuisine']
    continuous_features = list(set(feature_names) - set(categorical_features))
    meta_features = ['recipename', 'recipeimagelink-src']
    data = pd.DataFrame()
    similarity_df = pd.DataFrame()
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    train_idx = []
    test_idx = []

    def __init__(self, features_to_use: list, test_size = 0.0, random_state = 42):

        self.random_state = random_state
        self.test_size = test_size
        
        pkl_file_name = 'diyetkolik_dfs_dict.pkl'

        if os.path.exists(data_dir+pkl_file_name):
            with open(data_dir+pkl_file_name, 'rb') as f:
                dfs = pickle.load(f)
        else:
            raw, ing_df = self.load_and_parse_raw_data()
            dfs = dict(raw=raw, ingredients_df = ing_df)
            
            with open(data_dir+pkl_file_name, 'wb') as f:    
                pickle.dump(dfs, f)
       
        self.raw = dfs['raw']
        self.ingredients_df = dfs['ingredients_df'].astype(int)
        self.all_ingredients = sorted(list(self.ingredients_df.columns))
        
        self.set_data_features(features_to_use)
        
        self.translate_feature_names()
        self.log_scale_continuous_features(self.used_cont_features)
        lst = self.nutrient_names + ["Kalori"]
        self.convert_continuous_to_categorical(lst)

    def translate_feature_names(self):
        self.nutrient_names = ["K.hidrat", "Protein", "Yağ", "Lif"]
        self.feature_names = ['Kategori', 'Malzemeler', 'Mutfak', 'Süre', 'Kalori'] + self.nutrient_names
        self.categorical_features = ['Kategori', 'Malzemeler', 'Mutfak']
        self.continuous_features = list(set(self.feature_names) - set(self.categorical_features))
        self.meta_features = ['Yemek_Adı', 'yemek_foto_link']
        feature_map = {"Karbonhidrat (g)":"K.hidrat", "Protein (g)":"Protein", "Yağ (g)":"Yağ", "Lif":"Lif", 'category':'Kategori', \
                       'ingredients':'Malzemeler', 'cuisine':'Mutfak', 'total_prep_time':'Süre', 'calories':'Kalori', \
                       'recipename':'Yemek_Adi', 'recipeimagelink-src':'yemek_foto_link'}
        self.raw.rename(feature_map, axis=1, inplace=True)
        self.data.rename(feature_map, axis=1, inplace=True)
        self.used_cont_features = [feature_map[feature] for feature in self.used_cont_features if feature in feature_map]
    
    def log_scale_continuous_features(self, feature_names: list):
        target_features = list(set(self.used_cont_features) & set(feature_names))
        self.data[target_features] = (self.data[target_features] + 1e-6).apply(np.log).values
    
    def convert_continuous_to_categorical(self, feature_names, include_values=False):
        log_features = (self.raw[feature_names] + 1e-6).apply(np.log)

        q_vals = [33, 66, 100]
        quantile_df = pd.DataFrame(np.array([np.percentile(log_features, q, axis=0) for q in q_vals]), index=q_vals, columns=feature_names)
        
        cat_names = ['Düşük', 'Orta', 'Yüksek']

        categorical_vals = pd.DataFrame()
        
        for feature in feature_names:
            categorical_vals[feature] = pd.cut(log_features[feature], [-9999]+list(quantile_df[feature]), labels=cat_names).astype(str)

        self.raw[feature_names] = categorical_vals.values
        self.data.drop(feature_names, axis=1, inplace=True)
        cats = pd.get_dummies(categorical_vals, prefix_sep=": ")
        self.data = pd.concat((cats, self.data), axis=1)
        self.used_cat_features += cats.columns.tolist()
        self.used_cont_features = list(set(self.used_cont_features) - set(feature_names)) 

    def set_data_features(self, feature_list):

        self.used_cont_features = []
        self.used_cat_features = []

        if "nutritions" in feature_list:
            self.used_cont_features += self.nutrient_names

        if "calories" in feature_list:
            self.used_cont_features += ['calories']
        
        if "total_prep_time" in feature_list:
            self.used_cont_features += ["total_prep_time"]

        self.data = pd.DataFrame(self.raw[self.used_cont_features], index=self.raw.index)
        
        if "category" in feature_list:
            category_df = pd.get_dummies(self.raw['category']).drop("Diğer", axis=1)
            self.data = pd.concat((self.data, category_df), axis=1)
            self.used_cat_features += category_df.columns.tolist()
        
        if "cuisine" in feature_list:
            cuisine_df = pd.get_dummies(self.raw['cuisine'])
            self.data = pd.concat((self.data, cuisine_df), axis=1)
            self.used_cat_features += cuisine_df.columns.tolist()

        if 'ingredients' in feature_list:
            self.data = pd.concat((self.data, self.ingredients_df.astype(float)), axis=1)
            self.used_cat_features += self.ingredients_df.columns.tolist()

    def load_and_parse_raw_data(self):
        
        #reading the dataset
        df = pd.read_excel(data_dir + 'diyetkolik_recipes.xlsx')
        df.columns = df.columns.str.lower()
        
        #converting string dicts of BesinDegeriTotal to dicts and then dicts to columns 
        nutritions = pd.DataFrame.from_records(df['nutritions'].apply(ast.literal_eval)).replace('None', np.nan).astype(float)
        df[nutritions.columns] = nutritions
        
        #removing the time unit 'dk.' from prepTime and CookingTime and converting them to floats
        prepTime = df['preptime'].apply(lambda term: float(term.replace(' dk.', '')) if type(term) is str else float(term))
        cookTime = df['cookingtime'].apply(lambda term: float(term.replace(' dk.', '')) if type(term) is str else float(term))
        df['total_prep_time'] = prepTime + cookTime

        #replacing some rare categories and cuisines with similar ones
        df['category'].replace({'Meyveli Tatlı':'Tatlılar', 'Zeytinyağlı Baklagil Yemekleri':'Baklagil Yemekleri', \
            "Zeytinyağlı Dolmalar & Sarmalar":"Dolmalar ve Sarmalar", 'Kek':'Kekler', "Ekmekli Kahvaltılıklar":"Kahvaltılıklar", \
            "Süt ve Peynirli Kahvaltılıklar":"Kahvaltılıklar", "Et ve Yumurtalı Kahvaltılıklar": "Kahvaltılıklar", "Makarna":"Diğer",\
            "Diyet Sebze Yemekleri":"Sebze Yemekleri", "Poğaça/Tost":"Hamur İşleri", "Kahveler": 'İçecekler', "Çaylar": "İçecekler",\
            "Etsiz Baklagil Yemekleri":"Baklagil Yemekleri", "Baklagilli Salatalar":"Salatalar", "Sütlü Tatlı":"Tatlılar", \
            "Etli Çorbalar": "Çorbalar", "Etli Pilav":"Pilav", "Peynirli Tarifler":"Diğer", "Börek":"Hamur İşleri", \
            "Yeşil Yapraklı Salatalar":"Salatalar"}, inplace=True)

        df['cuisine'].replace({'Dünya - Dünya Mutfağı':'Dünya - Genel', 'Türkiye - Dünya Mutfağı':'Türkiye - Genel', \
            'Türkiye - Çin Mutfağı':'Çin - Çin Mutfağı', 'Çin - Genel':'Çin - Çin Mutfağı', 'Lübnan - Arap Mutfağı':'Lübnan - Lübnan Mutfağı', \
            'Hindistan - Genel': 'Hindistan - Hint Mutfağı', 'İtalya - Dünya Mutfağı':'İtalya - İtalyan Mutfağı', 'İtalya - Genel':'İtalya - İtalyan Mutfağı'}, inplace=True)
        
        #filling null values in cuisine with "Türkiye - Genel" as the most frequent cuisine in the dataset
        df['cuisine'].fillna("Türkiye - Genel", inplace=True)

        #converting string lists to lists
        df['ingredients'] = df['ingredients'].apply(ast.literal_eval)

        #discarding terms in paranthesis 
        def remove_term_in_paranthesis(word_group):
            if '(' in word_group:
                word_group = word_group[:word_group.index('(')]
                if word_group.endswith(' '):
                    word_group = word_group[:-1]
            return word_group
        
        df['ingredients'] = df['ingredients'].apply(lambda lst: list(map(remove_term_in_paranthesis, lst)))
        
        #one-hot encoding the ingredients
        mlb = MultiLabelBinarizer()
        ing_df = pd.DataFrame(mlb.fit_transform(df['ingredients']), columns=mlb.classes_, index=df.index)

        #dropping rare ingredients
        ingredients_to_drop = ing_df.columns[ing_df.sum() < 5]
        ing_df.drop(ingredients_to_drop, axis=1, inplace=True)

        #dropping if some rows have zero ingredients after dropping the rare ingredients
        ing_df = ing_df[ing_df.sum(axis=1) > 0]
        df = df.loc[ing_df.index]

        #filling null values in Category with the most similar mean ingredient vector representation
        cat_ing = pd.concat([df['category'], ing_df], axis=1)
        cat_repr = cat_ing.groupby('category').mean()
        r = pairwise_distances(ing_df.values, cat_repr.values, 'chebyshev')
        best_cat = np.array(cat_repr.index)[r.argmin(axis=1)]
        mask = df['category'].isna()
        df.loc[mask, 'category'] = best_cat[mask]

        ing_df = ing_df[df['category'] != 'İçecekler'].reset_index(drop=True)
        df = df[df['category'] != 'İçecekler'].reset_index(drop=True)
        
        #filling null values in total_prep_time with the mean total_prep_time values of categories
        cat_time_df = df[['category', 'total_prep_time']]
        mean_time_by_cat = cat_time_df.groupby('category').mean()
        cat_time_df['category'].fillna(-1, inplace=True)
        df["total_prep_time"] = cat_time_df.apply(lambda row: mean_time_by_cat.loc[row['category'], 'total_prep_time'] if np.isnan(row['total_prep_time']) and row['category'] != -1 else row['total_prep_time'], axis=1) 
   
        #filling the remaining null values with mean values
        num_cols = df.columns[df.dtypes != object]
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean(axis=0))

        #rounding ones digit of total_prep_time features to 0 or 5
        df["total_prep_time"] = df["total_prep_time"].apply(lambda num: 5 * round(num/5))
        
        cols = ['recipename', 'recipeimagelink-src', 'category', 'cuisine', 'total_prep_time', 'ingredients', 'calories'] + nutritions.columns.tolist()
        df = df[cols]

        assert df.isna().sum().sum() == 0
        assert df.index.tolist() == ing_df.index.tolist()

        df.reset_index(drop=True, inplace=True)
        ing_df.reset_index(drop=True, inplace=True)
        
        return df, ing_df
    
    def split(self):
        if self.test_size > 0:
            self.train_idx, self.test_idx = train_test_split(self.data.index.tolist(), test_size=self.test_size, random_state=self.random_state)
        else:
            self.train_idx = self.data.index.tolist()
            self.test_idx = []
        print(f">> Dataset splitted with test ratio of {self.test_size} into train({len(self.train_idx)}) and test({len(self.test_idx)}) datasets.")
        return self.train_idx, self.test_idx

    def preprocess(self, data=None, data_transformer=None):
        if data is None and data_transformer is None:
            raise Exception("Either data or data_transfomer must be provided.")
        
        if data_transformer:
            self.data_transformer = data_transformer
        
        if data is None:
            data = self.data

        self.data_transformer.fit(data)
        self.X_train = self.transform(self.data.loc[self.train_idx])
        if len(self.test_idx) > 0:
            self.X_test = self.transform(self.data.loc[self.test_idx])
    
    def transform(self, x):
        return self.data_transformer.transform(x)        
    

class DataTransformer():
    def __init__(self, test_size, random_state, vectorizer=None):
        self.test_size = test_size
        self.random_state = random_state 
        #self.vectorizer = vectorizer
        #self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def fit(self, data):
        #normalized_data = self.std_scaler.fit_transform(data)
        scaled_data = self.minmax_scaler.fit_transform(data)
        return scaled_data

    def transform(self, data):
        #normalized_data = self.std_scaler.transform(data)
        scaled_data = self.minmax_scaler.transform(data)
        if type(data) is pd.DataFrame:
            scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        
        return scaled_data

    # below part could be used for transforming data including categorical and continuous features   
    # def fit(self, data, cat_features, cont_features, ingredient_names):
        
    #     self.categorical_features = cat_features
    #     self.continuous_features = cont_features
    #     self.selected_data_idx = data.index.tolist()
        
    #     X = pd.DataFrame(index=data.index.tolist())

    #     if cat_features:
    #         concat_str_feature = pd.Series('', data.index)
    #         vocabulary = []

    #         for feature in cat_features:

    #             if feature == 'ingredients':
    #                 str_feature = data[feature].apply(";".join)
    #                 vocabulary += ingredient_names
    #             else:
    #                 str_feature = data[feature]
    #                 vocabulary += list(str_feature.unique())
                
    #             concat_str_feature += ";" + str_feature

    #         self.vectorizer.vocabulary = vocabulary
    #         self.vectorizer.tokenizer = lambda s: s.split(";")
    #         self.vectorizer.fit(concat_str_feature)

    #         self.train_idx, self.test_idx = self.split(self.selected_data_idx, self.test_size, self.random_state)
    #         vec_categorical = self.vectorizer.transform(concat_str_feature)
        
    #         cat_X = pd.DataFrame(vec_categorical.asformat('array'),
    #                          columns=self.vectorizer.get_feature_names_out(),
    #                          index=self.selected_data_idx)
    #         X = pd.concat((X, cat_X), axis=1)
        
    #     if cont_features:
    #         cont_X = (data[cont_features] + 0.001).apply(np.log)
    #         X = pd.concat((X, cont_X), axis=1)
        
    #     X_train = X.loc[self.train_idx]
    #     X_train = self.std_scaler.fit_transform(X_train)
    #     X_train = self.minmax_scaler.fit_transform(X_train)

    #     return self.train_idx, self.test_idx

    
    # def transform(self, data:pd.DataFrame):
        
    #     concat_str_feature = pd.Series('', data.index)
    
    #     if self.categorical_features:
    #         for feature in self.categorical_features:

    #             if feature == 'ingredients':
    #                 str_feature = data[feature].apply(";".join)
    #             else:
    #                 str_feature = data[feature]
    
    #             concat_str_feature += ";" + str_feature

    #         cat_X = self.vectorizer.transform(concat_str_feature).toarray()
        
        
    #     if self.scaler:
    #         x = self.scaler.transform(x)
        
    #     return x
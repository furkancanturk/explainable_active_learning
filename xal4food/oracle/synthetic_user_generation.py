import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import itertools
import pickle
import os
import random 
from copy import deepcopy
from typing import Union, Dict, List
from oracle.user import SyntheticUser
from data_builder import *

random.seed(42)
np.random.seed(42)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)

dir = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), "..", "..",
    )
)
data_dir = dir+"/data/"
oracle_dir = dir+"/oracle/"


#Foodataset1 LARGE SET: 200 ingredients
ing_groups1 = dict(

#meat 
meaty = ["butt", "broth", "fillet", "beef", "rib", "liver", "bacon", "chile", "lamb", "chicken", "top", "thigh", "breast", "roast", "sausage", "steak", "chop", "heart", "pork", "stock", "turkey", "meat", "ham", "wings"], #MAIN 
seafood = ["shrimp", "clam", "shell", "oyster", "crab", "scallop", "mussels"], #MAIN 

#vegetables
green = ["parsley", "leaves", "basil", "marjoram", "sprout", "broccoli", "peas",  "greens", "scallion", "leaf", "tarragon", "spinach", "olive", "dill", "lettuce", "leek"], #MAIN 
vegetable = ["onion", "tomato", "maca", "vegetable", "cabbage", "horseradish", "eggplant", "carrot", "mushroom", "potato", "celery", "shallot", "corn", "zucchini", "cucumber", "chives"], #MAIN 
fruit = ["apple", "raisins",  "avocado", "apricot", "cantaloupe", "dates", "peach", "fruit", 'limes', "peel", "lemon", "banana", "orange",  "halve", "half", "concentrate", "squash", "mango", "cherries", "grape", "strawberries", "pumpkin", "pear", "pineapple", "cranberries", "blueberries", "raspberries"],#AUXILIARY

#energy
oily = ["oil", "margarine", "shortening"],#AUXILIARY
pastry = ["flour", "yeast", "shortening", "crumbs", "crust"],#MAIN 
carby = ["bread", "noodles", "pasta", "rice", "tortellini"],#MAIN 

#animal
milky = ["milk", "cream", "chee"],#AUXILIARY
eggy = ["eggs", "white", "yolk"],#AUXILIARY

#taste
savory = ["sal"],#TASTE
sweet = ["cream", "sugar", "vanilla", "honey", "syrup", "chocolate", "syrup", "cocoa", "gel", "jam", "preserves", "molasses"],#TASTE
sour = ["lemon", "orange", "cranberries"],#TASTE
sweet_spicy = ["cinnamon", "clove", "gin", "coriander", "cardamom", "turmeric", "mint", "rind"],#TASTE
spicy = ["paprika", "garlic", "thyme", "cumin", "cilantro", "oregano", "basil", "spice", "seasoning", "rosemary", "cayenne", "jalapeno", "peppers"],#TASTE

#other
saucey = ["sauce", "vinegar", "mayonnaise", "ketchup", "mustard", "dressing"],#AUXILIARY
beverage = ["juice", "soda", "coffee", "ice", "sage"],#AUXILIARY
alcohol = ["wine", "rum", "beer", "sherry", "liqueur"],#AUXILIARY
snack = ["chips", "chocolate", "cereal", "cracker", "marshmallows", "bar", "peppercorn", "wedge", "tortilla", "stick"],#AUXILIARY
nuts = ["nutmeg", "almond", "walnuts", "nuts", "pecans", "coconut", "seed", "peanuts"],#AUXILIARY
grain = ["bean", "lentil", "bran", "oats","oatmeal", "chickpeas"],
uncategorized = ['cap', 'cube', 'tofu', 'paste', "puree", "sprig", "all", "zest", "water", "coloring", "powder", "extract", "spray", "slice", "granules", "substitute", "soup", "mix", "meal", "flakes"]
)

#Foodataset2 LARGE SET: 200 ingredients
ing_groups2 = dict(

#meat 
meaty = ["butt", "boneless", "prosciutto", "pepperoni", "broth", "fillet", "tail", 'poultry', "beef", "rib", "liver", 'bun', "bacon", 'head', "chile", "lamb", "chicken", "top", "thigh", "breast", "roast", "sausage", "steak", "chop", "heart", "pork", "stock", "turkey", "meat", "ham", "wings"], #MAIN 
seafood = ["shrimp", "clam", "crabmeat", "tuna", "salmon", "seafood", "shell", "oyster", "crab", "scallop", "mussels"], #MAIN 

#vegetables
green = ["parsley", "leaves", "herb", "basil", "asparagus", "marjoram", "caper", "sprout", "broccoli", "peas",  "greens", "scallion", "leaf", "tarragon", "spinach", "olive", "dill", "lettuce", "leek"], #MAIN 
vegetable = ["onion", "tomato", 'beet', 'chive', "garnish", 'salad', 'pea', 'okra', 'tomatillo', "maca", "vegetable", "cabbage", "horseradish", "eggplant", "carrot", "mushroom", "potato", "celery", "shallot", "corn", "zucchini", "cucumber", "chives"], #MAIN 
fruit = ["apple", "raisins", "strawberry", "blueberry", "raspberry", "cranberries", "applesauce", "avocado", 'cherry', "apricot", "cantaloupe", "dates",  'cranberry', "raisin", "peach", "fruit", 'limes', "peel", "lemon", "banana", "orange",  "halve", "half", "concentrate", "squash", "mango", "cherries", "grape", "strawberries", "pumpkin", "pear", "pineapple", "cranberries", "blueberries", "raspberries"],#AUXILIARY

#energy
oily = ["oil", "margarine", "shortening", 'butter'],#AUXILIARY
pastry = ["flour", "yeast", "crumbs", "crust", 'dough', 'pie', "pastry", "cornmeal", 'cornstarch'],#MAIN 
carby = ["bread", "noodles", "pasta", "rice", "tortellini", 'noodle', 'crumb', 'macaroni'],#MAIN 

#animal
milky = ["milk", "cream", "chee", "cheese", "yogurt"],#AUXILIARY
eggy = ["eggs", 'egg', "white", "yolk"],#AUXILIARY

#taste
savory = ["sal", 'salt', 'sauerkraut'],#TASTE
sweet = ["cream", "sugar", "vanilla", "honey", "tahini", "syrup", "chocolate", "syrup", "cocoa", "gel", "jam", "preserves", "molasses", "sweet"],#TASTE
sour = ["lemon", 'lime'],#TASTE
sweet_spicy = ["cinnamon",  "ginger", 'saffron', 'bay', "clove", "gin", "coriander", "cardamom", "turmeric", "mint", "rind"],#TASTE
spicy = ["paprika", "garlic", 'pepper', "chili", 'masala', "taco", "thyme", "cumin", "cilantro", "oregano", "basil", "spice", "seasoning", "rosemary", "cayenne", "jalapeno", "peppers"],#TASTE

#other
saucey = ["sauce", "vinegar", "mayonnaise", "ketchup", "mustard", "dressing", 'salsa', 'creole', 'pesto'],#AUXILIARY
beverage = ["juice", "soda", "coffee", "ice", "sage", "buttermilk"],#AUXILIARY
alcohol = ["wine", "rum", "beer", "sherry", "liqueur", "brandy", 'tequila', 'whiskey'],#AUXILIARY
snack = ["chips", "chocolate", "cereal", "cracker", "marshmallows", "bar", "peppercorn", "wedge", "tortilla", "stick"],#AUXILIARY
nuts = ["nutmeg", "almond", "walnuts", "walnut", "nut", "nuts", "pecans", "coconut", "seed", "peanuts", 'peanut', "cajun"],#AUXILIARY
grain = ["bean", "lentil", "bran", "oats","oatmeal", 'oat', "chickpeas", 'wheat', "hominy", 'quinoa'],
uncategorized = ['cap', 'cube', 'tofu', "cut", "bag", "bell", "roll", "food", "round", 'inch', "taste", "cooking", "thawed", "lengthwise", "fluid", "ground", 'chunk', 'paste', 'pint', 'medium', "puree", "sprig", "all", "zest", "water", "coloring", "powder", "extract", "spray", "slice", "granules", "substitute", "soup", "mix", "meal", "flakes", 'smoke', 'wrapper', 'blend', 'piece', 'chopped']
)

styles = dict(
vegan = dict(preferences = dict(vegetable=(8,10), fruit=(8,10), green=(8,10), meaty=(1,1), eggy=(1,1), milky=(1,1))),
sportive = dict(preferences = dict(milky=(6,10), eggy=(6,10), meaty=(7,10), seafood=(8,10), nuts=(8,10), oily=(1,3), snack=(1,1), pastry=(1,3), alcohol=(1,3), saucey=(1,5), carby=(5,7))),
unhealthy = dict(preferences = dict(alcohol=(7,10), oily=(6,10), sweet=(7,10), snack=(8,10), vegetable=(1,3), green=(1,1), saucey=(5,10),fruit=(1,3), seafood=(1,3))),
elder = dict(preferences = dict(savory=(1,5), oily=(1,3), grain=(5,10), nuts=(5,10), fruit=(7,10), green=(7,10), pastry=(1,5), alcohol=(1,3), saucey=(1,3), meaty=(5,10), snack=(1,3))),
)

users = dict(
user1 = dict(preferences = dict(pastry=(7,10), meaty=(6,10), spicy=(2,6), sweet_spicy=(1,5), sweet=(5,7), sour=(5,9), seafood=(1,7), oily=(5,8), carby=(5,10), milky=(8,10), nuts=(8,10), alcohol=(1,1), eggy=(7,10), fruit=(5,10))),
user2 = dict(preferences = dict(sweet=(7,10), sour=(1,5), spicy=(1,1), meaty=(7,10), milky=(1,3), pastry=(5,10), green=(1,5), beverage=(5,10), snack=(8,10), fruit=(7,10))),
user3 = dict(preferences = dict(saucey=(6, 10), sweet=(8,10), spicy=(5,7), meaty=(7,10), milky=(6,10), pastry=(7,10), green=(3,6), fruit=(7,10), alcohol=(8,10), carby=(6,10), sweet_spicy=(1, 3), seafood=(2,5), sour=(1,4)))
)

users['sportive_user1'] = deepcopy(users['user1'])
users['sportive_user1']['preferences'].update(styles['sportive']['preferences'])

users['vegan_user2'] = deepcopy(users['user2'])
users['vegan_user2']['preferences'].update(styles['vegan']['preferences'])

users['unhealthy_user1'] = deepcopy(users['user1'])
users['unhealthy_user1']['preferences'].update(styles['unhealthy']['preferences'])

users['elder_user2'] = deepcopy(users['user2'])
users['elder_user2']['preferences'].update(styles['elder']['preferences'])

users['sportive_user3'] = deepcopy(users['user3'])
users['sportive_user3']['preferences'].update(styles['sportive']['preferences'])

users['unhealthy_user3'] = deepcopy(users['user3'])
users['unhealthy_user3']['preferences'].update(styles['unhealthy']['preferences'])

synthetic_user_names = list(users.keys()) 

def generate_users(user_names: list, dataset: Union[FoodDataset, FoodDataset2], dataset_code: int, min_feature_freq: float, random_state: int):
    np.random.seed(random_state)

    selected_features = dataset.data.columns.tolist()

    base_user_ing_scores = np.random.uniform(low=0.5, high=10.5, size=len(selected_features)).round() #uniformly distributed random scores 
    base_user_ing_scores = pd.Series(base_user_ing_scores, index=selected_features)
    ing_groups = ing_groups1 if dataset_code == 1 else ing_groups2
    #some words belonging to user profiles may not be in FoodDataset, therefore they are removed from user profiles
    for user_name in user_names:
        profile = users[user_name]
        user_ing_scores = base_user_ing_scores.copy()

        n_effective_features = len(selected_features) // 2
        effective_features = list(np.random.choice(selected_features, replace=False, size = n_effective_features))
        
        no_effective_features_recipes = dataset.data[dataset.data[effective_features].sum(axis=1) == 0]
        
        def select(row):
            ingredients = list(row.iloc[np.nonzero(row.values)].index)
            feature =  np.random.choice(ingredients, size=1)[0]
            return feature

        if no_effective_features_recipes.shape[0] > 0:
            new_effective_features = no_effective_features_recipes.apply(select, axis=1)   
            effective_features += list(set(new_effective_features.values))
        
        ineffective_features = sorted(list(set(selected_features) - set(effective_features)))
 
        for ing_group, score_bounds in profile['preferences'].items():
            ingredients = ing_groups[ing_group]
            for i in ingredients[:]:
                if i not in selected_features:
                    ingredients.remove(i)

            new_scores = np.random.uniform(low=score_bounds[0]-0.5, high=score_bounds[1]+0.5, size=len(ingredients)).round()
            user_ing_scores[ingredients] = new_scores
            user_ing_scores[ineffective_features] = np.nan
 
        profile['ing_scores'] = user_ing_scores[selected_features]
    
    users['random'] = dict(preferences = dict(), ing_scores = base_user_ing_scores[selected_features])

    user_classes = {}
    selected_ing_data = dataset.ingredients_df[selected_features]
    for user_name, profile in users.items():
        like_threshold = 5.5#np.random.uniform(4,6)
        user_classes[user_name] = SyntheticUser(user_name, like_threshold, ingredients_df=selected_ing_data, random_state=random_state, **profile)

    for user_name, user_class in user_classes.items():
        with open(oracle_dir+f"synthetic_users/{user_name}_{dataset_code}_{random_state}_{min_feature_freq}.pkl", "wb") as f:
            pickle.dump(user_class, f)

    return user_classes


if __name__ == '__main__':

    data_random_state = 0
    test_size = 0.0
    diverse_data_size = 5000
    min_feature_freq = 0.001
    dataset_code = 2

    data_features = ['ingredients']
    food_dt_obj_file = f"fooddataset{dataset_code}_{data_random_state}_{test_size}_{min_feature_freq}_{diverse_data_size}.pkl"
    dataset = get_dataset(dataset_code, data_random_state, test_size, min_feature_freq, data_features, food_dt_obj_file, base=True)



    for random_state in range(5):
        
        # it can be used if data_features are only ingredients since synthetic user profiles are generated over ingredients
        generate_users(synthetic_user_names, dataset, dataset_code, min_feature_freq, random_state)

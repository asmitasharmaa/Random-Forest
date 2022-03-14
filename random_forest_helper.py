import copy
import json
import random

from math import log

import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin


def sse(y):
    # SSE = Σ(ŷi – yi)2
    pred = np.mean(y)
    e = np.sqrt(((y - pred)**2).sum())
    return e

def gain(y,x):    
    g = 0
    distinctVal = x.unique()
    df = pd.concat([x, y], axis=1) 
    for i in distinctVal:
        ser = df.loc[df[x.name] == i, y.name]
        g += sse(ser) * x.value_counts()[i]/len(y)
    return sse(y) - g


def gain_ratio(y,x):
    g = gain(y,x)
    return g/sse(y)


def select_split(X,y):
    col = None
    max_g = 0
    for i in X.columns:
        g = gain_ratio(y,X[i])
        if g > max_g:
            max_g = g
            col= i
    
    return col,max_g


# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))

    
def generate_rules(tree, prefix = []):
    rules = []
    # Base case:
    if type(tree) != dict:
        return tree
    
    root = list(tree.keys())[0]

    for k, v in tree[root].items():
        if type(v) == dict:
            t = generate_rules(v, prefix + [(root, k)])
            
            for i in t:
                rules.append(i)
        else:
            t = generate_rules(v, prefix + [(root, k)])
            l = prefix + [(root, k)] + [t]   
            rules.append(l)

    return rules


def reg_prediction(predictions):
    return pd.DataFrame(predictions).mode().to_numpy()[0]
    

def make_tree(X,y):
    tree = {}
    depth = 3
    # base case 1:
    if X.columns.size == 0:
        return stats.mode(y)
    
    # base case 2:
    if len(y.unique()) == 1:
        return y.unique()[0]
    
    if depth is None or depth >= len(X.columns):
            depth = len(X.columns)
            
    feature_index = random.sample(range(len(X.columns)), depth)
    features = []
    
    for i in range(len(X.columns)):
        if i in feature_index:
            features.append(X.columns[i])
    X = X.loc[:,features]
    
    col, gr = select_split(X,y)
    
    # base case 3:
    if gr < 0.000000001:
        return y.mode()[0]
    
    column = X[col]
    for value in column.unique():
        valx = X[column==value]
        valy = y[column==value]
        if col in tree:
            tree[col][value] = make_tree(valx, valy)
        else:
            tree[col] = {}
            tree[col][value] = make_tree(valx, valy)
        
    X = X.drop([col], axis = 1)
    return tree


# Resampling the data and making multiple trees
def make_trees(X, y, ntrees):
    trees = []
    for i in range(ntrees):
        X_sample, y_sample = resample(X, y)
        trees.append(make_tree(X_sample, y_sample))
    return trees

# Generate a rule for each tree, make a prediction for each rule
def reg_predict_all(trees, x, default):
    predic = []
    for tree in trees:
        rule = generate_rules(tree)
        prediction = []
        for i in range(len(x)):
            prediction.append(make_prediction(rule, x.iloc[i], default))
        predic.append(np.array(prediction))
    return np.array(predic)


def make_prediction(rules,x,default):
    for r in rules:
        for val in r:
            if type(val)!=tuple:
                return val
            if val[0] not in x:
                l = val[0].split("<")
                if (str(x[l[0]] < float(l[1])) != val[1]):
                    break
            elif x[val[0]] != val[1]:
                break        
    return(default)


## Lab 5 code ##

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)

# Bagging, make trees
def make_trees_bag(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        X_sample, y_sample = resample(X, y)
        trees.append(get_learner(X_sample, y_sample, max_depth=max_depth))
        
    return trees

# Bagging, prediction
def make_prediction_lab5(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

# Boosting, make trees
def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2):
    trees = []
    yval_pred = []
    ytrain_pred = []
    train_RMSEs = [] # the root mean square errors for the validation dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    ytrain_orig = copy.deepcopy(ytrain)
    for i in range(max_ntrees):
        tree = get_learner(Xtrain, ytrain, max_depth = max_depth)
        trees.append(tree)
        yval_pred.append(make_prediction_lab5([tree], Xval))
        ytrain_pred.append(make_prediction_lab5([tree], Xtrain))
        val_RMSEs.append(np.sqrt(((sum(yval_pred) - yval) ** 2).sum() / len(yval)))
        train_RMSEs.append(np.sqrt(((sum(ytrain_pred) - ytrain_orig) ** 2).sum() / len(ytrain_orig)))
        residuals = ytrain_orig - sum(ytrain_pred)
        ytrain = residuals
        
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    # Your solution here that finds the minimum validation score and uses only the trees up to that
    index = np.where(val_RMSEs == np.amin(val_RMSEs))[0][0]
    return trees[:index + 1]

# Boosting, make prediction
def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)
import copy
import json
import numpy as np
import pandas as pd
import random

from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)


def make_trees(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        X_sample, y_sample = resample(X, y)
        trees.append(get_learner(X_sample, y_sample, max_depth=max_depth))
        
    return trees

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

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
        yval_pred.append(make_prediction([tree], Xval))
        ytrain_pred.append(make_prediction([tree], Xtrain))
        val_RMSEs.append(np.sqrt(((sum(yval_pred) - yval) ** 2).sum() / len(yval)))
        train_RMSEs.append(np.sqrt(((sum(ytrain_pred) - ytrain_orig) ** 2).sum() / len(ytrain_orig)))
        residuals = ytrain_orig - sum(ytrain_pred)
        ytrain = residuals
        
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    # Your solution here that finds the minimum validation score and uses only the trees up to that
    index = np.where(val_RMSEs == np.amin(val_RMSEs))[0][0]
    return trees[:index + 1]

def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)


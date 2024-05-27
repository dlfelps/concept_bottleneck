from pathlib import Path
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class CAV():

  # trains a classifier for each attribute individually

  def __init__(self):
     self.concept_predictors = []

  def fit(self, X, y):
    for i in range(y.shape[1]):
      self.concept_predictors.append(self._fit_one(X, y[:,i]))

  def _fit_one(self, X, y):
    pipe = make_pipeline(StandardScaler(with_std=False), PCA(n_components=0.95), LogisticRegression(max_iter=1000))
    param_grid = {'logisticregression__C': [.001, .005, .01, .05, .1, .25, .5, 1]}
    clf = GridSearchCV(pipe, param_grid)
    clf.fit(X,y)
    return clf.best_estimator_

  def predict_proba(self, X):
    preds = []
    for cp in self.concept_predictors:
      preds.append(cp.predict_proba(X))

    return np.hstack(preds) # [N, 312]
  
  def predict_proba(self, X):
    preds = []
    for cp in self.concept_predictors:
      preds.append(cp.predict(X))

    return np.hstack(preds) # [N, 28]

  def dump_concept_predictors(self, file='cav.pkl'):
    with open(file, 'wb') as f:
      pickle.dump(self.concept_predictors, f)

  def load_concept_predictors(self, file='cav.pkl'):
    with open(file, 'rb') as f:
      self.concept_predictors = pickle.load(f)
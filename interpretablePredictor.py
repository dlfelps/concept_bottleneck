import pickle
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class InterpretablePredictor():

  def __init__(self):
    self.predictor = None

  def fit(self, attr_probs, classes):
    pipe = make_pipeline(StandardScaler(with_std=False), PCA(n_components=0.95), SVC())
    param_grid = {'svc__C': [.001, .005, .01, .05, .1, .25, .5, 1],
                  'svc__gamma': [.001, .005, .01, .025, .05]}
    clf = GridSearchCV(pipe, param_grid)
    clf.fit(attr_probs,classes)
    self.predictor = clf.best_estimator_

  def predict(self, attr_probs):
    return self.predictor.predict(attr_probs)

  def dump_interpretable_predictor(self, file='ip.pkl'):
    with open(file, 'wb') as f:
      pickle.dump(self.predictor, f)

  def load_interpretable_predictor(self, file='ip.pkl'):
    with open(file,'rb') as f:
      self.predictor = pickle.load(f)
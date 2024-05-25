import pickle

from datasets.CUB200 import CUB200, CUB200_attributes
from concept_bottleneck.cav import CAV
from concept_bottleneck.interpretablePredictor import InterpretablePredictor


class ConceptBottleneck():

  def __init__(self):
    self.concept_predictors = None
    self.interpretable_predictor = None

  def fit(self, embeddings, attributes, classes):
    cav = CAV()
    cav.fit(embeddings, attributes)
    self.concept_predictors = cav.concept_predictors
    probs = cav.predict_proba(embeddings)

    ip = InterpretablePredictor()
    ip.fit(probs, classes)
    self.interpretable_predictor = ip.predictor
    
  def predict(self, embeddings):
    cav = CAV()
    cav.concept_predictors = self.concept_predictors
    probs = cav.predict_proba(embeddings)

    ip = InterpretablePredictor()
    ip.predictor = self.interpretable_predictor
    preds = ip.predict(probs)
    return preds
  
  def load_concept_predictors(self, file='cav.pkl'):
    with open(file, 'rb') as f:
      self.concept_predictors = pickle.load(f)

  def load_interpretable_predictor(self, file='ip.pkl'):
    with open(file,'rb') as f:
      self.interpretable_predictor = pickle.load(f)
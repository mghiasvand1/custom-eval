from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import abc
import re


class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, **kwargs):
    self.X = X

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None

class kCenterGreedy(SamplingMethod):

  def __init__(self, X, metric='euclidean'):
    self.X = X
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []
    print('shape of features')
    print(X.shape)

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, features, already_selected, N, **kwargs):
    try:
      print('Getting transformed features...')
      self.features = features
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    if already_selected is None:
        already_selected = []
    self.already_selected = already_selected
    print(self.already_selected)

    new_batch = []

    for _ in range(N):
      if self.already_selected == []:
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      assert ind not in already_selected
      
      if self.min_distances is None:
        print('min distances is None')
      else:
        print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
      
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      
      if self.already_selected is None:
          self.already_selected = []
      else:
          self.already_selected.append(ind)

    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))
    
    return self.already_selected

areas = ["MENTALCHAT", "DEPTHQA"]
area = areas[0]
DATA_NUM = 60

df = pd.read_csv(f'data/synthetic/train_data_{area}.csv')

score_dict = {"1": [], "2": [], "3": [], "4": [], "5": []}
index_dict = {"1": [], "2": [], "3": [], "4": [], "5": []}

def parse_input_text(text):
    criteria = ""
    input_text = ""
    answer_text = ""
    
    criteria_match = re.search(r'Criteria:\s*(.*?)\n(?:Input:|$)', text, re.DOTALL)
    if criteria_match:
        criteria = criteria_match.group(1).strip()
    
    input_match = re.search(r'Input:\s*(.*?)\n(?:Answer:|$)', text, re.DOTALL)
    if input_match:
        input_text = input_match.group(1).strip()
    
    answer_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    
    return criteria, input_text, answer_text

def parse_output_text(text):
    explanation = ""
    score = None
    match = re.search(r'(.*?)\n\s*The final score is\s*(\d)\s*out of 5\.', text, re.DOTALL)
    if match:
        explanation = match.group(1).strip()
        score = match.group(2).strip()
    return explanation, score

for idx, row in df.iterrows():
    inp = row['input']
    out = row['output']
    
    crit, input_text, answer_text = parse_input_text(inp)
    explanation, score = parse_output_text(out)
    
    if score is None:
        continue
    
    concat_text = " ".join([crit, input_text, answer_text, explanation])
    
    if score in score_dict:
        score_dict[score].append(concat_text)
        index_dict[score].append(idx)
    else:
        score_dict[score] = [concat_text]
        index_dict[score] = [idx]

model = SentenceTransformer('all-MiniLM-L6-v2') 

encoded_dict = {}
for score, texts in score_dict.items():
    if texts:  
        encoded_texts = model.encode(texts)
        encoded_dict[score] = encoded_texts
    else:
        encoded_dict[score] = np.array([])

selected_indices_global = []

for score, encoded_features in encoded_dict.items():
    if encoded_features.size == 0:
        continue 
    
    n_samples = encoded_features.shape[0]
    n_select = min(DATA_NUM, n_samples)
    
    kcenter = kCenterGreedy(encoded_features, metric='euclidean')
    
    selected_local = kcenter.select_batch_(features=encoded_features, already_selected=[], N=n_select)
    
    original_indices = [index_dict[score][i] for i in selected_local]
    selected_indices_global.extend(original_indices)

df_selected = df.loc[selected_indices_global].reset_index(drop=True)

df_selected.to_csv(f"data/synthetic/filtered_train_data_{area}.csv", index=False)
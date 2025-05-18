import pandas as pd 
import numpy as np 
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from gensim import models, corpora 
from itertools import combinations
import pyLDAvis
import pyLDAvis.gensim_models 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import ast
import plotly.express as px 
import pickle


import spacy 
import string
import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

train=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/train.csv')
test=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/test.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])


use_embeddings = np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/use_embeddings.npy')
bert_embeddings= np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy')

print('loaded embeddings')
use_model_balanced = KMeans(n_clusters=3, random_state=42)
use_model_balanced_labels = use_model_balanced.fit_predict(use_embeddings)

print('calculated use bal')
use_model_regular = KMeans(n_clusters=83, random_state=42)
use_model_regular_labels = use_model_regular.fit_predict(use_embeddings)
print('calculated use reg')
bert_model_balanced = KMeans(n_clusters=81, random_state=42)
bert_model_balanced_labels = bert_model_balanced.fit_predict(bert_embeddings)
print('calculated bert bal')

bert_model_regular = KMeans(n_clusters=83, random_state=42)
bert_model_regular_labels = bert_model_regular.fit_predict(bert_embeddings)

# saving kmeans model for bert 
with open("kmean_model.pkl", "wb") as f:
    pickle.dump(bert_model_regular, f)

print('saved kmeans bert model')
print('calculated bert reg')

np.save('bert_model_regular_labels.npy',bert_model_regular_labels)
np.save('bert_model_balanced_labels.npy',bert_model_balanced_labels)
np.save('use_model_regular_labels.npy',use_model_regular_labels)
np.save('use_model_balanced_labels.npy',use_model_balanced_labels)

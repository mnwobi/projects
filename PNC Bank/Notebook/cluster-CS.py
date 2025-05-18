# getting the avg cohernce score based on clusters from Kmeans to try to get rid of some topic overlaps 

import pandas as pd 
import numpy as np 
import spacy 
import string
import random
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim import models, corpora 
from itertools import combinations
from collections import defaultdict





nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
stop_words= list(stop_words) + ['card','I']

def remove_stopwords_woLemma(text):
    
    # Convert to lowercase converts to tokens so easy to get lemmatize 
    doc = nlp(text.lower())
    cleaned_words = [
    token.text  # Lemmatization
    for token in doc
    if token.text not in string.punctuation and token.text not in stop_words
    
]

    return " ".join(cleaned_words).split()


# def compute_coherence(df):
#     # Tokenization
#     df['tokens'] = df['text'].apply(remove_stopwords_woLemma)  # Replace with better not removing stop words

#     # Group by label
#     label_docs = defaultdict(list)
#     for label, tokens in zip(df['labels_km'], df['tokens']):
#         label_docs[label].append(tokens)
    
#     coherence_scores = {}
#     top_words=[] # choosing top word from clusters as topic
#     for label, docs in label_docs.items(): # to get top word from each topic to creat topic
#         top_words=[]
#         dictionary = corpora.Dictionary(docs)
#         # corpus = [dictionary.doc2bow(text) for text in docs]
#         sorted_words = sorted(dictionary.dfs.items(), key=lambda x: x[1], reverse=True)
#         top_words.append(dictionary.get(sorted_words[0][0]))
    
#     # actually creating model
#     all_docs = [doc for docs in label_docs.values() for doc in docs]
#     dictionary = corpora.Dictionary(all_docs)

#     coherence_model = models.CoherenceModel(topics=top_words,
#             texts=all_docs,
#             dictionary=dictionary, 
#             coherence='c_v' 
#         )
        
#     coherence_scores[label] = coherence_model.get_coherence()
    
#     return coherence_scores



def compute_coherence(df):
    # Tokenization
    df['tokens'] = df['text'].apply(remove_stopwords_woLemma)

    # Group by label
    label_docs = defaultdict(list)
    for label, tokens in zip(df['labels_km'], df['tokens']):
        label_docs[label].append(tokens)
    
    coherence_scores = {}
    
    # For each label/cluster
    for label, docs in label_docs.items():
        # Create dictionary for this cluster
        dictionary = corpora.Dictionary(docs)
        
        # Get top words from this cluster as its "topic"
        sorted_words = sorted(dictionary.dfs.items(), key=lambda x: x[1], reverse=True)
        top_words = [dictionary[id] for id, freq in sorted_words[:10]]  # get top 10 words
        
        # Calculate coherence for this cluster
        coherence_model = models.CoherenceModel(
            topics=top_words,  # Must be list of lists
            texts=docs,          # Use only docs from this cluster
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_scores[label] = coherence_model.get_coherence()
    
    return coherence_scores




train=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/train.csv')
test=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/test.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])
vec_data= np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy')
data_ts_3= np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/TSNE_3.npy')


# starting loop 
# print('calculating scores')

tot_scores=[]
for clusters in range(10,91,10):
    print('calculating scores for cluster ',clusters ) 
    avg_score=[]
    kmeans = KMeans(n_clusters=91, random_state=42)
    labels_km = kmeans.fit_predict(vec_data)


    data['labels_km']=labels_km

    scores = compute_coherence(data)

    avg_score= sum(scores.values())/len(scores) #getting avg coherence scores
    # print(f'scores for {clusters}: {scores.values()} ') 
    print(f'for {clusters}, avg score is {avg_score}')

    tot_scores.append(avg_score)

plt.plot(list(range(10, 10 * (len(avg_score) + 1), 10)),avg_score)

plt.xlabel('Number of Clusters')
plt.ylabel('Coherence Scores')

plt.savefig('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/cluster-scores.png')
plt.clf()

print('saved fig ')

    
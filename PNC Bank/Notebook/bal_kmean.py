import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

import pandas as pd 
import matplotlib.pyplot as plt 

def balanced_score(X, labels, lambda_param=0.5):
    """
    Calculate the balanced score for clustering evaluation.
    
    Parameters:
    X: Feature matrix
    labels: Cluster assignments
    lambda_param: Weight of the penalty term (0 to 1)
    
    Returns:
    avg_balanced_score: Average balanced score across all samples
    individual_scores: Individual balanced scores for each sample
    """
    # Calculate traditional silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, labels) # traditional silhoutte score 
    
    # Get unique clusters and their sizes
    unique_clusters = np.unique(labels)
    K = len(unique_clusters)
    N = len(labels)
    cluster_sizes = np.array([np.sum(labels == i) for i in unique_clusters])
    
    # Calculate coefficient of variation (σ/μ)
    mean_cluster_size = np.mean(cluster_sizes)
    std_cluster_size = np.std(cluster_sizes)
    cv = std_cluster_size / mean_cluster_size if mean_cluster_size > 0 else 0
    
    # Calculate the penalty term
    cluster_size_deviations = np.sum(np.abs((cluster_sizes/N) - (1/K))) # penalty term
    penalty = lambda_param * cv * cluster_size_deviations #lmbda * sig/mean * pentaly
    
    # Calculate balanced scores for each sample
    balanced_scores = silhouette_vals - penalty
    
    return np.mean(balanced_scores), balanced_scores # silhoute score returns the mean so 

def find_optimal_k(X, k_range=range(2, 10), lambda_param=0.5):
    """
    Find the optimal number of clusters using the balanced score.
    
    Parameters:
    X: Feature matrix
    k_range: Range of k values to test
    lambda_param: Weight of the penalty term
    
    Returns:
    optimal_k: Optimal number of clusters
    scores: Dictionary of scores for each k
    """
    
    for k in k_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate balanced score
        bal_score, _ = balanced_score(X, cluster_labels, lambda_param)
        scores_bal[k] = bal_score
        sil_score = silhouette_score(X,cluster_labels)
        scores_orig[k] = sil_score
        
        # print(f"For k={k}, balanced score: {bal_score:.4f}, reg_score: {sil_score:.4f}")
    
    # Find k with the highest score
    optimal_k_bal = max(scores_bal, key=scores_bal.get)
    optimal_k_reg = max(scores_orig, key=scores_orig.get)

    return optimal_k_bal, scores_bal, optimal_k_reg, scores_orig

 #---------For USE embeddings---------


# 1 of the 3 mentioned in article 
embeddings = np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/use_embeddings.npy')


# getting the avg scores of 
use_bal_score=[]
use_reg_score=[]

for i in range(0,50):
    scores_bal = {}
    scores_orig={}
# Find optimal k
    optimal_k_bal, scores_bal, optimal_k_reg, scores_orig  = find_optimal_k(embeddings, k_range=range(2,86), lambda_param=0.5)

    # print(f"Optimal number of clusters for balance: {optimal_k_bal, optimal_k_reg} for regular")

    use_bal_score.append(optimal_k_bal)
    use_reg_score.append(optimal_k_reg)
    print(f'USE round {i}')

# Visualize scores
plt.figure(figsize=(10, 6))
plt.plot(list(scores_bal.keys()), list(scores_bal.values()), marker='o',label='Balanced')
plt.plot(list(scores_orig.keys()), list(scores_orig.values()), marker='o',label='Regular')
plt.xticks(np.arange(min(list(scores_orig.keys()))-1, max(list(scores_orig.keys()))-1, 5)) 
plt.xlabel('Number of clusters (k)')
plt.ylabel('Scores')
plt.title('Balanced Score vs Regular Score USE Embeddings')
plt.grid(True)
plt.legend(title="Score Types", loc='lower right', frameon=True, facecolor='lightgray', edgecolor='black')
plt.show()

plt.savefig('balance_sil_USE.png')

# Making histogram 

plt.figure(figsize=(10, 6))
# plt.hist(use_bal_score,label='Balanced',bins=10)
# plt.hist(use_reg_score,label='Regular', bins=10)
all_scores = np.concatenate([use_bal_score, use_reg_score])
bins = np.histogram_bin_edges(all_scores, bins='auto')

plt.hist([use_bal_score, use_reg_score], 
         label=['Balanced', 'Regular'], 
         bins=bins, 
         alpha=0.7, 
         edgecolor='black')
plt.xlabel('silhoutte score')
plt.ylabel('frequency')
plt.title('USE Embeddings Silhoutte Score Distrobution')
plt.grid(True)
plt.legend(title="Score Types", loc='lower right', frameon=True, facecolor='lightgray', edgecolor='black')
plt.show()

plt.savefig('distrobution_sil_USE.png')


# ____________For Bert__________

embeddings = np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy')

# getting the avg scores of 
bert_bal_score=[]
bert_reg_score=[]

for i in range(0,50):
    scores_bal = {}
    scores_orig={}
# Find optimal k
    optimal_k_bal, scores_bal, optimal_k_reg, scores_orig  = find_optimal_k(embeddings, k_range=range(2,86), lambda_param=0.5)

    # print(f"Optimal number of clusters for balance: {optimal_k_bal, optimal_k_reg} for regular")

    bert_bal_score.append(optimal_k_bal)
    bert_reg_score.append(optimal_k_reg)
    print(f'Bert round {i}')

# Visualize scores
plt.figure(figsize=(10, 6))
plt.plot(list(scores_bal.keys()), list(scores_bal.values()), marker='o',label='Balanced')
plt.plot(list(scores_orig.keys()), list(scores_orig.values()), marker='o',label='Regular')
plt.xticks(np.arange(min(list(scores_orig.keys()))-1, max(list(scores_orig.keys()))-1, 5)) 
plt.xlabel('Number of clusters (k)')
plt.ylabel('Scores')
plt.title('Balanced Score vs Regular Score BERT Embeddings')
plt.grid(True)
plt.legend(title="Score Types", loc='lower right', frameon=True, facecolor='lightgray', edgecolor='black')
plt.show()

plt.savefig('balance_sil_Bert.png')

# Making histogram 

plt.figure(figsize=(10, 6))
# plt.hist(bert_bal_score,label='Balanced',bins=10)
# plt.hist(bert_reg_score,label='Regular', bins=10)

all_scores = np.concatenate([bert_bal_score, bert_reg_score])
bins = np.histogram_bin_edges(all_scores, bins='auto')

plt.hist([bert_bal_score, bert_reg_score], 
         label=['Balanced', 'Regular'], 
         bins=bins, 
         alpha=0.7, 
         edgecolor='black')
plt.xlabel('silhoutte score')
plt.ylabel('frequency')
plt.title('Bert Embeddings Silhoutte Score Distrobution')
plt.grid(True)
plt.legend(title="Score Types", loc='lower right', frameon=True, facecolor='lightgray', edgecolor='black')
plt.show()

plt.savefig('distrobution_sil_Bert.png')


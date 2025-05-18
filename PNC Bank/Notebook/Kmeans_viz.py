from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np
# Get embedding of data 
vec_data= np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy')

print('computing  inertai and sil score')

# finding optimal number of clusters

inertia = []  # Stores the sum of squared distances (WCSS)
silhouette_scores=[] # Stores sil scores

K_range = range(2, 101)  
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vec_data)
    inertia.append(kmeans.inertia_)  # Sum of squared distances to centroids
    labels = kmeans.fit_predict(vec_data)
    score = silhouette_score(vec_data, labels)
    silhouette_scores.append(score)

print('completed scores')
print('Best cluster amount is ' ,np.argmax(silhouette_scores))
# Plot Elbow Curve
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.savefig('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/KM-Elbow.png')
plt.clf()


# Plot Silhouette Scores
plt.plot(range(2, 101), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.savefig('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/Sil_scores.png')
plt.clf()

print('Saved visuals')

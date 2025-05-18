from sklearn.manifold import TSNE
import numpy as np 
import pandas as pd
import tensorflow_hub as hub
import pickle

#________CREATING TSNE BASED ON BERT AND USE__________
# Get BERT embedding of data to
vec_data= np.load('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy')
print('Doing TSNE 3d')
tsne = TSNE(n_components=3)
data_ts = tsne.fit_transform(vec_data)

with open("tsne_model.pkl", "wb") as f:
    pickle.dump(tsne, f)

print('svaed 3d tsne model for bert ')

print('completed TSNE')
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/TSNE_3.npy', data_ts)

print('Saved file as TSNE_3.npy')

print('Doing TSNE 2d')
tsne = TSNE(n_components=2)
data_ts = tsne.fit_transform(vec_data)

print('completed TSNE')
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/TSNE.npy', data_ts)

print('Saved file as TSNE.npy')


# USE 

# have to run with gpu

# Load the USE model

train=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/train.csv')
test=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/test.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])


print('attempting full')
# performing SBERT vectorization
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") # 1 of the 3 mentioned in article 
embeddings = use_model(data['text'])
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/use_embeddings.npy',embeddings)
print('completed embedding')

# Get embedding of data 
print('Doing TSNE')
tsne = TSNE(n_components=3)
data_ts = tsne.fit_transform(embeddings.numpy())

print('completed TSNE 3D')
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/USE_TSNE_3.npy', data_ts)

print('Saved file as USE_TSNE_3.npy')

print('Doing TSNE 2D')
tsne = TSNE(n_components=2)
data_ts = tsne.fit_transform(embeddings.numpy())

print('completed TSNE 2d')
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/USE_TSNE.npy', data_ts)
print('complete')



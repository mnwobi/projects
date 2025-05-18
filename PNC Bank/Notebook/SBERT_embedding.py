import numpy as np 
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle


train=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/train.csv')
test=pd.read_csv('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/task-specific-datasets/banking_data/test.csv')
data=pd.concat([train,test]).reset_index()
data=data.drop(columns=['index'])


print('attempting full')
# performing SBERT vectorization
sb_model = SentenceTransformer("all-MiniLM-L6-v2") # 1 of the 3 mentioned in article 
embeddings = sb_model.encode(data['text'])
print(embeddings.shape)
print('completedt embedding')
np.save('/ocean/projects/cis240109p/mmarius/PNC_work/NLP/embedding.npy', embeddings)

print('Saved file as embedding.npy')

print('saved bert model')
with open("sb_model.pkl", "wb") as f:
    pickle.dump(sb_model, f)

# Load
# with open("sb_model.pkl", "rb") as f:
#     sb_model = pickle.load(f)
      



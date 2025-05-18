from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline 
import torch 
from huggingface_hub import login

import pandas as pd
import re

import numpy as np
from transformers import StoppingCriteria
from chromadb.config import Settings
import chromadb



# normalizing text before measurements
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# EM metric
def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))


# F1 score
def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = set(pred_tokens) & set(true_tokens)
    num_same = sum(min(pred_tokens.count(w), true_tokens.count(w)) for w in common)
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return int(pred_tokens == true_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_response(prompt):
    results = collection.query(
        query_texts=[prompt],
        n_results=3,
        include=["documents", "distances"]
    )
    
    # Get best match with lowest distance
    best_idx = np.argmin(results['distances'][0])
    context = results['documents'][0][best_idx]
    distance = results['distances'][0][best_idx]
    
    # Format for Llama 3 Instruct
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Context: {context[:20000]}
Question: {prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    # Generate response
    sequences = text_generator(formatted_prompt)
    
    return sequences[0]['generated_text'].strip(), distance




# Add custom stopping criteria
class NewlineStopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1]
        return last_token == tokenizer.encode("\n")[0]







# loading data set
dataset=pd.read_csv('/ocean/projects/cis240109p/mmarius/Genai_work/project/baseline_scores_llama3.csv')

#logging in 
print('loggingn in')
login(token="")
print('logged in')


print('loading model')

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
print('loaded model')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print('getting tokenizer')
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print('got tokenizer')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
print('loaded model')

# Updated text_generator configuration
print('making pipeline')
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,  # Reduced from 128
    temperature=0.3,    # Lower for more focused answers
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    stopping_criteria=[NewlineStopper()]  # Stop at first newline
)

# running model and getting results 


# building rag model 

chroma_client = chromadb.PersistentClient(
    path="ocean/projects/cis240109p/mmarius/chroma_data",  # Local storage directory
    settings=Settings(allow_reset=True)
)


collection = chroma_client.get_or_create_collection(
    name="llama3",
    metadata={"hnsw:space": "cosine"}
    
        # "embedding_function": cohere_ef # can change this to use differnt embeddings model
    # default query and vectorizer is all-MiniLM-L6-v2
)

# to mitigate None or NA issue 
valid_indices = dataset['URL_text'].notna()
filtered_documents = dataset.loc[valid_indices, 'URL_text'].tolist()
filtered_ids = [str(i) for i in dataset.loc[valid_indices, 'ID']]


# batching becuase document size too big 


# Split into batches of 5000 documents each
batch_size = 200  # Under Chroma's 5461 limit
total_docs = len(filtered_documents[:1000]) # comment out 1000 part
for i in range(0, total_docs, batch_size):
    end_idx = min(i + batch_size, total_docs)
    batch_doc = filtered_documents[i:end_idx]
    batch_ids = filtered_ids[i:end_idx]

    
    # documents = batch_doc
    # ids = [str(x) for x in batch_ids]
    
    # Add validation check
    if len(batch_doc) != len(batch_ids):
        raise ValueError(f"Document/ID count mismatch in batch {i}-{end_idx}")
    
    print(f"Adding batch {i//batch_size + 1}/{(total_docs//batch_size)+1}")
    collection.add(
        documents=batch_doc,
        ids=batch_ids
    )




# Getting respinses 
f1_score=[]
EM_score=[]
gen_response=[]
cos_score=[]

c=0
for index, row in dataset[:1000].iterrows(): # comment out 1000 part

    llama_gen,cos=get_response(row['question'])
    cos_score.append(cos)
    gen_response.append(llama_gen)
    f1_score.append(compute_f1(llama_gen,row['answer']))
    EM_score.append(compute_exact(llama_gen,row['answer']))
    print(cos,c)
    c+=1
new_data=dataset[:1000] # comment out 
new_data['llama3R_response']=gen_response
new_data['llama3R_f1_score']=f1_score
new_data['llama3R_EM_score']=EM_score
new_data['llama3R_cos_score']=cos_score

new_data.to_csv('baseline_scores_llama3R.csv', index=False)


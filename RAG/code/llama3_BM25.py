import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import config
from data_utils import chunk_document, load_nq_data
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline 
import torch 
from huggingface_hub import login
import pandas as pd
import re
from transformers import StoppingCriteria


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
    # getting context using BM25
    results = retriever.retrieve(prompt)
    

    context = results[0]['chunk']
    score = results[0]['score']
    # Format for Llama 3 Instruct
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Context: {context[:20000]}
Question: {prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    # Generate response
    sequences = text_generator(formatted_prompt)
    
    return sequences[0]['generated_text'].strip(), score




# Add custom stopping criteria
class NewlineStopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1]
        return last_token == tokenizer.encode("\n")[0]



# adding BM25 retriver
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Retriever:
    """BM25 retriever for Natural Questions dataset."""
    
    def __init__(self, chunk_size: int = None, stride: int = None):
        self.chunk_size = chunk_size or config.BM25_CONFIG["chunk_size"]
        self.stride = stride or config.BM25_CONFIG["stride"]
        self.bm25 = None
        self.chunks = []
        self.doc_ids = []  # To map chunks back to original documents
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25."""
        if type(text) == int:
            return word_tokenize(text)
        return word_tokenize(text.lower())
    
    def build_index(self, data, force_rebuild: bool = False):
        """Build BM25 index from the data."""
        index_path = os.path.join(config.MODEL_DIR, "bm25_index.pkl")
        chunks_path = os.path.join(config.MODEL_DIR, "bm25_chunks.pkl")
        
        # Check if index already exists
        if os.path.exists(index_path) and os.path.exists(chunks_path) and not force_rebuild:
            logger.info("Loading existing BM25 index...")
            with open(index_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(chunks_path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.chunks = loaded_data["chunks"]
                self.doc_ids = loaded_data["doc_ids"]
            return
        
        logger.info("Building BM25 index...")
        self.chunks = []
        self.doc_ids = []
        tokenized_chunks = []
        
        # Process each document
        for ind,example in  tqdm(data.iterrows()):
            # For simplified format, we'll use the answer as the "document"
            # This is a simplified approach that works for small demos
            
            document = str(example["URL_text"])
            
            
            # If there's no document content, create a simple placeholder with the question
            if not document:
                document = " No answer provided."
            
            # Create chunks (for simplicity, we'll just use the whole answer as one chunk)
            self.chunks.append(document)
            self.doc_ids.append(ind)
            tokenized_chunks.append(self.preprocess_text(document))
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Save index and chunks for future use
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        with open(chunks_path, 'wb') as f:
            pickle.dump({"chunks": self.chunks, "doc_ids": self.doc_ids}, f)
        
        logger.info(f"Built BM25 index with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for a query."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index first.")
        
        # Tokenize query
        tokenized_query = self.preprocess_text(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k chunk indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return top chunks with scores and doc_ids
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": scores[idx],
                "doc_id": self.doc_ids[idx]
            })
        
        return results

def oracle_retrieval(data: List[Dict[str, Any]], query_idx: int) -> Dict[str, Any]:
    """Get the oracle retrieval (chunk containing the answer) for evaluation."""
    example = data[query_idx]
    
    # For simplified format, we'll just use the answer directly
    answer = example.get("answer", "")
    
    # If no answer, use a placeholder
    if not answer:
        answer = "No answer available"
    
    return {
        "chunk": answer,
        "score": 1.0,  # Maximum score for oracle
        "doc_id": query_idx
    }
            


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


retriever = BM25Retriever()
retriever.build_index(dataset[:1000])



# Getting respinses 
f1_score=[]
EM_score=[]
gen_response=[]
cos_score=[]
# not cosine score but relevance score
for index, row in tqdm(dataset[:1000].iterrows()): # comment out 1000 part

    llama_gen,cos=get_response(row['question'])
    cos_score.append(cos)
    gen_response.append(llama_gen)
    f1_score.append(compute_f1(llama_gen,row['answer']))
    EM_score.append(compute_exact(llama_gen,row['answer']))
    print(cos)

new_data=dataset[:1000] # comment out 
new_data['llama3B_response']=gen_response
new_data['llama3B_f1_score']=f1_score
new_data['llama3B_EM_score']=EM_score
new_data['llama3B_relevance_score']=cos_score # BM25 uses relevance

new_data.to_csv('baseline_scores_llama3BM25.csv', index=False)


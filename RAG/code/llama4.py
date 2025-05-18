from transformers import AutoProcessor, AutoTokenizer, Llama4ForCausalLM, BitsAndBytesConfig, pipeline 
import torch 
from huggingface_hub import login
from datasets import load_dataset,load_from_disk
import re
import pandas as pd 
from transformers import StoppingCriteria

class NewlineStopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1]
        return last_token == tokenizer.encode("\n")[0]


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
    formatted_prompt = format_prompt(prompt)
    sequences = text_generator(formatted_prompt)
    return sequences[0]['generated_text'].strip()

def format_prompt(question):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# loading data
print('trying loading data set')

print('loggingn in')
login(token="")
# export XDG_CACHE_HOME=/ocean/projects/cis240109p/mmarius/.cache

print('loggin in ')

#export HF_HOME=/ocean/projects/cis240109p/mmarius/
print('starting')


model_name="meta-llama/Llama-4-Scout-17B-16E"
processor = AutoProcessor.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = Llama4ForCausalLM.from_pretrained(model_name,
#                                         attn_implementation='flex_attention'
#                                         ,torch_dtype=torch.bfloat16, device_map="auto")



tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token



# Final working configuration


model = Llama4ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

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





data=pd.read_csv('/ocean/projects/cis240109p/mmarius/Genai_work/project/baseline_scores_llama3.csv')


f1_score=[]
EM_score=[]
gen_response=[]

c=0
for index, row in data.iterrows():
 
  llama_gen=get_response(row['question'])
  gen_response.append(llama_gen)
  
  print(c)
  c+=0

  f1_score.append(compute_f1(llama_gen,row['answer']))
  EM_score.append(compute_exact(llama_gen,row['answer']))

data['llama4_response']=gen_response
data['llama4_f1_score']=f1_score
data['llama4_EM_score']=EM_score

data.to_csv('baseline_scores_llama4.csv', index=False)

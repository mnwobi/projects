from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline 
import torch 
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
import re
from bs4 import BeautifulSoup

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin


# This fill is how we get our data for our baseline so will return the csv file with teh F! and EM
# score, also the document/text from teh url 
# -----------TO GET TEXT FROM URL------------

def extract_text_from_page(url):
    """
    Extracts text from a given URL.
    """
    try:
        # Send a GET request to the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unwanted elements (e.g., scripts, styles, navbars, footers)
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()

    # Extract text from the main content
    text = soup.get_text(separator='\n', strip=True)
    return text




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

# loading data
print('trying loading data set')
dataset = load_dataset(
    "json",
    data_files="/ocean/projects/cis240109p/mmarius/Genai_work/project/data/v1.0-simplified_simplified-nq-train.jsonl",
    split="train"
)
# logging in to access hugging face models 
print('loggingn in')
login(token="REMOVED")
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

# for getting long and short answers from data 
def detokenize(text):
    # Remove spaces before punctuation
    text = re.sub(r'\s([?.!,;:"])', r'\1', text)
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


# getting answers from dataset 
def extract_and_clean_answer(example):
    tokens = example["document_text"].split()
    annotation = example["annotations"][0]
    
    # Long Answer
    long_start = annotation["long_answer"]["start_token"]
    long_end = annotation["long_answer"]["end_token"]
    long_answer = " ".join(tokens[long_start:long_end]) if long_start != -1 else None

    # Short Answers
    short_answers = []
    for sa in annotation["short_answers"]:
        start = sa["start_token"]
        end = sa["end_token"]
        short = " ".join(tokens[start:end])
        short_answers.append(short)

    # Clean them up
    if long_answer:
        long_answer = detokenize(clean_html(long_answer))
    short_answers = [detokenize(clean_html(sa)) for sa in short_answers]

    return long_answer, short_answers

# Add this function for proper instruction formatting
def format_prompt(question):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

from transformers import StoppingCriteria

# Add custom stopping criteria
class NewlineStopper(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1]
        return last_token == tokenizer.encode("\n")[0]

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

# Revised get_response function
def get_response(prompt):
    formatted_prompt = format_prompt(prompt)
    sequences = text_generator(formatted_prompt)
    return sequences[0]['generated_text'].strip()



# _______________Base LINE 
f1_score=[]
EM_score=[]
gen_response=[]
question=[]
answer=[]
Id=[]
text_doc=[]
stoper=0
for data in dataset:
  long_ans, short_ans_list = extract_and_clean_answer(data)
 
  if long_ans==None:
    continue
  answer.append(long_ans)
  question.append(data['question_text'])
  Id.append(data['example_id'])
  llama_gen=get_response(data['question_text'])
  gen_response.append(llama_gen)
  text_doc.append(extract_text_from_page(data['document_url']))
  print(stoper)
  print(gen_response)  
  f1_score.append(compute_f1(llama_gen,long_ans))
  EM_score.append(compute_exact(llama_gen,long_ans))
  if stoper==10000:
      break
  stoper+=1



df=pd.DataFrame({'question':question, 'response':gen_response, 'answer':answer, 'f1':f1_score, 'EM':EM_score, 'ID':Id, 'URL_text':text_doc})
df.to_csv('baseline_scores_llama3.csv', index=False)

# Base LINE 
# # -__-------- NOTESSSSSSSSSSS

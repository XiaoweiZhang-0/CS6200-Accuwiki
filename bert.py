import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertModel

#load dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# take first 2 as example
dataset = dataset.select(range(2))
# for i, example in enumerate(dataset):
#     print(f"id:{i}:")
#     print(example)

# initialize BERT Tokenizer & Model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

# chunkify
def chunkify_text(text, tokenizer, max_len=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunk_size = max_len - 2  #thresohld for [CLS] and [SEP]

    overlap = 20
    stride = chunk_size - overlap
    total_chunks = math.ceil(len(tokens) / stride)

    chunked_tokens = []
    for i in range(total_chunks):
        start = i * stride
        end = start + chunk_size
        chunk = tokens[start:end]
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunked_tokens.append(chunk)
        if end >= len(tokens):
            break


def process_single_doc(doc_id, text, tokenizer, model, max_len=512):

    chunked_tokens = chunkify_text(text, tokenizer, max_len)
    doc_id_list = []
    vector_list = []

    for tokens in chunked_tokens:
        input_ids = torch.tensor([tokens], device=model.device)
        with torch.no_grad():
            outputs = model(input_ids)
        # outputs.last_hidden_state: [1, seq_length, hidden_size]
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
        
        doc_id_list.append(doc_id)
        # pytorch tensor
        vector_list.append(cls_vec.squeeze(0).cpu())  # [hidden_size]

    return doc_id_list, vector_list


# main
all_doc_ids = []
all_vectors = []

for row in dataset:  
    doc_id = row["id"]
    text = row["text"]

    doc_id_list, vector_list = process_single_doc(doc_id, text, tokenizer, model)

    all_doc_ids.extend(doc_id_list)    # e.g. [1,1,1,2,2,2,2,...]
    all_vectors.extend(vector_list)    # e.g. [tensor(...), tensor(...), ...]

#----OUTPUT---
#print("all_doc_ids:", all_doc_ids)
#print("all_vectors (前3個):", all_vectors[:3])


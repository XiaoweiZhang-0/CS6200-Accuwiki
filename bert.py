import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertModel

# chunkify
def chunkify_text(text, max_len=256):
    chunked_text = []
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        chunked_text.append(chunk)
    return chunked_text


def process_single_doc(doc_id, text, model, max_len=256):

    chunked_tokens = chunkify_text(text, max_len)
    doc_id_list = []
    vector_list = []

    for tokens in chunked_tokens:
        doc_id_list.append(doc_id)

        vector_list.append(model.encode(tokens))  # [hidden_size]

    return doc_id_list, vector_list


def get_doc_id_and_vector(dataset, model):
    doc_id_list = []
    vector_list = []
    ## read from doc_id_list.pt and vector_list.pt if they exist
    try:
        doc_id_list = torch.load(f"doc_id_lists/doc_id_list_{doc_id}.pt")
        print("doc_id_list.pt exists")
        vector_list = torch.load(f"vector_lists/vector_list_{doc_id}.pt")
        print("vector_list.pt exists")
        ## skip to next doc right after the last processed doc
        last_doc_id = doc_id_list[-1]
        print("Resuming processing from doc_id:", last_doc_id)

        last_doc_idx = dataset["id"].index(last_doc_id)
        ## select the remaining docs
        dataset = dataset.select(range(last_doc_idx+1, len(dataset)))
        print("Processing started")
    except FileNotFoundError:
        print("Starting processing from the beginning")

    except Exception as e:
        print("file corrupted")
        print("Starting processing from the beginning")
        # return doc_id_list, vector_list
    
    for row in dataset:
        print("Processing doc_id:", row["id"])
        doc_id = row["id"]
        text = row["text"]
        doc_ids, vectors = process_single_doc(doc_id, text, model)
        doc_id_list.extend(doc_ids)
        vector_list.extend(vectors)
        torch.save(doc_id_list, f"doc_id_lists/doc_id_list_{doc_id}.pt")
        torch.save(vector_list, f"vector_lists/vector_list_{doc_id}.pt")

    print("Processing completed")
    return doc_id_list, vector_list



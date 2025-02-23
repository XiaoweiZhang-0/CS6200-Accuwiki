import os

import faiss
from sentence_transformers import SentenceTransformer
from bert import process_single_doc, get_doc_id_and_vector
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertModel
from faiss_index import build_and_save_index

#load dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# take first 2 as example
# dataset = dataset.select(range(100))

## confirm the size of the dataset
# print("Dataset size:", len(dataset))


# initialize sentencebert model
model = SentenceTransformer("all-MiniLM-L6-v2")


# get the doc_id and vectors
doc_id_list, vector_list = get_doc_id_and_vector(dataset, model)
embeddings = np.stack(vector_list)

## build the faiss index
build_and_save_index(dataset, np.array(embeddings))

## Load the dataset
# ds = Dataset.load_from_disk("wikipedia_dataset")

ds = dataset

# Load the FAISS index
index = faiss.read_index("faiss_index.index")


# Query
query = "Tigers"
query_embedding = np.stack(process_single_doc(0, query, model)[1])

# Set k to retrieve enough chunk matches
k = 100
distances, indices = index.search(query_embedding, k)

# Group the retrieved chunks by document ID and record the best (lowest) distance per document.
doc_to_best_score = {}
for dist, idx in zip(distances[0], indices[0]):
    # Look up the document id corresponding to this chunk
    doc_id = doc_id_list[idx]
    # Update with the best score seen so far for that document
    if doc_id not in doc_to_best_score or dist < doc_to_best_score[doc_id]:
        doc_to_best_score[doc_id] = dist

# Sort the documents by their best matching score (lowest first)
sorted_docs = sorted(doc_to_best_score.items(), key=lambda x: x[1])
top_5_docs = [doc_id for doc_id, score in sorted_docs[:5]]

## print the title of the top 5 docs and their scores
for doc_id in top_5_docs:
    doc_idx = dataset["id"].index(doc_id)
    print(f"Title: {dataset[doc_idx]['title']}, Score: {doc_to_best_score[doc_id]}")



from datasets import load_dataset
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from bert import process_single_doc
import torch


#load dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# initialize sentencebert model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the FAISS index
index = faiss.read_index("faiss_index.index")

doc_id_list = torch.load("doc_id_list.pt")

# Query

## ask for user input
query = input("Enter your query: ") 
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
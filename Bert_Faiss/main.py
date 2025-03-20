import os
from functools import partial

import faiss
import numpy as np
import pandas as pd
import torch
from batch_embedding import process_batch
from datasets import Dataset, load_dataset
from faiss_index import build_faiss_index
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer

# #load dataset
# dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# # initialize sentencebert model with cuda acceleration
# device = "cuda" if torch.cuda.is_available() else "cpu"
# confirm_device = input(f"Device: {device}. Continue? (y/n): ")
# if confirm_device.lower() != "y":
#     exit()
# model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# # Apply batch processing
# processed_data = dataset.map(lambda batch: process_batch(model, device, batch), batched=True, batch_size=1024, remove_columns=["text", "title", "url"])

# # Save processed_data to disk
# processed_data.save_to_disk("wikipedia_dataset_with_embeddings")


# load wikipeida_dataset_with_embeddings
processed_data = Dataset.load_from_disk("wikipedia_dataset_with_embeddings")


build_faiss_index(processed_data)




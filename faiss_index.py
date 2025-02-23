import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset\

import torch
from transformers import BertTokenizer, BertModel


## concatenate the tensor chunks for each doc
# def concatenateChunk(vector_list):
#     chunk_tensor = torch.stack(vector_list)
#     if chunk_tensor.shape[0] > 1:
#         mean_vector = torch.mean(chunk_tensor, dim=0)
#         max_vector, _ = torch.max(chunk_tensor, dim=0)
#         min_vector, _ = torch.min(chunk_tensor, dim=0)
#         std_vector = torch.std(chunk_tensor, dim=0, unbiased=False)  # or use the if-else approach
#     else:
#         mean_vector = chunk_tensor.squeeze(0)
#         max_vector = chunk_tensor.squeeze(0)
#         min_vector = chunk_tensor.squeeze(0)
#         std_vector = torch.zeros(chunk_tensor.shape[1], device=chunk_tensor.device)

#     aggregated_vector = torch.cat([mean_vector, max_vector, min_vector, std_vector], dim=0)
#     print("Aggregated vector shape:", aggregated_vector.shape)
#     ## Concatenate all these vectors into one aggregated vector
#     return aggregated_vector



## build the faiss index and save the dataset with index 
def build_and_save_index(ds, embeddings):


    # build the faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # save the faiss index
    faiss.write_index(index, "faiss_index.index")





# Initialize model and index
# model = SentenceTransformer("all-MiniLM-L6-v2")
# index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())  # 384-dim for "all-MiniLM-L6-v2"
# corpus = []
# embeddings = []

# # Process data in batches and save incrementally
# batch_size = 256
# count = 0

# with open("corpus.jsonl", "w") as corpus_file:
#     for example in ds:
#         title = example["title"]
#         print("Processing the "+ str(count) + " file")
#         count += 1
#         text = example["text"]
#         corpus.append(text)
        
#         # Encode in batches to reduce memory usage
#         if len(corpus) % batch_size == 0:
#             batch_embeddings = model.encode(corpus, convert_to_tensor=False)
#             embeddings.extend(batch_embeddings)
#             corpus = []  # Reset batch to free memory

# # Add remaining data
# if len(corpus) > 0:
#     batch_embeddings = model.encode(corpus, convert_to_tensor=False)
#     embeddings.extend(batch_embeddings)

# embeddings = np.array(embeddings, dtype=np.float32)

# # Add FAISS index to the dataset
# ds.add_faiss_index_from_external_arrays(
#     external_arrays=embeddings,
#     index_name="embeddings",
#     device=0  # Use GPU (-1 for CPU)
# )

# ds.save_faiss_index("embeddings", "wikipedia_index.faiss")

# # 2. Remove the index from the dataset
# ds.drop_index("embeddings")

# # 3. Now save the dataset to disk
# ds.save_to_disk("wikipedia_dataset")


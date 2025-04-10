import gc
import json
import math
import os
import time

import faiss
import numpy as np
import psutil
import torch
from constants import EMBEDDING_DIMENSION, N_PROBE
from tqdm import tqdm


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024 ** 3:.2f} GB")


def build_faiss_index(dataset):
    # Configuration
    EMBEDDING_DIM = EMBEDDING_DIMENSION
    INDEX_TYPE = "IVF4096,Flat"  # Good balance of speed/quality for 600M vectors
    NPROBE = N_PROBE  # Search parameter - will be saved with index
    BATCH_SIZE = 1000000  # Process this many vectors at once
    OUTPUT_PATH = "wikipedia_faiss_index"

    print("📊 Starting FAISS index building process")
    start_time = time.time()

    # Load dataset - just metadata, actual vectors loaded on demand
    # dataset = load_from_disk("Bert_Faiss/wikipedia_dataset_with_embedding")
    num_vectors = len(dataset)
    print(f"📚 Dataset loaded with {num_vectors:,} vectors")
    print_memory_usage()

    # Check dataset columns
    columns = dataset.column_names
    print(f"📋 Dataset columns: {columns}")

    if "id" not in columns or "embedding" not in columns:
        raise ValueError(f"Expected 'id' and 'embedding' columns but found: {columns}")

    # Check and save ID mapping (important for retrieval later)
    print("🔑 Creating ID mapping...")

    # Verify embedding dimension
    sample = dataset[0]["embedding"]
    if sample is None or len(sample) == 0:
        raise ValueError("embedding are missing or empty in dataset")
    actual_dim = len(sample)
    if actual_dim != EMBEDDING_DIM:
        print(
            f"⚠️ Warning: Expected embedding dimension {EMBEDDING_DIM} but found {actual_dim}"
        )
        EMBEDDING_DIM = actual_dim

    # Create index
    print(f"🏗️ Creating {INDEX_TYPE} index with dimension {EMBEDDING_DIM}...")

    # Determine number of clusters based on dataset size
    # Rule of thumb: sqrt(N) for small datasets, 4*sqrt(N) for large ones
    num_clusters = min(16384, int(4 * math.sqrt(num_vectors)))
    # Ensure it's a power of 2 for efficiency
    num_clusters = 2 ** int(math.log2(num_clusters) + 0.5)

    print(f"🔍 Using {num_clusters} clusters for IVF index")
    index_type = f"IVF{num_clusters},Flat"

    # Create empty index
    if torch.cuda.is_available():
        print("🎯 Using GPU for index building")
        # Configure for GPU usage
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexIVFFlatConfig()
        config.device = 0  # GPU id
        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        # Create a CPU index first
        print("🔧 Creating ScalarQuantizer CPU index...")
        # cpu_index = faiss.IndexIVFScalarQuantizer(
        #     quantizer,
        #     EMBEDDING_DIM,
        #     num_clusters,
        #     faiss.ScalarQuantizer.QT_8bit,
        #     faiss.METRIC_INNER_PRODUCT,
        # )
        M = 12  # 384 / 32 = 12 subquantizers (adjust based on dim)
        nbits = 8  # Higher bitrate for PQ
        cpu_index = faiss.IndexIVFPQ(
            quantizer,
            EMBEDDING_DIM,
            num_clusters,
            M,
            nbits,
        )

        # Convert to GPU
        print("🚀 Converting CPU index to GPU...")
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        # index.nprobe = NPROBE
    # else:
    #     print("🖥️ Using CPU for index building")
    #     quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
    #     index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, num_clusters,
    #                               faiss.METRIC_INNER_PRODUCT)
    #     index.nprobe = NPROBE

    # First, collect sample for training
    print("🧠 Collecting samples for training...")
    # train_size = min(1000000, num_vectors)  # 1M samples or entire dataset
    train_size = min(100 * num_clusters, num_vectors)

    # Generate random indices for training
    if num_vectors > train_size:
        train_indices = np.random.choice(num_vectors, train_size, replace=False)
    else:
        train_indices = np.arange(num_vectors)

    # Load training vectors in batches to avoid memory issues
    batch_size = min(BATCH_SIZE, train_size)
    num_batches = (train_size + batch_size - 1) // batch_size

    # Pre-allocate array if embedding dimensions are known (faster than appending)
    # Assuming embedding_dim is the dimension of each vector
    if EMBEDDING_DIM:
        train_vectors = np.zeros((train_size, EMBEDDING_DIM), dtype=np.float32)

        # Fill pre-allocated array batch by batch
        current_idx = 0
        for i in tqdm(range(num_batches), desc="Loading training vectors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_size)
            batch_indices = train_indices[start_idx:end_idx]

            batch = dataset.select(batch_indices)
            batch_vectors = np.array(batch["embedding"], dtype=np.float32)

            # Copy batch vectors to pre-allocated array
            batch_length = len(batch_vectors)
            train_vectors[current_idx : current_idx + batch_length] = batch_vectors
            current_idx += batch_length

            # Free memory
            del batch, batch_vectors

    else:
        # If embedding_dim is unknown, use list append + vstack approach
        train_vectors = []
        for i in tqdm(range(num_batches), desc="Loading training vectors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_size)
            batch_indices = train_indices[start_idx:end_idx]

            batch = dataset.select(batch_indices)
            batch_vectors = np.array(batch["embedding"], dtype=np.float32)
            train_vectors.append(batch_vectors)

            # Free memory
            del batch

    # Concatenate all vectors at once
    train_vectors = np.vstack(train_vectors)
    M = 12
    opq = faiss.OPQMatrix(EMBEDDING_DIM, M)
    opq.train(train_vectors)
    train_vectors_opq = opq.apply(train_vectors)

    # Train the index
    print("🏋️ Training the index...")
    # train_vectors = train_vectors.astype(np.float32)  # Ensure correct data type
    train_vectors_opq = train_vectors_opq.astype(np.float32)  # Ensure correct data type
    index.train(train_vectors_opq)
    del train_vectors, train_vectors_opq
    gc.collect()
    print_memory_usage()

    # Add vectors to index in batches
    print("➕ Adding vectors to index in batches...")
    num_batches = (num_vectors + BATCH_SIZE - 1) // BATCH_SIZE

    # Create a mapping from FAISS index to document ID
    id_mapping = {}
    current_idx = 0

    for i in tqdm(range(num_batches), desc="Adding to index"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, num_vectors)

        # Load batch of vectors
        batch = dataset.select(range(start_idx, end_idx))
        batch_vectors = np.array(batch["embedding"])
        batch_ids = batch["id"]

        # Add to index
        batch_vectors = batch_vectors.astype(np.float32)  # Ensure correct data type
        index.add(batch_vectors)

        # Store ID mapping
        for doc_id in batch_ids:
            id_mapping[current_idx] = doc_id
            current_idx += 1

        # Free memory
        del batch, batch_vectors, batch_ids
        gc.collect()

    # Convert to CPU index for saving if using GPU
    if torch.cuda.is_available():
        print("🔄 Converting GPU index to CPU for saving...")
        index = faiss.index_gpu_to_cpu(index)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save index
    print("💾 Saving index...")
    faiss.write_index(index, os.path.join(OUTPUT_PATH, "wikipedia.index"))

    # Save ID mapping
    print("💾 Saving ID mapping...")
    # Save in chunks to avoid memory issues with large mappings
    mapping_chunks = {}
    chunk_size = 1000000  # 1M entries per file

    for chunk_start in range(0, len(id_mapping), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(id_mapping))
        chunk_dict = {
            str(idx): id_mapping[idx] for idx in range(chunk_start, chunk_end)
        }

        chunk_file = os.path.join(
            OUTPUT_PATH, f"id_mapping_{chunk_start}_{chunk_end}.json"
        )
        with open(chunk_file, "w") as f:
            json.dump(chunk_dict, f)

        mapping_chunks[f"{chunk_start}_{chunk_end}"] = chunk_file

    # Save mapping metadata
    with open(os.path.join(OUTPUT_PATH, "id_mapping_index.json"), "w") as f:
        json.dump(
            {
                "total_vectors": len(id_mapping),
                "chunk_size": chunk_size,
                "chunks": mapping_chunks,
            },
            f,
        )

    # Save metadata
    with open(os.path.join(OUTPUT_PATH, "metadata.txt"), "w") as f:
        f.write(f"Dataset: wikipedia_dataset_with_embedding\n")
        f.write(f"Number of vectors: {num_vectors}\n")
        f.write(f"Embedding dimension: {EMBEDDING_DIM}\n")
        f.write(f"Index type: {index_type}\n")
        f.write(f"nprobe: {NPROBE}\n")

    elapsed_time = time.time() - start_time
    print(f"✅ Index building completed in {elapsed_time/60:.2f} minutes")
    print(f"📁 Index saved to {os.path.join(OUTPUT_PATH, 'wikipedia.index')}")
    print_memory_usage()

    return index, id_mapping

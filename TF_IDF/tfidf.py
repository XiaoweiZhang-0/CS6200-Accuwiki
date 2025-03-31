import gc
import glob
import math
import os
from pathlib import Path

import cudf
import cupy as cp
import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
from datasets import Dataset, load_dataset
from rank import search_tfidf_index
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def arrow_to_cudf(table):
    """
    Convert a PyArrow table to cuDF DataFrame, handling list columns properly.

    Args:
        table: PyArrow Table object

    Returns:
        cudf.DataFrame
    """
    # For columns with list type, we need special handling
    data = {}
    for col_name in table.column_names:
        col = table[col_name]
        if pa.types.is_list(col.type):
            # Convert list column to Python lists
            data[col_name] = col.to_pylist()
        else:
            # Regular columns can be converted directly
            data[col_name] = col.to_pandas()

    # Create cuDF DataFrame
    return cudf.DataFrame.from_pandas(data)


def build_tfidf_index_gpu(
    processed_folder,
    output_folder,
    min_df=5,
    max_df=0.5,
    max_features=100000,
    batch_size=5000,
    file_batch_size=1,
    sample_size=25000,
):
    """
    Build TF-IDF index from preprocessed Wikipedia data using GPU acceleration
    with improved memory management for systems with limited memory
    """
    os.makedirs(output_folder, exist_ok=True)

    print("🔍 Finding all processed files...")
    parquet_files = list(Path(processed_folder).glob("*.parquet"))
    print(f"📂 Found {len(parquet_files)} files to process")

    # Create a sampling index file to avoid loading all doc IDs in memory
    doc_ids_file = os.path.join(output_folder, "doc_ids.joblib")
    if not os.path.exists(doc_ids_file):
        print("📝 Processing files to extract document IDs...")

        # We'll just collect IDs and a sample of documents initially
        all_doc_ids = []
        sampling_docs = []
        current_sample_size = 0

        # Process files to extract doc IDs and sample
        for file_idx, file_path in enumerate(
            tqdm(parquet_files, desc="Processing files for IDs")
        ):
            try:
                # Read with PyArrow but only ID column
                table = pq.read_table(file_path, columns=["id"])
                ids_list = table["id"].to_pandas().tolist()
                all_doc_ids.extend(ids_list)

                # If we need more samples, load tokens too
                if current_sample_size < sample_size:
                    # Read tokens for sampling
                    tokens_table = pq.read_table(file_path, columns=["tokens"])
                    tokens_list = tokens_table["tokens"].to_pylist()

                    # Only take what we need to reach sample_size
                    needed = min(len(tokens_list), sample_size - current_sample_size)

                    # Convert tokens to strings for the sample
                    for i in range(needed):
                        tokens = tokens_list[i]
                        if isinstance(tokens, list):
                            sampling_docs.append(" ".join(tokens))
                        else:
                            sampling_docs.append(str(tokens))

                    current_sample_size += needed

                    # Clean up
                    del tokens_table, tokens_list

                # Clean up
                del table, ids_list
                gc.collect()

                # Report progress
                if (file_idx + 1) % 10 == 0:
                    process = psutil.Process(os.getpid())
                    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
                    print(
                        f"Memory usage after {file_idx + 1} files: {memory_gb:.2f} GB"
                    )
                    print(f"Collected {len(all_doc_ids)} document IDs so far")
                    print(f"Sampling docs: {current_sample_size}/{sample_size}")

            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                print(f"Skipping this file and continuing...")

        # Save all document IDs
        joblib.dump(all_doc_ids, doc_ids_file)
        print(f"📊 Total documents collected: {len(all_doc_ids)}")

        # Clean up IDs list to free memory
        doc_count = len(all_doc_ids)
        del all_doc_ids
        gc.collect()
    else:
        # Get doc count from the existing file
        print(f"Using existing document IDs file: {doc_ids_file}")
        doc_count = len(joblib.load(doc_ids_file))
        print(f"📊 Total documents found: {doc_count}")

        # Collect samples
        print("Collecting document samples for vocabulary building...")
        sampling_docs = []
        current_sample_size = 0

        # Process files to extract samples
        for file_idx, file_path in enumerate(
            tqdm(parquet_files, desc="Processing files for samples")
        ):
            # Stop if we have enough samples
            if current_sample_size >= sample_size:
                break

            try:
                # Read tokens for sampling
                tokens_table = pq.read_table(file_path, columns=["tokens"])
                tokens_list = tokens_table["tokens"].to_pylist()

                # Only take what we need to reach sample_size
                needed = min(len(tokens_list), sample_size - current_sample_size)

                # Convert tokens to strings for the sample
                for i in range(needed):
                    tokens = tokens_list[i]
                    if isinstance(tokens, list):
                        sampling_docs.append(" ".join(tokens))
                    else:
                        sampling_docs.append(str(tokens))

                current_sample_size += needed

                # Clean up
                del tokens_table, tokens_list
                gc.collect()

            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                print(f"Skipping this file and continuing...")

    # Step 2: Build TF-IDF index using the samples
    print("🔢 Building TF-IDF vocabulary on GPU using samples...")

    # Create GPU TF-IDF vectorizer
    vectorizer = cuTfidfVectorizer(
        analyzer="word",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        use_idf=True,
        norm="l2",
    )

    # Fit on the samples
    print(f"Fitting vocabulary on {len(sampling_docs)} sampled documents...")
    vectorizer.fit(cudf.Series(sampling_docs))

    # Get vocabulary and save it
    vocab = vectorizer.get_feature_names()
    vocab_list = vocab.to_arrow().to_pylist()  # Convert to Python list
    vocabulary = {term: idx for idx, term in enumerate(vocab_list)}

    # Save vocabulary
    vocab_path = os.path.join(output_folder, "vocabulary.joblib")
    joblib.dump(vocabulary, vocab_path)
    print(f"Vocabulary size: {len(vocabulary)}")

    # Save IDF values
    idf_values = cp.asnumpy(vectorizer.idf_)
    idf_path = os.path.join(output_folder, "idf_values.npy")
    np.save(idf_path, idf_values)

    # Clean up sample documents
    del sampling_docs
    gc.collect()

    # Step 3: Transform documents in batches and save in chunks
    print("Transforming documents in batches...")

    # We'll process and save in manageable chunks
    chunk_size = 500000  # Number of documents per chunk
    num_chunks = (doc_count + chunk_size - 1) // chunk_size

    # Create a file index mapping to track which documents are in which files
    print("Creating file index mapping...")
    file_doc_counts = []
    cumulative_doc_count = 0

    for file_path in tqdm(parquet_files, desc="Counting documents per file"):
        try:
            # Read only ID column to count documents
            table = pq.read_table(file_path, columns=["id"])
            doc_count_in_file = len(table)
            file_doc_counts.append(
                (
                    file_path,
                    cumulative_doc_count,
                    cumulative_doc_count + doc_count_in_file,
                )
            )
            cumulative_doc_count += doc_count_in_file

            # Clean up
            del table
            gc.collect()
        except Exception as e:
            print(f"❌ Error counting documents in {file_path}: {str(e)}")
            print(f"Skipping this file and continuing...")

    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_doc = chunk_idx * chunk_size
        end_doc = min((chunk_idx + 1) * chunk_size, doc_count)

        print(
            f"Processing chunk {chunk_idx+1}/{num_chunks} (documents {start_doc+1}-{end_doc})"
        )

        # Initialize lists for this chunk
        chunk_doc_ids = []
        chunk_matrices = []

        # Find files containing documents for this chunk
        relevant_files = []
        for file_path, file_start, file_end in file_doc_counts:
            # Check if file's document range overlaps with chunk's document range
            if not (file_end <= start_doc or file_start >= end_doc):
                relevant_files.append((file_path, file_start, file_end))

        print(f"Found {len(relevant_files)} files containing documents for this chunk")

        # Process only the relevant files for this chunk
        for file_idx, (file_path, file_start, file_end) in enumerate(
            tqdm(relevant_files, desc=f"Processing files for chunk {chunk_idx+1}")
        ):
            try:
                # Read the file
                table = pq.read_table(file_path, columns=["id", "tokens"])

                # Get IDs and tokens
                ids_list = table["id"].to_pandas().tolist()
                tokens_list = table["tokens"].to_pylist()

                # Calculate which documents from this file belong to the current chunk
                chunk_start_idx = max(0, start_doc - file_start)
                chunk_end_idx = min(len(ids_list), end_doc - file_start)

                # Extract only documents that belong to this chunk
                batch_doc_ids = ids_list[chunk_start_idx:chunk_end_idx]
                batch_tokens = tokens_list[chunk_start_idx:chunk_end_idx]

                # Join tokens to strings
                batch_docs = []
                for tokens in batch_tokens:
                    if isinstance(tokens, list):
                        batch_docs.append(" ".join(tokens))
                    else:
                        batch_docs.append(str(tokens))

                # Add to chunk collections
                chunk_doc_ids.extend(batch_doc_ids)

                # Transform in mini-batches to manage GPU memory
                mini_batch_size = min(batch_size, len(batch_docs))
                mini_batch_count = (
                    len(batch_docs) + mini_batch_size - 1
                ) // mini_batch_size

                for i in range(mini_batch_count):
                    mini_start = i * mini_batch_size
                    mini_end = min((i + 1) * mini_batch_size, len(batch_docs))

                    # Get mini-batch
                    mini_batch = batch_docs[mini_start:mini_end]

                    # Convert to cuDF Series
                    cu_batch = cudf.Series(mini_batch)

                    # Transform mini-batch
                    mini_matrix = vectorizer.transform(cu_batch)

                    # Convert to SciPy sparse and store
                    if hasattr(mini_matrix, "to_scipy_sparse"):
                        scipy_matrix = mini_matrix.to_scipy_sparse()
                    else:  # For newer versions
                        scipy_matrix = mini_matrix.get()

                    # Store
                    chunk_matrices.append(scipy_matrix)

                    # Free GPU memory
                    del cu_batch, mini_matrix
                    cp._default_memory_pool.free_all_blocks()
                    gc.collect()

                # Clean up
                del (
                    table,
                    ids_list,
                    tokens_list,
                    batch_docs,
                    batch_doc_ids,
                    batch_tokens,
                )
                gc.collect()

            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                print(f"Skipping this file and continuing...")

        # Combine matrices for this chunk
        print(f"Combining {len(chunk_matrices)} matrices for chunk {chunk_idx+1}...")
        chunk_matrix = sp.vstack(chunk_matrices)

        # Save chunk TF-IDF matrix
        chunk_tfidf_path = os.path.join(
            output_folder, f"tfidf_matrix_chunk_{chunk_idx}.npz"
        )
        sp.save_npz(chunk_tfidf_path, chunk_matrix)

        # Save chunk document IDs
        chunk_ids_path = os.path.join(
            output_folder, f"doc_ids_chunk_{chunk_idx}.joblib"
        )
        joblib.dump(chunk_doc_ids, chunk_ids_path)

        # Print chunk stats
        print(f"Chunk {chunk_idx+1} stats:")
        print(f"  - Documents: {len(chunk_doc_ids)}")
        print(f"  - Matrix shape: {chunk_matrix.shape}")
        print(
            f"  - Matrix non-zeros: {chunk_matrix.nnz} ({chunk_matrix.nnz/chunk_matrix.shape[0]:.1f} per document)"
        )
        print(
            f"  - Sparsity: {100 - (100 * chunk_matrix.nnz / (chunk_matrix.shape[0] * chunk_matrix.shape[1])):.2f}%"
        )

        # Clean up
        del chunk_matrices, chunk_matrix, chunk_doc_ids
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    # Free GPU memory
    del vectorizer
    cp._default_memory_pool.free_all_blocks()
    gc.collect()

    # Print final stats
    print(f"✅ TF-IDF index built successfully!")
    print(f"📊 Stats:")
    print(f"  - Total documents: {doc_count}")
    print(f"  - Vocabulary size: {len(vocabulary)}")
    print(f"  - Saved in {num_chunks} chunks")

    return {
        "vocabulary": vocab_path,
        "idf_values": idf_path,
        "num_chunks": num_chunks,
        "doc_ids": doc_ids_file,
    }


if __name__ == "__main__":
    # Configure these parameters based on your system
    processed_folder = "wikipedia_processed"
    output_folder = "wikipedia_index"

    # Build the index
    index_files = build_tfidf_index_gpu(
        processed_folder=processed_folder,
        output_folder=output_folder,
        min_df=3,  # Appear in at least 3 documents
        max_df=0.7,  # Appear in at most 70% of documents
        max_features=150000,  # Keep top 150k terms by frequency (reduced)
        batch_size=2000,  # Process 2k documents at a time (reduced)
        file_batch_size=1,  # Process 1 file at a time to conserve memory
        sample_size=100000,  # Sample 100k documents for vocabulary building
    )

    print(f"📁 Index files saved to: {output_folder}")
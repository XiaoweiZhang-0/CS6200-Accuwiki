import glob

import joblib
import numpy as np
from datasets import load_dataset
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import \
    cosine_similarity as sklearn_cosine_similarity


def search_tfidf_index(query, index_folder="wikipedia_index", top_k=10):
    """
    Search the TF-IDF index for documents most similar to the query.

    Args:
        query (str): Search query text
        index_folder (str): Folder containing the TF-IDF index files
        top_k (int): Number of top results to return

    Returns:
        list: List of (doc_id, score) tuples for top matching documents
    """
    print(f"🔍 Searching for: '{query}'")

    # Load the vocabulary and IDF values
    print("Loading vocabulary and IDF values...")
    vocabulary = joblib.load(f"{index_folder}/vocabulary.joblib")
    idf_values = np.load(f"{index_folder}/idf_values.npy")

    # Make sure the vocabulary is dictionary mapping terms to indices
    if not isinstance(vocabulary, dict):
        raise ValueError("Vocabulary must be a dictionary mapping terms to indices")

    print(f"Vocabulary size: {len(vocabulary)}")

    # Use sklearn's vectorizer instead of cuML to avoid GPU issues
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, use_idf=True, norm="l2")
    # print(len(idf_values[0]), len(vocabulary))
    # Set the IDF values manually - make sure length matches vocabulary size
    if len(idf_values[0]) != len(vocabulary):
        raise ValueError(
            f"IDF values length ({len(idf_values[0])}) doesn't match vocabulary size ({len(vocabulary)})"
        )

    vectorizer.idf_ = idf_values[0]

    # Transform the query to a TF-IDF vector
    print("Transforming query...")
    query_vector_cpu = vectorizer.transform([query])

    # Find all chunk files
    chunk_files = sorted(glob.glob(f"{index_folder}/tfidf_matrix_chunk_*.npz"))
    num_chunks = len(chunk_files)
    print(f"Found {num_chunks} index chunks to search")

    # Track top results across all chunks
    all_scores = []
    all_doc_ids = []

    # Process each chunk
    for chunk_idx in range(num_chunks):
        print(f"Searching chunk {chunk_idx+1}/{num_chunks}...")

        # Load the chunk's TF-IDF matrix
        chunk_matrix_path = f"{index_folder}/tfidf_matrix_chunk_{chunk_idx}.npz"
        chunk_matrix = load_npz(chunk_matrix_path)

        print(f"Chunk matrix shape: {chunk_matrix.shape}")

        # Check matrix dimensions match vocabulary size
        if chunk_matrix.shape[1] != len(vocabulary):
            raise ValueError(
                f"Chunk matrix has {chunk_matrix.shape[1]} features but vocabulary has {len(vocabulary)} terms"
            )

        # Load the chunk's document IDs
        chunk_ids_path = f"{index_folder}/doc_ids_chunk_{chunk_idx}.joblib"
        chunk_doc_ids = joblib.load(chunk_ids_path)

        # Calculate cosine similarity between query and all documents in this chunk
        similarities = sklearn_cosine_similarity(
            query_vector_cpu, chunk_matrix
        ).flatten()

        # Add scores and IDs to our tracking lists
        all_scores.extend(similarities)
        all_doc_ids.extend(chunk_doc_ids)

        # Clean up
        del chunk_matrix, chunk_doc_ids, similarities

    # Find the top K results
    if len(all_scores) > top_k:
        top_indices = np.argsort(all_scores)[-top_k:][
            ::-1
        ]  # Get indices of top K scores in descending order
        top_results = [(all_doc_ids[idx], all_scores[idx]) for idx in top_indices]
    else:
        # If we have fewer results than top_k, sort all of them
        top_indices = np.argsort(all_scores)[::-1]  # Descending order
        top_results = [(all_doc_ids[idx], all_scores[idx]) for idx in top_indices]

    print(f"✅ Search complete. Found {len(top_results)} results.")
    return top_results


if __name__ == "__main__":
    query = "What is the capital of France?"
    results = search_tfidf_index(query=query, index_folder="wikipedia_index", top_k=10)

    # Load the dataset once before the loop
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    # Create a list of doc_ids to filter just once
    doc_ids = [doc_id for doc_id, _ in results]

    # Filter the dataset to get all needed documents at once
    filtered_docs = dataset.filter(lambda x: x["id"] in doc_ids)

    # Create a mapping from doc_id to document for quick lookup
    doc_map = {doc["id"]: doc for doc in filtered_docs}

    print(f"\nTop results for query: {query} are")
    # Now iterate through results without loading the dataset each time
    for i, (doc_id, score) in enumerate(results):
        if doc_id in doc_map:
            doc = doc_map[doc_id]
            print(
                f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}, Title: {doc['title']}"
            )
        else:
            print(f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}, Title: Not found")
import json
import os
import time
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Constants
INDEX_PATH = "wikipedia_faiss_index"  # Update this
INDEX_FILE = "wikipedia.index"
METADATA_FILE = "metadata.txt"
ID_MAPPING_INDEX = "id_mapping_index.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Update with your model name
TOP_K = 10  # Number of results to return


class WikipediaSearcher:
    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path
        self.index = None
        self.id_mapping = {}
        self.model = None
        self.metadata = {}

        # Load index and resources
        self.load_resources()

    def load_resources(self):
        """Load the FAISS index, ID mapping, and embedding model."""
        print("🔄 Loading resources...")
        start_time = time.time()

        # Load FAISS index
        index_file = os.path.join(self.index_path, INDEX_FILE)
        print(f"📂 Loading index from {index_file}...")
        self.index = faiss.read_index(index_file)

        # Load metadata
        metadata_file = os.path.join(self.index_path, METADATA_FILE)
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                lines = f.readlines()
                self.metadata = {
                    line.split(": ")[0]: line.split(": ")[1].strip()
                    for line in lines
                    if ": " in line
                }
            print(f"ℹ️ Index metadata: {self.metadata}")

        # Set nprobe from metadata or default
        if "nprobe" in self.metadata:
            self.index.nprobe = int(self.metadata["nprobe"])
        else:
            self.index.nprobe = 64  # Default value

        # Load ID mapping index
        mapping_index_file = os.path.join(self.index_path, ID_MAPPING_INDEX)
        if os.path.exists(mapping_index_file):
            with open(mapping_index_file, "r") as f:
                mapping_info = json.load(f)

            # Lazy loading approach - we'll only load mappings when needed
            self.mapping_info = mapping_info
            print(
                f"📊 ID mapping: {mapping_info['total_vectors']} vectors in {len(mapping_info['chunks'])} chunks"
            )

        # Load model
        print(f"🧠 Loading embedding model {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        elapsed_time = time.time() - start_time
        print(f"✅ Resources loaded in {elapsed_time:.2f} seconds")

    def get_document_id(self, index: int) -> str:
        """Get the document ID for a given index."""
        # Find which chunk contains this index
        for chunk_key, chunk_file in self.mapping_info["chunks"].items():
            start, end = map(int, chunk_key.split("_"))
            if start <= index < end:
                # Load the chunk if needed
                if not hasattr(self, f"chunk_{start}_{end}"):
                    with open(chunk_file, "r") as f:
                        setattr(self, f"chunk_{start}_{end}", json.load(f))

                # Get the document ID from the chunk
                chunk_data = getattr(self, f"chunk_{start}_{end}")
                return chunk_data.get(str(index), f"Unknown_{index}")

        return f"Unknown_{index}"

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the query.

        Args:
            query: The search query text
            top_k: Number of results to return

        Returns:
            List of results with document IDs and scores
        """
        print(f"🔍 Searching for: {query}")
        start_time = time.time()

        # Generate embedding for the query
        query_vector = self.model.encode([query])[0]
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        # If using METRIC_INNER_PRODUCT, normalize the query vector
        if "METRIC_INNER_PRODUCT" in str(self.index):
            faiss.normalize_L2(query_vector)

        # Search the index
        distances, indices = self.index.search(query_vector, top_k)

        # Process results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                doc_id = self.get_document_id(idx)
                # Convert distance to score based on metric
                score = float(dist)  # Inner product: higher is better

                results.append(
                    {
                        "rank": i + 1,
                        "document_id": doc_id,
                        "score": score,
                        "index": int(idx),
                    }
                )

        elapsed_time = time.time() - start_time
        print(f"✅ Search completed in {elapsed_time:.4f} seconds")

        return results

    def move_to_gpu(self):
        """Move the index to GPU if available."""
        if torch.cuda.is_available():
            print("🚀 Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("✅ Index moved to GPU")
        else:
            print("⚠️ GPU not available, using CPU")

    def print_index_stats(self):
        """Print statistics about the index."""
        print(f"📊 Index statistics:")
        print(f"   - Type: {type(self.index).__name__}")
        print(f"   - Dimension: {self.index.d}")
        print(f"   - nprobe: {self.index.nprobe}")
        print(f"   - ntotal: {self.index.ntotal}")
        if hasattr(self.index, "nlist"):
            print(f"   - nlist: {self.index.nlist}")


# Example usage
def search_wikipedia(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Search the Wikipedia index for the given query.

    Args:
        query: The search query text
        top_k: Number of results to return

    Returns:
        List of results with document IDs and scores
    """
    searcher = WikipediaSearcher()

    # Optionally move to GPU for faster search
    # searcher.move_to_gpu()

    # Print index stats
    searcher.print_index_stats()

    # Perform search
    results = searcher.search(query, top_k)

    # Print results
    print("\n🔎 Search results:")
    for result in results:
        print(
            f"{result['rank']}. {result['document_id']} (Score: {result['score']:.4f})"
        )

    return results


if __name__ == "__main__":
    # Example usage
    query = "quantum computing applications"
    results = search_wikipedia(query)

    # Load the dataset once before the loop
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    # Create a list of doc_ids to filter just once
    doc_ids = [result["document_id"] for result in results]

    # Filter the dataset to get all needed documents at once
    filtered_docs = dataset.filter(lambda x: x["id"] in doc_ids)

    # Create a mapping from doc_id to document for quick lookup
    doc_map = {doc["id"]: doc for doc in filtered_docs}

    print(f"\nTop results for query: {query} are")
    # Now iterate through results without loading the dataset each time
    for i, result in enumerate(results):
        doc_id = result["document_id"]
        score = result["score"]
        if doc_id in doc_map:
            doc = doc_map[doc_id]
            print(
                f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}, Title: {doc['title']}"
            )
        else:
            print(f"{i+1}. Document ID: {doc_id}, Score: {score:.4f}, Title: Not found")
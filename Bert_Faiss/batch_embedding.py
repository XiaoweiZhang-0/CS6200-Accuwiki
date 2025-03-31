# Text chunking function
import numpy as np
from constants import SENTENCE_LENGTH

MAX_LEN = SENTENCE_LENGTH
def chunkify_text(text, max_words=MAX_LEN, overlap=50):
    words = text.split()
    chunks = [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words - overlap)
    ]

    # Ensure last chunk has max_words (padded with spaces if needed)
    if len(chunks[-1].split()) < max_words:
        chunks[-1] += " " * (max_words - len(chunks[-1].split()))

    return chunks


# Process a batch of documents
def process_batch(model, device, batch, max_len=MAX_LEN):
    doc_ids, texts = batch["id"], batch["text"]

    chunked_texts = []
    bacthed_ids = []

    # Chunk all texts first
    for doc_id, text in zip(doc_ids, texts):
        chunks = chunkify_text(text, max_len)
        chunked_texts.extend(chunks)
        bacthed_ids.extend([doc_id] * len(chunks))

    # Encode using GPU-accelerated batch processing
    batch_embeddings = model.encode(
        chunked_texts, batch_size=1024, show_progress_bar=True, device=device, normalize_embeddings=True
    )

    ## confirm the length of batched_ids equal to the length of batch_embeddings
    assert len(bacthed_ids) == batch_embeddings.shape[0]

    # Prepare data to return
    chunked_data = {
        "id": bacthed_ids,
        "embedding": batch_embeddings.tolist(),  # Convert to list for compatibility
    }

    return chunked_data

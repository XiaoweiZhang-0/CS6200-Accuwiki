import os
from gensim import corpora, models, similarities
from tqdm import tqdm

# ========== Paths ==========
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
RESULT_FILE = os.path.join(DESKTOP_PATH, "top100_results.txt")

# ========== Ask for document folder path ==========
print("📂 Please enter the full path to the folder containing the documents:")
DATA_PATH = input("Document folder path: ").strip()

# ========== Check if the path exists ==========
if not os.path.exists(DATA_PATH):
    print(f"❌ Folder '{DATA_PATH}' does not exist. Please check and try again.")
    exit()

# ========== Read up to 1000 documents ==========
documents = []
file_list = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.txt')])[:1000]

print(f"📖 Reading {len(file_list)} documents...")
for filename in tqdm(file_list, desc="Reading Files"):
    file_path = os.path.join(DATA_PATH, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        documents.append(file.read())

# ========== Generate TF-IDF Model ==========
texts = [doc.lower().split() for doc in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)

# ========== Build Similarity Index ==========
index = similarities.MatrixSimilarity(tfidf[corpus])

# ========== Retrieve Top 100 Documents ==========
def retrieve_top_100(query):
    """Retrieve top 100 most relevant documents (or fewer if not enough documents)"""

    # Convert query to TF-IDF vector
    query_bow = dictionary.doc2bow(query.lower().split())
    query_tfidf = tfidf[query_bow]

    # Compute similarities
    sims = index[query_tfidf]

    # Get Top 100 (or all if fewer than 100 documents exist)
    top_n = sorted(enumerate(sims), key=lambda x: -x[1])[:100]

    # Save results to desktop
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        for rank, (doc_id, score) in tqdm(enumerate(top_n, start=1), desc="Writing Results"):
            f.write(f"Rank {rank}: Document {file_list[doc_id]}, Score={score:.4f}\n")

    print(f"✅ Top 100 results saved to: {RESULT_FILE}")

# ========== Main Process ==========
if __name__ == "__main__":
    query = input("Enter your search query: ").strip()
    retrieve_top_100(query)

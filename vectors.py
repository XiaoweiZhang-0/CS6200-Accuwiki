import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 文档目录和保存路径
docs_path = "/Users/wangeiden/Desktop/wikipedia_processed"
tfidf_save_path = "/Users/wangeiden/Desktop/tfidf_vectors.npz"
vectorizer_save_path = "/Users/wangeiden/Desktop/tfidf_vectorizer.pkl"

# 读取所有文本文件
def load_documents(directory):
    documents = []
    filenames = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
            filenames.append(file)
    return documents, filenames

# 计算 TF-IDF 向量并保存
def compute_tfidf():
    documents, filenames = load_documents(docs_path)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 保存 TF-IDF 矩阵和文件名
    np.savez(tfidf_save_path, tfidf_matrix=tfidf_matrix.toarray(), filenames=filenames)
    
    # 保存 TF-IDF 向量化器
    with open(vectorizer_save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"TF-IDF 计算完成，结果保存在 {tfidf_save_path}")

# 查询最相关文档
def search_query(query):
    # 载入 TF-IDF 数据
    data = np.load(tfidf_save_path, allow_pickle=True)
    tfidf_matrix = data["tfidf_matrix"]
    filenames = data["filenames"]
    
    # 载入 TF-IDF 向量化器
    with open(vectorizer_save_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    # 计算查询的 TF-IDF 向量
    query_vector = vectorizer.transform([query]).toarray()
    
    # 计算余弦相似度
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # 获取前 100 个最相关文档
    top_indices = np.argsort(similarities)[::-1][:100]
    
    # 输出结果
    print("Top 100 relevant documents:")
    for i in top_indices:
        print(f"{filenames[i]} - Similarity: {similarities[i]:.4f}")

# 计算 TF-IDF
compute_tfidf()

# 示例查询
query = "machine learning algorithms"
search_query(query)

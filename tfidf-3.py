import os
import time
import numpy as np
from tqdm import tqdm  # 导入进度条模块
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

# 📂 设定文件路径
data_dir = "/Users/wangeiden/Desktop/wikipedia_processed"
output_dir = "/Users/wangeiden/Desktop/tfidf_output"  # 存储输出文件的文件夹
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f"{output_dir}/tfidf_log_{timestamp}.txt"

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".arrow.txt")])

# 分批次处理，每批处理7个文件
batch_size = 7
num_batches = len(file_list) // batch_size + (1 if len(file_list) % batch_size > 0 else 0)

# 用于记录处理的文档数和特征数
total_documents = 0
total_features = 0

# 逐批处理文件
for batch_num in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
    batch_start = batch_num * batch_size
    batch_end = min((batch_num + 1) * batch_size, len(file_list))
    current_files = file_list[batch_start:batch_end]

    # 读取数据并计算 TF-IDF
    documents = []
    print(f"\n📂 正在处理第 {batch_num + 1} 批文件，共 {len(current_files)} 个文件...")
    for file_name in current_files:
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    documents.append(line.strip()[:1000])  # 只取每行的前1000个字符
        except Exception as e:
            print(f"❌ 错误: 无法读取文件 {file_name} - {e}")
    
    # 计算 TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 归一化
    normalizer = Normalizer()
    tfidf_matrix = normalizer.transform(tfidf_matrix)
    
    # 保存结果到输出文件夹
    output_file = f"{output_dir}/tfidf_batch_{batch_num + 1}_{timestamp}.npz"
    from scipy.sparse import save_npz
    save_npz(output_file, tfidf_matrix)

    # 记录文档和特征数
    total_documents += len(documents)
    total_features = tfidf_matrix.shape[1]

    print(f"✅ 批次 {batch_num + 1} 处理完成，TF-IDF 矩阵大小: {tfidf_matrix.shape}. 结果保存至 {output_file}.")

    # 清理内存
    del documents
    del tfidf_matrix
    del vectorizer
    del normalizer
    import gc
    gc.collect()  # 显式调用垃圾回收

# 记录日志
with open(log_file, "w", encoding="utf-8") as log_f:
    log_f.write(f"TF-IDF 计算完成，共 {total_documents} 个文档，{total_features} 个特征。\n")
    log_f.write(f"处理的文件共 {len(file_list)} 个文件。\n")
    log_f.write(f"所有输出结果已保存至 {output_dir} 文件夹。\n")

print(f"\n✅ 所有批次处理完成，日志已保存至 {log_file}。")

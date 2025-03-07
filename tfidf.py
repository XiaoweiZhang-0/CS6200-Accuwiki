import os
import re
import math
import ssl
import nltk
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm  # ✅ 进度条库
import gc  # ✅ 强制释放内存

# ✅ 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ 下载 NLTK 资源（避免报错）
nltk.download("stopwords")

# ✅ 目标文件夹
input_folder = "/Users/wangeiden/Desktop/wikipedia_processed"
output_folder = "/Users/wangeiden/Desktop/tfidf"  # ✅ 结果保存目录
os.makedirs(output_folder, exist_ok=True)  # ✅ 确保目录存在

# ✅ **加载并处理单个文件**
def process_single_file(file_path):
    """ 读取单个文件并计算 TF-IDF """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()  # 一次性读取整个文件，提高 I/O 速度
            text = re.sub(r'\*\*\* START OF .*? \*\*\*', '', text)
            text = re.sub(r'\*\*\* END OF .*? \*\*\*', '', text)
            words = re.findall(r'\b[a-z]{3,}\b', text)  # 提取至少 3 个字母的单词
    except Exception as e:
        print(f"⚠️ 读取 {file_path} 失败: {e}")
        return None
    
    if not words:
        print(f"⚠️ 文件 {file_path} 没有有效单词，跳过处理。")
        return None

    # ✅ 计算 TF
    word_count = Counter(words)
    total_words = len(words)
    tf_map = {word: count / total_words for word, count in word_count.items()}

    return words, tf_map, word_count

# ✅ **计算 DF**
def compute_df(df_map, words):
    """ 更新 DF 统计 """
    for word in set(words):
        df_map[word] += 1

# ✅ **计算 IDF**
def compute_idf(df_map, num_docs):
    return {word: math.log((num_docs + 1) / (df + 1)) + 2.0 for word, df in df_map.items()}

# ✅ **计算 TF-IDF**
def compute_tfidf(tf_map, idf_map):
    return {word: tf_map[word] * idf_map.get(word, 0) for word in tf_map}

# ✅ **过滤低 TF-IDF 词**
def filter_low_tfidf(tfidf_map, min_score=0.0001):
    return {word: score for word, score in tfidf_map.items() if score >= min_score}

# ✅ **获取停用词**
def get_stopwords(word_count, total_words):
    nltk_stopwords = set(stopwords.words("english"))
    sklearn_stopwords = set(ENGLISH_STOP_WORDS)
    auto_high_freq_stopwords = {word for word, count in word_count.items() if count > 0.02 * total_words}
    custom_stopwords = {
        "said", "one", "would", "could", "like", "got", "went", "well", "voice", "work", "works",
        "must", "know", "little", "see", "may", "yet", "also", "upon", "way", "even",
        "another", "thing", "two", "three", "many", "first", "last", "every", "though",
        "began", "white", "time", "back", "good", "hand", "head", "make", "made",
        "look", "long", "came", "day", "thought", "give", "just", "think", "let",
        "sure", "right", "left", "around", "really", "next", "come", "put"
    }
    return nltk_stopwords.union(sklearn_stopwords, auto_high_freq_stopwords, custom_stopwords)

# ✅ **过滤停用词**
def filter_stopwords(tfidf_map, stopword_list):
    return {word: score for word, score in tfidf_map.items() if word.lower() not in stopword_list}

# ✅ **主函数**
def main():
    print("📂 正在逐个文件处理文本...")

    file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
    if not file_list:
        print("⚠️ 目录为空，没有找到 `.txt` 文件！")
        return

    num_docs = 0  # 统计总文档数
    df_map = defaultdict(int)  # 记录 DF

    # ✅ **第一遍遍历所有文件，统计 DF**
    for filename in tqdm(file_list, desc="📊 统计 DF", unit="file"):
        file_path = os.path.join(input_folder, filename)
        result = process_single_file(file_path)
        if result:
            words, _, _ = result
            compute_df(df_map, words)
            num_docs += 1
        
        gc.collect()  # ✅ 强制释放内存，避免内存泄漏

    # ✅ **计算 IDF**
    idf_map = compute_idf(df_map, num_docs)

    # ✅ **第二遍遍历所有文件，计算 TF-IDF 并写入文件**
    for filename in tqdm(file_list, desc="📊 计算 TF-IDF", unit="file"):
        file_path = os.path.join(input_folder, filename)
        result = process_single_file(file_path)
        if not result:
            continue

        words, tf_map, word_count = result
        tfidf_map = compute_tfidf(tf_map, idf_map)

        # ✅ **获取并应用停用词**
        stopword_list = get_stopwords(word_count, len(words))
        tfidf_map = filter_stopwords(filter_low_tfidf(tfidf_map), stopword_list)

        # ✅ **排序 TF-IDF 结果**
        sorted_tfidf = sorted(tfidf_map.items(), key=lambda x: x[1], reverse=True)

        # ✅ **打印所有 TF-IDF 词**
        print(f"\n📄 文件 {filename} 处理完成，所有 TF-IDF 词：")
        for word, score in sorted_tfidf:
            print(f"{word}: {score:.6f}")  # ✅ 更高精度

        # ✅ **保存完整 TF-IDF 结果到文件**
        output_path = os.path.join(output_folder, f"{filename}_tfidf.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"📄 文件名: {filename}\n")
            f.write(f"📊 处理后 TF-IDF 关键词:\n")
            for word, score in sorted_tfidf:
                f.write(f"{word}: {score:.6f}\n")  # ✅ 保持 6 位小数精度

        print(f"✅ 结果已完整保存至: {output_path}")

        # ✅ **释放当前文件的变量**
        del words, tf_map, word_count, tfidf_map
        gc.collect()

    print("\n🎉 **所有文件处理完成！**")

# ✅ **运行程序**
if __name__ == "__main__":
    main()

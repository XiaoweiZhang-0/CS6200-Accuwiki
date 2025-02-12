import os
import re
import ssl
import nltk
from datasets import load_dataset
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ 下载 NLTK 资源（确保没有缺失）
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ 直接指定路径
input_folder = "/Users/wangeiden/Desktop/wikipedia_"
output_folder = "/Users/wangeiden/Desktop/wikipedia_processed/"

# ✅ 创建存储目录
os.makedirs(output_folder, exist_ok=True)

# ✅ 初始化工具
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # ✅ 预处理停用词（用 `set()` 提高查找速度）

# ✅ **极速 `preprocess_text()`**
def preprocess_text(text):
    text = text.lower()  # 🔹 转小写
    text = re.sub(r"[^\w\s]", "", text)  # 🔹 去标点

    words = text.split()  # 🔹 **比 `word_tokenize()` 更快**
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # 🔹 **批量 Lemmatization + 去停用词**

    return " ".join(words)  # 🔹 **最快拼接**

# ✅ **获取所有 `.arrow` 文件**
arrow_files = [f for f in os.listdir(input_folder) if f.endswith(".arrow")]
if not arrow_files:
    raise ValueError(f"目录 {input_folder} 中没有 Arrow 文件！")

# ✅ **逐个处理每个 `.arrow` 文件**
for arrow_file in tqdm(arrow_files, desc="📂 处理文件进度", unit="file"):
    input_path = os.path.join(input_folder, arrow_file)
    output_path = os.path.join(output_folder, f"processed_{arrow_file}.txt")

    print(f"\n📂 开始处理文件: {input_path}")

    try:
        dataset = load_dataset("arrow", data_files=input_path, split="train")

        if "text" in dataset.column_names:
            texts = dataset["text"]
            
            # ✅ **🔥 高速批量处理文本**
            processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="✅ 处理文本", unit="条")]

            # ✅ **仅保存处理后的文本**
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(processed_texts))

            print(f"🎉 **文件 {arrow_file} 处理完成，结果已保存至: {output_path}**")

        else:
            print(f"⚠️ 文件 {arrow_file} 中没有 'text' 列，跳过！")

    except Exception as e:
        print(f"❌ 处理 {arrow_file} 失败: {e}")

print(f"\n🎉 **所有文件处理完成！**")
print(f"📁 预处理结果已存放在: {output_folder}")

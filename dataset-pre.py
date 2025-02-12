import os
import re
import ssl
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tqdm import tqdm

# ✅ 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ 下载 NLTK 资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ✅ **文件路径**
input_folder = "/Users/wangeiden/Desktop/wikipedia_"  # 输入 `.arrow` 文件的文件夹
output_folder = "/Users/wangeiden/Desktop/wikipedia_processed/"  # 处理后文件存放的文件夹

# ✅ **创建存储目录**
os.makedirs(output_folder, exist_ok=True)

# ✅ **初始化工具**
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # 用 `set()` 提高查找速度

# ✅ **词性映射 (提高 Lemmatization 准确度)**
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # 获取POS标注的首字母
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # 默认作为名词处理

# ✅ **修正 contractions（避免 omens 错误）**
contractions = {
    "women's": "woman",
    "can't": "cannot",
    "isn't": "is not",
    "doesn't": "does not",
    "won't": "will not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
}

def expand_contractions(text):
    words = text.split()
    words = [contractions[word] if word in contractions else word for word in words]
    return " ".join(words)

# ✅ **文本预处理**
def preprocess_text(text):
    text = expand_contractions(text)  # **先处理 contractions**
    text = text.lower()  # **转小写**
    text = re.sub(r"[^\w\s]", "", text)  # **去除标点**
    words = word_tokenize(text)  # **分词**
    
    # **🔥 词形还原 + 词性标注**
    processed_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))  # **确保 `known` 变成 `know`**
        for word in words if word not in stop_words
    ]
    return " ".join(processed_words)  # **最快拼接**

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
            
            # ✅ **批量处理文本**
            processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="✅ 处理文本", unit="条")]

            # ✅ **保存处理后的文本**
            with open(output_path, "w", encoding="utf-8") as f:
                for raw, processed in zip(texts, processed_texts):
                    f.write(f"原始文本:\n{raw}\n")
                    f.write(f"处理后文本:\n{processed}\n")
                    f.write("=" * 100 + "\n")

            print(f"🎉 **文件 {arrow_file} 处理完成，结果已保存至: {output_path}**")

        else:
            print(f"⚠️ 文件 {arrow_file} 中没有 'text' 列，跳过！")

    except Exception as e:
        print(f"❌ 处理 {arrow_file} 失败: {e}")

print(f"\n🎉 **所有文件处理完成！**")
print(f"📁 预处理结果已存放在: {output_folder}")
import os
import re
import ssl
import nltk
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tqdm import tqdm

# ✅ 解决 SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ 下载 NLTK 资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ✅ **文件路径**
input_folder = "/Users/wangeiden/Desktop/wikipedia_"  # 输入 `.arrow` 文件的文件夹
output_folder = "/Users/wangeiden/Desktop/wikipedia_processed/"  # 处理后文件存放的文件夹

# ✅ **创建存储目录**
os.makedirs(output_folder, exist_ok=True)

# ✅ **初始化工具**
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # 用 `set()` 提高查找速度

# ✅ **词性映射 (提高 Lemmatization 准确度)**
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # 获取POS标注的首字母
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # 默认作为名词处理

# ✅ **修正 contractions（避免 omens 错误）**
contractions = {
    "women's": "woman",
    "can't": "cannot",
    "isn't": "is not",
    "doesn't": "does not",
    "won't": "will not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
}

def expand_contractions(text):
    words = text.split()
    words = [contractions[word] if word in contractions else word for word in words]
    return " ".join(words)

# ✅ **文本预处理**
def preprocess_text(text):
    text = expand_contractions(text)  # **先处理 contractions**
    text = text.lower()  # **转小写**
    text = re.sub(r"[^\w\s]", "", text)  # **去除标点**
    words = word_tokenize(text)  # **分词**
    
    # **🔥 词形还原 + 词性标注**
    processed_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))  # **确保 `known` 变成 `know`**
        for word in words if word not in stop_words
    ]
    return " ".join(processed_words)  # **最快拼接**

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
            
            # ✅ **批量处理文本**
            processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="✅ 处理文本", unit="条")]

            # ✅ **保存处理后的文本**
            with open(output_path, "w", encoding="utf-8") as f:
                for raw, processed in zip(texts, processed_texts):
                    f.write(f"原始文本:\n{raw}\n")
                    f.write(f"处理后文本:\n{processed}\n")
                    f.write("=" * 100 + "\n")

            print(f"🎉 **文件 {arrow_file} 处理完成，结果已保存至: {output_path}**")

        else:
            print(f"⚠️ 文件 {arrow_file} 中没有 'text' 列，跳过！")

    except Exception as e:
        print(f"❌ 处理 {arrow_file} 失败: {e}")

print(f"\n🎉 **所有文件处理完成！**")
print(f"📁 预处理结果已存放在: {output_folder}")

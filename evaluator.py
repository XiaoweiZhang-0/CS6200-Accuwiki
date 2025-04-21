from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



def detect_hallucination(claim, evidence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    premise = evidence
    hypothesis = claim

    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    probabilities = torch.softmax(output["logits"][0], -1).tolist()
    pred_label = probabilities.index(max(probabilities))  # Get index of max value
    labels = {0:"supports", 1: "neutral", 2: "contradiction"}

    return labels[pred_label]

def get_answer_text_pair(file_path):
    #csv file format: answer, text
    answer_list = []
    text_list = []
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # # Limit to first 433 rows
    # df = df.head(433)

    # Extract 'answer' and 'text' columns as lists
    answer_list = df["answer"].tolist()
    text_list = df["text"].tolist()
    return answer_list, text_list

import unicodedata
import re

def normalize(text: str) -> str:
    # Normalize Unicode characters (e.g., é → é)
    text = unicodedata.normalize("NFKC", str(text))
    
    # Remove extra spaces and newlines
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # (Optional) Replace fancy quotes/dashes with standard ones
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    
    return text

if __name__ == "__main__":
    # load answer and text of deepseek_with_rag.csv and deepseek_without_rag.csv
    # with_rag_file_path = "../data/deepseek_with_rag.csv"
    # without_rag_file_path = "../data/deepseek_without_rag.csv"
    with_rag_file_path = "../data/gpt4o-mini_with_rag.csv"
    without_rag_file_path = "../data/gpt4o-mini_without_rag.csv"
    with_rag_answer_list, with_rag_text_list = get_answer_text_pair(with_rag_file_path)
    without_rag_answer_list, without_rag_text_list = get_answer_text_pair(without_rag_file_path)
    
    with_rag_labels = []
    without_rag_labels = []

    # detect hallucination
    for i in range(len(with_rag_answer_list)):
        print(f"verifying {i}th pair")
        with_rag_answer = with_rag_answer_list[i]
        with_rag_text = with_rag_text_list[i]
        without_rag_answer = without_rag_answer_list[i]
        without_rag_text = without_rag_text_list[i]
        with_rag_label = detect_hallucination(normalize(with_rag_answer), normalize(with_rag_text))
        without_rag_label = detect_hallucination(normalize(without_rag_answer), normalize(without_rag_text))
        with_rag_labels.append(with_rag_label)
        without_rag_labels.append(without_rag_label)

    # print("with_rag_labels: ", with_rag_labels)
    # print("without_rag_labels: ", without_rag_labels)
    # supports and neutral count as correct, calculate acc, precision, recall, f1
    # with_rag_correct = with_rag_labels.count("supports") + with_rag_labels.count("neutral")
    with_rag_correct = with_rag_labels.count("supports")
    print("with_rag_correct: ", with_rag_correct)
    # without_rag_correct = without_rag_labels.count("supports") + without_rag_labels.count("neutral")
    without_rag_correct = without_rag_labels.count("supports")
    print("without_rag_correct: ", without_rag_correct)
    with_rag_acc = with_rag_correct / len(with_rag_labels)
    without_rag_acc = without_rag_correct / len(without_rag_labels)
    with_rag_precision = with_rag_correct / (with_rag_labels.count("supports") + with_rag_labels.count("neutral"))
    without_rag_precision = without_rag_correct / (without_rag_labels.count("supports") + without_rag_labels.count("neutral"))
    with_rag_recall = with_rag_correct / len(with_rag_labels)
    without_rag_recall = without_rag_correct / len(without_rag_labels)
    with_rag_f1 = 2 * with_rag_precision * with_rag_recall / (with_rag_precision + with_rag_recall)
    without_rag_f1 = 2 * without_rag_precision * without_rag_recall / (without_rag_precision + without_rag_recall)  
    print("without RAG:")
    print(" acc: ", without_rag_acc)
    print(" precision: ", without_rag_precision)
    print(" recall: ", without_rag_recall)
    print(" f1: ", without_rag_f1)
    print("--------------------------------")
    print("with RAG:")
    print(" acc: ", with_rag_acc)
    print(" precision: ", with_rag_precision)
    print(" recall: ", with_rag_recall)
    print(" f1: ", with_rag_f1)
    


# test
# claim = 'Beijing is the capital city of China'
# evidence = "Beijing is the capital city of China"
# detect_hallucination(claim,evidence)
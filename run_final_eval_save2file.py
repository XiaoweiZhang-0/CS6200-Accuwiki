import pandas as pd
from generator import query_generator
from Bert_Faiss.rank import WikipediaSearcher
import requests
from urllib.parse import urlparse, parse_qs
import json 
import os
import lmdb
from collections import defaultdict

def title_from_url(url):
    # Parse the URL
    parsed = urlparse(url)
    
    # Check domain
    if "wikipedia.org" not in parsed.netloc:
        return None  # Not a Wikipedia link
    
    # Possible forms:
    # 1) /wiki/Page_Name
    # 2) /w/index.php?title=Page_Name (& oldid=123)
    path_parts = parsed.path.strip("/").split("/")
    
    # /wiki/Page_Name case
    if len(path_parts) > 1 and path_parts[0] == "wiki":
        page_name = path_parts[1]
    
    # /w/index.php?title=Page_Name case
    if len(path_parts) > 0 and path_parts[0] == "w":
        query_params = parse_qs(parsed.query)
        if "title" in query_params:
            page_name = query_params["title"][0]
    return page_name

def get_bert_result(query):
    BF_rank = WikipediaSearcher()
    BF_scores = BF_rank.search(query)
    bf_dict = defaultdict(lambda: 0, {r["document_id"]: r["score"] for r in BF_scores})# [{}]
    top_10_docs = [doc_id for doc_id, _ in sorted(bf_dict.items(), key=lambda x: x[1], reverse=True)]
    return top_10_docs

# load NQ, filer first 10 query begining with "what", save query-url pair as parameter
def get_query_url_pair():
    # Check if filtered data already exists
    filtered_data_path = "../data/filtered_questions.json"
    
    if os.path.exists(filtered_data_path):
        # Load from saved file
        with open(filtered_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["query_list"], data["url_list"]
    
    # If not, filter and save
    query_list = []
    url_list = []
    # load Natural Qustion dataset
    nq_file_path = "..//data//v1.0-simplified_simplified-nq-train.jsonl//simplified-nq-train.jsonl"
    # filter first 1000 query begining with "what"
    with open(nq_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data["question_text"].startswith("what"):
                query_list.append(data["question_text"])
                url_list.append(data["document_url"])
                if len(query_list) == 1000:
                    break
    
    # Save filtered data to file
    os.makedirs(os.path.dirname(filtered_data_path), exist_ok=True)
    with open(filtered_data_path, 'w', encoding='utf-8') as f:
        json.dump({"query_list": query_list, "url_list": url_list}, f)
    
    return query_list, url_list
    
# Get plain text from wikipedia url
def get_plain_text(url):
    try:
        # extract title from url
        title = title_from_url(url)
        if not title:
            print(f"Warning: No title found in URL: {url}")
            return "I didn't find evidence for this question"
            
        # get plain text from wikipedia via wikipedia api
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(url)
        data = response.json()
        
        # Check if 'extract' key exists, otherwise return a default value
        if 'extract' in data:
            return data["extract"]
        else:
            print(f"Warning: No 'extract' key found in response for URL: {url}")
            return "I didn't find evidence for this question"
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return "I didn't find evidence for this question"
    
# get url from doc id
def get_url_from_doc_ids(doc_ids):
    urls = []
    lmdb_path = "id_to_url.lmdb"
    env = lmdb.open(lmdb_path, map_size=10**9)
    with env.begin(write=False) as txn:
        for doc_id in doc_ids:
            url = txn.get(str(doc_id).encode())  # get URL from LMDB
            url = url.decode('utf-8') if url else None
            urls.append(url)
    env.close()
    return urls
# Run eval and save answer-text to file
def run_eval(model, mode):
    wiki_searcher = WikipediaSearcher()
    query_list, url_list = get_query_url_pair()
    # print(query_list)
    # print(url_list)
    results = []

    for i in range(0, len(query_list)):
        try:
            query = query_list[i]
            url = url_list[i]
            if mode == "with_rag":
                evidence_ids = get_bert_result(query)
                evidence_urls = get_url_from_doc_ids(evidence_ids)
                evidence = ""
                for url in evidence_urls:
                    evidence += get_plain_text(url)
                evidence = evidence.replace("\n", " ")
                evidence = evidence.replace("\r", " ")
                evidence = evidence.replace("\t", " ")
                evidence = evidence.replace("\f", " ")
                evidence = evidence.replace("\v", " ")
                evidence = evidence.replace("\b", " ")
                evidence = evidence.replace("\a", " ")
            else:
                evidence = None
            print(f"retrieved evidence: {evidence}")
            
            print(f"--requesting {i}th query: {query}")
            answer = query_generator(query, model, evidence)
            print(f"answer: {answer}")

            text = get_plain_text(url)
            
            results.append({
                "answer": answer, 
                "text": text
            })
        except Exception as e:
            print(f"Error processing query {i}: {str(e)}")
            # Add a placeholder result to maintain the same number of entries
            results.append({
                "answer": "Error occurred during processing",
                "text": "No content"
            })

    filename = f"../data/{model}_{mode}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"save successfully: {filename}")

if __name__ == "__main__":
    # run_eval("deepseek", "without_rag")  
    run_eval("deepseek", "with_rag")    
    run_eval("gpt4o-mini", "without_rag")
    run_eval("gpt4o-mini", "with_rag")

# send query to Generator and get response
from openai import OpenAI

from Bert_Faiss.rank import search_wikipedia

from datasets import load_dataset

from api_keys import GENERATOR_DICT
def query_generator(query, generator_model, evidence=None):
    if not query:
        raise ValueError("Query cannot be empty")
    if evidence:
        query = f"Answer this question: {query} with evidence from Wikipedia: {evidence}"
    else:
        query = f"Answer this question: {query}"

    # Retrieve query from user
    generator_dict = GENERATOR_DICT

    if generator_model == "deepseek":
        # Use OpenAI API
        client = OpenAI(api_key=generator_dict["deepseek"]['api_key'], base_url=generator_dict["deepseek"]['base_url'])
        model = "deepseek-reasoner"
    else:
        client = OpenAI(api_key=generator_dict["chatgpt"]['api_key'])
        model = "gpt-4o-mini"

    # Query without retrieved documents
    response = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query},
        ],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = input("Enter your query: ")
    generator_model = "chatgpt"
    response = query_generator(query, generator_model)
    print(f"Response without RAG: {response}")
    evidence = search_wikipedia(query)
    response = query_generator(query, generator_model, evidence)
    print(f"Response with RAG: {response}")
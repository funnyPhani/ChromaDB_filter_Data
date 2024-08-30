# ChromaDB_filter_Data


```pyhton

import chromadb
import ollama
from rich import print
def add_documents_to_chromadb(documents, db_path):
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name="docs")
    except:
        collection = client.create_collection(name="docs")

    for i, doc in enumerate(documents):
        content = doc["content"]
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
        embedding = response["embedding"]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"dept": doc["dept"]}]
        )

def search_department_in_chromadb(db_path, department, query, n_results):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("docs")
    
    response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    query_embedding = response["embedding"]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"dept": department}
    )
    
    return results


documents = [
    {"dept": "HR", "content": "HR is responsible for recruiting new employees."},
    {"dept": "HR", "content": "HR handles employee benefits and payroll."},
    {"dept": "IT", "content": "IT manages the company’s internal network."},
    {"dept": "IT", "content": "IT provides technical support to employees."},
    {"dept": "Finance", "content": "Finance manages the company’s budget."},
    {"dept": "Finance", "content": "Finance oversees all financial transactions."},
    {"dept": "Marketing", "content": "Marketing handles the promotion of products."},
    {"dept": "Marketing", "content": "Marketing analyzes market trends and competitors."}
]

db_path = "docs/"
add_documents_to_chromadb(documents, db_path)
department = "IT"
query = "network"
n_results = 2
search_results = search_department_in_chromadb(db_path, department, query,n_results)
print(search_results)


```

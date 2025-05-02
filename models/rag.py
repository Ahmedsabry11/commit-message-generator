from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import pandas as pd

# Custom wrapper to use with LangChain's FAISS
class LocalSentenceTransformer(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

dataset_path = "dataset/commitbench.csv"
# Load the dataset
df = pd.read_csv(dataset_path)

# Drop rows with missing language info if needed
df = df.dropna(subset=['diff_languages'])

# extract the relevant columns
df = df[['diff', 'message']].dropna()
# Convert the DataFrame to a list of Document objects
documents = [
    Document(page_content=row['diff'], metadata={"commit_msg": row['message']})
    for _, row in df.iterrows()
]
# Initialize the embedding model and FAISS vector store
embedding = LocalSentenceTransformer()
vectorstore = FAISS.from_documents(documents, embedding)

class RAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_similar_context(self, diff, k=3):
        results = self.retrieve(diff, k=k)
        return "\n\n".join(
            f"Diff:\n{doc.page_content}\nMessage:\n{doc.metadata['commit_msg']}"
            for doc in results
        )

# # Example documents (diffs)
# documents = [
#     Document(page_content="diff --git a/foo.py b/foo.py\n+def add(x, y): return x + y", metadata={"commit_msg": "Add addition function"}),
#     Document(page_content="diff --git a/bar.py b/bar.py\n+def sub(x, y): return x - y", metadata={"commit_msg": "Add subtraction function"}),
# ]

# embedding = LocalSentenceTransformer()
# vectorstore = FAISS.from_documents(documents, embedding)

# # Retrieve top similar diffs
# query = "diff --git a/foo.py b/foo.py\n+def multiply(x, y): return x * y"
# results = vectorstore.similarity_search(query, k=2)

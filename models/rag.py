import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple,Union

class RAG:
    def __init__(self, index_path: str, documents: Union[List[Tuple[str, str]], str], model_name='all-MiniLM-L6-v2', create_index=False, dataset_path=None,use_gpu=False):
        self.use_gpu = use_gpu
        self.model = SentenceTransformer(model_name)
        if create_index:
            self.create_index(dataset_path,index_path, documents)
        else:
            # # Move to GPU
            # self.index = faiss.read_index(index_path)
            # Load CPU index
            # cpu_index = faiss.read_index(index_path)
            # # Move to GPU
            # self.index = self._move_index_to_gpu(cpu_index)

            # Load CPU index
            cpu_index = faiss.read_index(index_path)
            # Move to GPU if use_gpu is True else keep it on CPU
            self.index = self._move_index_to_gpu(cpu_index) if use_gpu else cpu_index

            self.documents = documents  # List of (diff, commit_msg)
            if isinstance(documents, str):
                with open(documents, "rb") as f:
                    self.documents = pickle.load(f)

    def retrieve(self, query: str, k: int = 5, ignore_first = True) -> List[Tuple[str, str]]:
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        # return [self.documents[i] for i in indices[0]]
        return [(self.documents[i][0], self.documents[i][1]) for i in indices[0] if i != 0] if ignore_first else [(self.documents[i][0], self.documents[i][1]) for i in indices[0]]
    def retrieve_similar_context(self, diff: str, k=3) -> str:
        results = self.retrieve(diff, k)
        return "\n\n".join(
            f"Diff:\n{doc[0]}\nMessage:\n{doc[1]}" for doc in results
        )
    
    def create_index(self,dataset_path :str, index_path: str, save_documents_path: str):
        df = pd.read_csv(dataset_path)
        df = df.dropna(subset=['diff_languages'])
        df = df[['diff', 'message']].dropna()
        texts = df['diff'].tolist()
        metas = df['message'].tolist()
        self.documents = list(zip(texts, metas))
        with open(save_documents_path, "wb") as f:
            pickle.dump(self.documents, f)
        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True).astype("float32")
        dimension = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dimension)
        cpu_index.add(embeddings)
        # faiss.write_index(cpu_index, index_path)
        # self.index = faiss.read_index(index_path)
        # # Move to GPU
        # self.index = self._move_index_to_gpu(cpu_index)

        # Move to GPU if use_gpu is True else keep it on CPU
        self.index = self._move_index_to_gpu(cpu_index) if self.use_gpu else cpu_index

        # Save only the CPU version (for reuse later)
        faiss.write_index(cpu_index, index_path)
        


    def load_documents(self, documents_path: str):
        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)
    def save_documents(self, documents_path: str):
        with open(documents_path, "wb") as f:
            pickle.dump(self.documents, f)
    def load_index(self, index_path: str):
        self.index = faiss.read_index(index_path)

    def _move_index_to_gpu(self, index):
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        return gpu_index


if __name__ == "__main__":
    print("RAG model initialized.")

    # ########################################################################################################
    # ################################################ JAVA ##################################################
    # ########################################################################################################    
    # # run index for java
    # print("Creating index for Java dataset...")
    # index_path = "faiss_index_java.index"
    # dataset_path = "dataset/java.csv"
    # save_documents_path = "documents_java.pkl"
    # rag = RAG(index_path=index_path, documents=save_documents_path, create_index=True, dataset_path=dataset_path)


    #######################################################################################################
    ############################################### Python ################################################
    #######################################################################################################
    print("Creating index for Python dataset...")    
    # run index for python
    index_path = "faiss_index_py.index"
    dataset_path = "dataset/py.csv"
    save_documents_path = "documents_py.pkl"
    rag = RAG(index_path=index_path, documents=save_documents_path, create_index=True, dataset_path=dataset_path, use_gpu=True)

    ########################################################################################################
    ############################################# Javascript ###############################################
    #######################################################################################################
    # print("Creating index for JavaScript dataset...")    
    # # run index for javascript
    # index_path = "faiss_index_js.index"
    # dataset_path = "dataset/js.csv"
    # # dataset_path = "dataset/js_sample.csv"
    # save_documents_path = "documents_js.pkl"
    # rag = RAG(index_path=index_path, documents=save_documents_path, create_index=True, dataset_path=dataset_path, use_gpu=True)

    # ########################################################################################################
    # ################################################# PHP ##################################################
    # ########################################################################################################
    # print("Creating index for PHP dataset...")    
    # # run index for php
    # index_path = "faiss_index_php.index"
    # dataset_path = "dataset/php.csv"
    # save_documents_path = "documents_php.pkl"
    # rag = RAG(index_path=index_path, documents=save_documents_path, create_index=True, dataset_path=dataset_path)

# # Load dataset
# df = pd.read_csv("dataset/java.csv")
# df = df.dropna(subset=['diff_languages'])
# df = df[['diff', 'message']].dropna()

# texts = df['diff'].tolist()
# metas = df['message'].tolist()

# documents = list(zip(texts, metas))

# # Create embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# # Create FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # Save index
# faiss.write_index(index, "faiss_index_java.index")

# # Save documents
# import pickle
# with open("documents_java.pkl", "wb") as f:
#     pickle.dump(documents, f)

# # Load the index and documents

# index = faiss.read_index("faiss_index_java.index")
# with open("documents_java.pkl", "rb") as f:
#     documents = pickle.load(f)

# # Initialize RAG
# rag = RAG(index_path="faiss_index_java.index", documents=documents)

# # Example usage
# diff = "def add(a, b):\n    return a + b"
# context = rag.retrieve_similar_context(diff, k=3)
# print("Retrieved context:")
# print(context)
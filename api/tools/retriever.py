import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

class Retriever:
    def __init__(self, model_name="BAAI/bge-m3", folder_path="data"):
        """
        Initialize the Retriever with a model name and folder path.
        
        Args:
            model (str): SentenceTransformer model from Hugging Face. Ex: 'BAAI/bge-m3'
            folder_path (str): Path to store the vector database and metadata.
        """
        self.model = SentenceTransformer(model_name)
        self.folder_path = folder_path
        self.index_file = os.path.join(folder_path, "faiss_index.bin")
        self.metadata_file = os.path.join(folder_path, "metadata.pkl")
        self.index = None
        self.metadata = []
        
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

    def encode(self, json_path):
        """
        Encode questions from a JSON file and store them in a FAISS index.
        
        Args:
            json_path (str): Path to the JSON file containing a list of dictionaries.
        """
        # Check if index and metadata already exist
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print("Loading existing index and metadata...")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            return

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions and store metadata
        questions = []
        self.metadata = []
        for item in data:
            if 'question' in item and isinstance(item['question'], str):
                questions.append(item['question'])
                self.metadata.append(item)
        
        if not questions:
            raise ValueError("No valid questions found in the JSON file.")

        # Encode questions
        print("Encoding questions...")
        embeddings = self.model.encode(questions, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("Index and metadata saved.")

    def retrieve(self, query, threshold=0.8, top_k=3):
        """
        Retrieve top_k most similar questions based on a query, with a similarity threshold.
        
        Args:
            query (str): The query string to search for.
            threshold (float): Minimum similarity score (cosine similarity converted from L2).
            top_k (int): Maximum number of results to return.
        
        Returns:
            list: List of dictionaries corresponding to the matched questions.
        """
        if self.index is None or not self.metadata:
            raise ValueError("Index not initialized. Run encode() first.")
        
        # Encode the query
        query_embedding = self.model.encode([query])[0]
        
        # Search in the FAISS index
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        # Convert L2 distances to cosine similarity and filter by threshold
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            # Convert L2 distance to cosine similarity
            l2_dist = dist
            cosine_sim = 1 - (l2_dist / 2)  # Approximation for normalized vectors
            if cosine_sim >= threshold:
                results.append(self.metadata[idx])
        
        return results[:top_k]

if __name__ == "__main__":
    # Example usage
    model = SentenceTransformer("all-MiniLM-L6-v2")
    retriever = Retriever(model=model, folder_path= "./vector_db")
    
    # Example JSON file
    sample_json = [
        {"question": "What is Python?", "answer": "A programming language"},
        {"question": "What is Java?", "answer": "Another programming language"},
        {"question": "What is AI?", "answer": "Artificial Intelligence"}
    ]
    with open("sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_json, f)
    
    # Encode the JSON file
    retriever.encode("sample.json")
    
    # Retrieve similar questions
    query = "What is a programming language?"
    results = retriever.retrieve(query, threshold=0.5, top_k=2)
    for result in results:
        print(result)
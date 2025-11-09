"""
Create external datastore for three types of question:
  1. yes/no/uncertain question.
  2. multiple choice question.
  3. How many + chained question ( challenging case )  
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import json
from sklearn.cluster import KMeans

embedding_model_name = "BAAI/bge-m3"
train_ys_path = "./../train_yn.json"
train_choice_path = "./../train_choice.json"
train_hm_path = "./../train_hm.json"

n_clusters = 25 # Don't set value greater than record number of minimize set  

class Cluster:
    def __init__(self):
        pass
    
    def clustering(self, questions, embedding_model, n_clusters):
        embeddings = embedding_model.encode(questions, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(normalized_embeddings)
        return kmeans.cluster_centers_

class Indexer:
    embedding_model: SentenceTransformer
    clusterer: Cluster
    def __init__(self, model):
        self.embedding_model = model
        self.clusterer = Cluster()
        pass

    def create_index(self, questions, clustering: bool = False, n_clusters: int = None):
        if clustering == True:
            embeddings = self.clusterer.clustering(questions=questions, embedding_model= self.embedding_model, n_clusters=n_clusters)
        else:
            question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)
            question_embeddings = question_embeddings.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize
        
        dimension = embeddings.shape[1]
        index_flat = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index_flat.add(embeddings)
        return index_flat


if __name__ == '__main__':
    embedding_model = SentenceTransformer(embedding_model_name)
    indexer = Indexer(model=embedding_model)
    # Load questions
    with open(train_ys_path, 'r', encoding='utf-8') as f:
        yn_ds = json.load(f)
    yn_questions = [item['question'] for item in yn_ds]

    with open(train_choice_path, 'r', encoding='utf-8') as f:
        choice_ds = json.load(f)
    choice_questions = [item['question'] for item in choice_ds]

    with open(train_hm_path, 'r', encoding='utf-8') as f:
        hm_ds = json.load(f)
    hm_questions = [item['question'] for item in hm_ds]

    # Create faiss index
    yn_index = indexer.create_index(yn_questions, True, n_clusters=n_clusters)
    faiss.write_index(yn_index, "./yn_index")

    choice_index = indexer.create_index(choice_questions, True, n_clusters=n_clusters)
    faiss.write_index(choice_index, "./choice_index")

    hm_index = indexer.create_index(hm_questions, True, n_clusters=n_clusters)
    faiss.write_index(hm_index, "./hm_index")
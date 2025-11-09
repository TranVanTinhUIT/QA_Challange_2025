import faiss
import numpy as np
import os
import time
import json
from sentence_transformers import SentenceTransformer

class SimilarClassification:
  """
  Classify question into conceptual question or computational question based on their external datastore.
  """

  yn_index: any
  choice_index: any
  hm_index: any
  def __init__(self, yn_index_path, choice_index_path, hm_index_path):
    self.yn_index = faiss.read_index(yn_index_path)
    self.choice_index = faiss.read_index(choice_index_path)
    self.hm_index = faiss.read_index(hm_index_path)
    pass


  def classify(self, question, embedding_model, k = 10, trace = False):
    """
    Clasify the providing question into one of three types:

    Return:
      - 1 for yes/no/uncertain 
      - 2 for multiple choice question
      - 3 for how many + chained question
    """
    question_embeddings = embedding_model.encode([question], convert_to_tensor=True)
    question_embeddings = question_embeddings.cpu().numpy()
    question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True) # normalize
    faiss.normalize_L2(question_embeddings)

    yn_distances, yn_indices = self.yn_index.search(question_embeddings, k)
    choice_distances, choice_indices = self.choice_index.search(question_embeddings, k)
    hm_distances, hm_indices = self.hm_index.search(question_embeddings, k)
    yn_score = np.sum(yn_distances[0])
    choice_score = np.sum(choice_distances[0])
    hm_score = np.sum(hm_distances[0])

    scores = [yn_score, choice_score, hm_score]
    max = np.max(scores)

    if max == yn_score:
      return 1, scores
    elif max == choice_score:
      return 2, scores
    else:
      return 3, scores
    
# if __name__ == '__main__':
#   model = SentenceTransformer("BAAI/bge-m3")
#   k = 10

#   yn_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datastore", "yn_index")
#   choice_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datastore", "choice_index")
#   hm_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datastore", "hm_index")
#   classifier = SimilarClassification(yn_index_path=yn_index_path, choice_index_path=choice_index_path, hm_index_path=hm_index_path)

#   test_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_choice.json")

#   with open(test_file_path, 'r', encoding='utf-8') as f:
#       test_ds = json.load(f)

#   i = 0
#   answers = []
#   for sample in test_ds:
#     print(f'Sample {i}...')
#     answer = dict(sample)
#     question = sample.get('questions', '')
#     start_time = time.perf_counter()
#     question_type, scores = classifier.classify(question=question, embedding_model=model, k=k)
#     end_time = time.perf_counter()
#     answer['q_type'] = question_type
#     answer['q_type_time'] = end_time - start_time
#     answers.append(answer)
#     i+=1
  
#   output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_ds_question_type.json")
#   with open(output_path, 'w', encoding='utf-8') as f:
#       json.dump(answers, f, indent=2, ensure_ascii=False)


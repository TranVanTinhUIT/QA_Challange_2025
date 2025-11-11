import json
import numpy as np
from sklearn.metrics import f1_score
import os

# Cal Precision, Recall, F1 for answer, idx
def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_f1_score(true_indices, pred_indices):
    """Calculate F1 score between true and predicted indices."""

    true_indices = np.array(true_indices)
    pred_indices = np.array(pred_indices)
    
    max_idx = max(max(true_indices) if true_indices.size > 0 else 0, 
                  max(pred_indices) if pred_indices.size > 0 else 0)
    
    if max_idx == 0:
        return 1.0 if len(true_indices) == 0 and len(pred_indices) == 0 else 0.0
    
    true_binary = np.zeros(max_idx + 1, dtype=int)
    pred_binary = np.zeros(max_idx + 1, dtype=int)
    
    for idx in true_indices:
        true_binary[idx] = 1
    
    for idx in pred_indices:
        pred_binary[idx] = 1
    
    return f1_score(true_binary, pred_binary)

def evaluate_predictions(file_path, eval_out_file_path):
    """Evaluate predictions against reference data."""
    ds = load_json_file(file_path)
    answer_scores = []
    idx_scores = []
    final_scores = []
    
    for i, item in enumerate(ds):
       
        answer_score = 1.0 if item['ref_answer'] == item['pred_answer'] else 0.0
        answer_scores.append(answer_score)

        idx_score = calculate_f1_score(item['ref_index'], item['pred_idx'])
        idx_scores.append(idx_score)
        
        final_score = answer_score * 0.6 + idx_score * 0.4
        final_scores.append(final_score)
    
    avg_answer_score = np.mean(answer_scores)
    avg_idx_score = np.mean(idx_scores)
    avg_final_score = np.mean(final_scores)
    
    out_json = {
        'answer_score': avg_answer_score,
        'idx_score': avg_idx_score,
        'final_score': avg_final_score,
        'details': list(zip(answer_scores, idx_scores, final_scores))
    }
    with open(eval_out_file_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'
    eval_out_folder = os.path.dirname(os.path.abspath(__file__)) + '/eval_out'

    # evaluate_predictions(file_path = out_folder + '/output_yn_no_CoT.json', eval_out_file_path=eval_out_folder + '/yn_no_CoT_eval_answer_idx.json')
    # evaluate_predictions(file_path = out_folder + '/output_choice_no_CoT.json', eval_out_file_path=eval_out_folder + '/choice_no_CoT_eval_answer_idx.json')
    evaluate_predictions(file_path = out_folder + '/output_choice_no_CoT_new.json', eval_out_file_path=eval_out_folder + '/choice_no_CoT_new_eval_answer_idx.json')
    # evaluate_predictions(file_path = out_folder + '/output_choice_pipeline.json', eval_out_file_path=eval_out_folder + '/choice_pileline_eval_answer_idx.json')
    evaluate_predictions(file_path = out_folder + '/output_choice_pipeline_new.json', eval_out_file_path=eval_out_folder + '/choice_pipeline_new_eval_answer_idx.json')
    # evaluate_predictions(file_path = out_folder + '/output_hm_no_CoT.json', eval_out_file_path=eval_out_folder + '/hm_no_CoT_eval_answer_idx.json')
    # evaluate_predictions(file_path = out_folder + '/output_hm_no_Retrieval.json', eval_out_file_path=eval_out_folder + '/hm_no_Retrieval_eval_answer_idx.json')
    # evaluate_predictions(file_path = out_folder + '/output_no_classify.json', eval_out_file_path=eval_out_folder + '/no_classify_eval_answer_idx.json')

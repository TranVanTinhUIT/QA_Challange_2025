import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
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

def calculate_metrics(true_indices, pred_indices):
    """Calculate precision, recall, and F1 score between true and predicted indices."""
    true_indices = np.sort(true_indices)
    pred_indices = np.sort(pred_indices)

    max_idx = max(max(true_indices) if true_indices.size > 0 else 0, 
                  max(pred_indices) if pred_indices.size > 0 else 0)
    
    if max_idx == 0:
        if len(true_indices) == 0 and len(pred_indices) == 0:
            return 1.0, 1.0, 1.0  
        return 0.0, 0.0, 0.0
    
    true_binary = np.zeros(max_idx + 1, dtype=int)
    pred_binary = np.zeros(max_idx + 1, dtype=int)
    
    for idx in true_indices:
        true_binary[idx] = 1
    
    for idx in pred_indices:
        pred_binary[idx] = 1
    
    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)
    
    return precision, recall, f1


def evaluate_predictions(file_path, eval_out_file_path, max_items=None):
    """Evaluate predictions against reference data."""
    ds = load_json_file(file_path)
    if max_items is not None:
        ds = ds[:max_items]
    answer_scores = []
    idx_precision_scores = []
    idx_recall_scores = []
    idx_f1_scores = []
    final_scores = []
    times = []
    for i, item in enumerate(ds):
       
        answer_score = 1.0 if item['ref_answer'] == item['pred_answer'] else 0.0
        answer_scores.append(answer_score)

        precision, recall, f1 = calculate_metrics(item['ref_index'], item['pred_idx'])
        idx_precision_scores.append(precision)
        idx_recall_scores.append(recall)
        idx_f1_scores.append(f1)
        
        final_score = answer_score * 0.6 + f1 * 0.4  # Still using F1 for final score
        final_scores.append(final_score)
        times.append(item['time'])
    
    avg_answer_score = np.mean(answer_scores)
    avg_idx_precision = np.mean(idx_precision_scores)
    avg_idx_recall = np.mean(idx_recall_scores)
    avg_idx_f1 = np.mean(idx_f1_scores)
    avg_final_score = np.mean(final_scores)
    agv_time = np.mean(times)
    
    out_json = {
        'answer_score': avg_answer_score,
        'precision_idx_score': avg_idx_precision,
        'recall_idx_score': avg_idx_recall,
        'f1_idx_score': avg_idx_f1,
        'final_score': avg_final_score,
        'agv_time': agv_time,
        'details': list(zip(answer_scores, idx_precision_scores, idx_recall_scores, idx_f1_scores, final_scores))
    }
    with open(eval_out_file_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'
    eval_out_folder = os.path.dirname(os.path.abspath(__file__)) + '/eval_out'
    max_items = None

    evaluate_predictions(max_items=max_items, file_path = out_folder + '/yes_no_pipeline_final.json', eval_out_file_path=eval_out_folder + '/yes_no_pipeline_final_eval.json')
    evaluate_predictions(max_items=max_items, file_path = out_folder + '/yes_no_pipeline_final_ablation_analyze.json', eval_out_file_path=eval_out_folder + '/yes_no_pipeline_final_ablation_analyze_eval.json')
    
    evaluate_predictions(max_items=max_items, file_path = out_folder + '/choice_pipeline_final.json', eval_out_file_path=eval_out_folder + '/choice_pipeline_final_eval.json')
    evaluate_predictions(max_items=max_items, file_path = out_folder + '/choice_pipeline_final_ablation_step1.json', eval_out_file_path=eval_out_folder + '/choice_pipeline_final_ablation_step1_eval.json')
    
    evaluate_predictions(file_path = out_folder + '/hm_pipeline_final.json', eval_out_file_path=eval_out_folder + '/hm_pipeline_final_eval.json')
    evaluate_predictions(file_path = out_folder + '/hm_pipeline_final_empty_retrieval.json', eval_out_file_path=eval_out_folder + '/hm_pipeline_final_empty_retrieval_eval.json')
    

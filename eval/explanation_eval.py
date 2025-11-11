import os
import json
from rouge_score import rouge_scorer
import numpy as np

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
def explanation_val(file_path, eval_out_file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    scores =[]

    for index, item in enumerate(ds):
        ref = item['ref_explanation']
        pred = item['pred_explanation']
        score = scorer.score(ref, pred)
        rouge1 = score['rouge1'].fmeasure
        rouge2 = score['rouge2'].fmeasure
        rougeL = score['rougeL'].fmeasure
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougel_scores.append(rougeL)
        scores.append({ 'rouge1': rouge1, 'rouge2': rouge2, 'rougel': rougeL})

    mean_rouge1 = np.mean(rouge1_scores)
    mean_rouge2 = np.mean(rouge2_scores)
    mean_rougeL = np.mean(rougel_scores)
    out_json = {
        'rouge1': mean_rouge1,
        'rouge2': mean_rouge2,
        'rougel': mean_rougeL,
        'details': scores
    }
    with open(eval_out_file_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'
    eval_out_folder = os.path.dirname(os.path.abspath(__file__)) + '/eval_out'

    # explanation_val(file_path = out_folder + '/output_yn_no_CoT.json', eval_out_file_path=eval_out_folder + '/yn_no_CoT_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_choice_no_CoT.json', eval_out_file_path=eval_out_folder + '/choice_no_CoT_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_choice_no_CoT_new.json', eval_out_file_path=eval_out_folder + '/choice_no_CoT_new_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_choice_pipeline.json', eval_out_file_path=eval_out_folder + '/choice_pipeline_eval_explanation.json')
    explanation_val(file_path = out_folder + '/choice_pipeline.json', eval_out_file_path=eval_out_folder + '/choice_pipeline_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_hm_no_CoT.json', eval_out_file_path=eval_out_folder + '/hm_no_CoT_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_hm_no_Retrieval.json', eval_out_file_path=eval_out_folder + '/hm_no_Retrieval_eval_explanation.json')
    # explanation_val(file_path = out_folder + '/output_no_classify.json', eval_out_file_path=eval_out_folder + '/no_classify_eval_explanation.json')

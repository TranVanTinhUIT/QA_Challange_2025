import os
import json
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from hm_pipeline3 import HmPipeline3  

def get_model(model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


# THIS FILE TEST ablation study without Classification module (this as test only How many pipeline)
if __name__ == "__main__":
    embedding_model = SentenceTransformer('BAAI/bge-m3')
    hm_pipeline = HmPipeline3(embedding_model=embedding_model) 
    tokenizer, model = get_model() # Model
    
    ds_folder = os.path.dirname(os.path.abspath(__file__)) + '/../dataset'
    out_folder = os.path.dirname(os.path.abspath(__file__)) + '/../out'

    with open(ds_folder + '/choice_test.json', "r", encoding="utf-8") as f:
        choice_test = json.load(f)
    with open(ds_folder + '/yn_test.json', "r", encoding="utf-8") as f:
        yn_test = json.load(f)
    with open(ds_folder + '/hm_test.json', "r", encoding="utf-8") as f:
        hm_test = json.load(f)

    test_ds = choice_test + yn_test + hm_test
    print(len(test_ds))

    results = []

    out_file = out_folder + f"/output_no_classify.json"
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
    for index, item in enumerate(test_ds):
        if index < len(results):
            continue
        print('Test item ', index)
        if index %50 == 0 :
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        premises = item['premises-NL']
        question = item['question']
        
        hm_result = hm_pipeline.run(premises, question, tokenizer, model, trace= False)

        result = {
            'premises': premises,
            'question': question
        }
        
        result['ref_answer'] = item['answer'],
        result['ref_index'] = item['idx']
        result['ref_explanation'] = item['explanation']
        result['pred_answer'] = hm_result['answer']
        result['pred_idx'] = hm_result['idx']
        result['pred_explanation'] = hm_result['explanation']
        results.append(result)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)